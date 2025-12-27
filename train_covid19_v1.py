import os
import json
import torch

from params import parse_args
from util import data_loader_covid19_v1 as data_loader
from model.any_covid19_v1 import AnyDiffuion
from util.logger import setuplogger, save_model, load_model, save_graph
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from model import utils

seed=0
np.random.seed(seed)
torch.manual_seed(seed)

class Exp:
    def __init__(self, args_obj): # 接收args对象
        self.args = args_obj
        self.map_configs_path = './configs/map_covid19_DAGT.json'
        if os.path.exists(self.map_configs_path):
            with open(self.map_configs_path) as f:
                self.map_configs = json.load(f)
        else:
            print(f"Config file not found: {self.map_configs_path}")

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None 

        self.model = None
        self.ddp_model = None
        self.backbone_optim = None
        self.prompt_optim = None
        self.edge_optim = None
        self.backbone_scheduler = None
        self.prompt_scheduler = None
        self.edge_scheduler = None

        self.device = f'cuda:3' if torch.cuda.is_available() else 'cpu'
        self.load_model_path = self.args.load_model_path

        self.logger = setuplogger(args_obj)
        self.logger.info(f"args:{self.args}")


    def _prepare_data_and_loaders(self):

        if self.train_dataset is None: 
            train_loader = data_loader.load_data(self.map_configs, self.args)
            self.train_loader = train_loader

        self.logger.info(f"DataLoaders created. Train batches: {len(self.train_loader)}")

    def _prepare_model_and_optimizers(self, tryload=False):
        self.model = AnyDiffuion(self.map_configs, self.args).to(self.device)
        self.model.device = self.device
        # --- 2. 收集 Graph 结构参数 (高 LR, 0 Decay) ---
        g_params = list(self.model.graph_learners.parameters())
        if self.args.use_prompt:
            p_params = list(self.model.prompt_learner.parameters())
            special_ids = set(map(id, g_params + p_params))
        else:
            special_ids = set(map(id, g_params))
            p_params = []
        base_params = [p for p in self.model.parameters() if id(p) not in special_ids]

        # --- 4. 定义优化器 ---
        self.optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': self.args.lr},
            {'params': p_params, 'lr': self.args.lr},
            {'params': g_params, 'lr': self.args.lr_z},
        ])
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.5
        )   
        self.logger.info("Model, optimizers, and DDP wrapper prepared.")


    def train_one_epoch(self, epoch_index):
        self.model.train()
        
        total_batches = len(self.train_loader)
        epoch_avg_loss = 0.0 # 用于记录整个epoch的平均外部循环损失
        epoch_avg_mse = 0.0
        epoch_avg_nll = 0.0

        batch_index = 0
        for batched_data_dict in self.train_loader:

            losses= self.model(batched_data_dict)
            loss = losses['loss']
            if torch.isnan(loss):
                print(f"Epoch {epoch_index} Batch {batch_index}: Outer loss NaN")
                batch_index += 1
                continue
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()

            epoch_avg_loss += loss.item()
            
            mse = losses['loss_mse']
            epoch_avg_mse += mse.item()

            nll_sum = losses['nll']
            epoch_avg_nll += nll_sum.item()
            batch_index += 1
        self.scheduler.step()

        final_epoch_avg_loss = epoch_avg_loss / total_batches if total_batches > 0 else 0.0
        final_epoch_avg_mse = epoch_avg_mse / total_batches if total_batches > 0 else 0.0
        final_epoch_avg_nll = epoch_avg_nll / total_batches if total_batches > 0 else 0.0

        return final_epoch_avg_loss, final_epoch_avg_mse, final_epoch_avg_nll

    def eval_net(self, save=False):
        """
        Evaluate similarity between learned graphs and GT graphs
        and return 2D results:
            tot_list[map_id][disease_id]
            tot_top_list[map_id][disease_id]
        """
        device = next(self.model.parameters()).device

        num_maps = self.model.map_num
        num_diseases = self.args.disease_num
        
        map_configs_sorted = sorted(self.map_configs, key=lambda x: x['map_index'])
        map_names = [mc['map_name'] for mc in map_configs_sorted]
        name_pearson_dict = {name: [] for name in map_names}

        # Output as 2D lists
        tot_list = [[None for _ in range(num_maps)] for _ in range(num_diseases)]
        tot_top_list = [[None for _ in range(num_maps)] for _ in range(num_diseases)]

        disease_ids = torch.arange(num_diseases, device=device)
        self.model.eval()

        with torch.no_grad():
            # 1. Learned adjacency for each map (batched over diseases)
            learned_all_maps = [
                self.model.graph_learners[map_id](disease_ids)
                for map_id in range(num_maps)
            ]

            # 2. Compute metrics
            for map_id, learned_map in enumerate(learned_all_maps):
                gt_map = self.model.edge_GT_list[map_id]   # list/array of GT graphs

                for disease_id in range(num_diseases):
                    g_learned = learned_map[disease_id]
                    g_gt = gt_map[disease_id]
                    g_gt = g_gt.T

                    K = int((g_gt > 0).sum())

                    pearson, pearson_topk = utils.cal_accuracy_adj_v2(
                        g_gt,
                        g_learned.cpu(),
                        K=K
                    )
                    name_pearson_dict[map_names[map_id]].append(pearson)
                    name_pearson_dict[map_names[map_id]].append(pearson_topk)

                    tot_list[disease_id][map_id] = pearson
                    tot_top_list[disease_id][map_id] = pearson_topk
            if save:
                save_graph(self.args, map_configs_sorted, learned_all_maps, self.logger)

        return tot_list, tot_top_list, name_pearson_dict

    def run(self): # args 现在是 self.args

        self._prepare_data_and_loaders()
        self._prepare_model_and_optimizers()

        self.epoch_training_losses = torch.zeros(self.args.train_epoch, device='cpu') # Rank 0记录
        
        best_train_loss = float('inf')
        best_train_mse = float('inf')
        
        for epoch_idx in range(self.args.train_epoch):

            avg_train_loss_epoch, avg_train_mse_epoch, avg_train_nll_epoch = self.train_one_epoch(epoch_idx)

            save_graph = False
            if avg_train_loss_epoch < best_train_loss or avg_train_mse_epoch < best_train_mse:
                save_graph = True
            tot_list, tot_top_list, name_pearson_dict = self.eval_net(save_graph)
            self.epoch_training_losses[epoch_idx] = avg_train_loss_epoch

            self.logger.info(f"--- V2 Epoch {epoch_idx+1} Summary ---")
            self.logger.info(f"  Avg Training Outer Loss: {avg_train_loss_epoch:.6f}")
            self.logger.info(f"  Avg Training Outer Mse: {avg_train_mse_epoch:.6f}")
            self.logger.info(f"  Avg Training Outer Nll: {avg_train_nll_epoch:.6f}")

            self.logger.info(f"  Edge tot: {np.array2string(np.array(tot_list))}")
            self.logger.info(f"  Top Edge tot: {np.array2string(np.array(tot_top_list))}")
            self.logger.info(f"  Edge details: {name_pearson_dict}")
            
            if avg_train_loss_epoch < best_train_loss or avg_train_mse_epoch < best_train_mse:
                best_train_loss = avg_train_loss_epoch
                best_train_mse = avg_train_mse_epoch
                save_model(epoch_idx, self.model, self.args,self.logger)
                self.logger.info(f"  Best model saved at epoch {epoch_idx+1} with loss {best_train_loss:.6f}")

# 修改全局的 train 函数作为 mp.spawn 的目标
def spawn_train_process(args_obj_for_exp): # 接收args对象
    exp_instance = Exp(args_obj_for_exp) # 每个进程创建自己的Exp实例
    exp_instance.run()


if __name__ == '__main__':
    
    args = parse_args(config_name='config_covid19_v1') # 解析命令行参数并加载配置文件
    if not hasattr(args, 'pwd'): args.pwd = os.getcwd() # 获取当前工作目录
    
    spawn_train_process(args)

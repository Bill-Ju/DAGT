import torch
import torch.nn.functional as F
import torch.nn as nn


from model import utils
import numpy as np
from collections import defaultdict

import os


from model.decoder_covid19_v1 import Decoder

class AnyDiffuion(nn.Module):
    def __init__(self, map_configs, args):
        super(AnyDiffuion, self).__init__()
        
        self.disease_num = args.disease_num
        
        self.set_map(map_configs, args)
        
        self.backbone = Decoder(args)


    def set_map(self, map_configs, args):
        self.use_prompt = args.use_prompt
        self.node_num_list = []
        self.edge_index_list = []
        self.edge_mask_list = []
        self.node_mask_list = []
        self.edge_GT_list = []
        self.g_list = nn.ParameterList()
        self.graph_learners = nn.ModuleList() 
        
        self.map_num = len(map_configs)

        map_configs_sorted = sorted(map_configs, key=lambda x: x['map_index'])

        for config in map_configs_sorted:
            map_index = config['map_index']
            map_name = config['map_name']
            loc_num = config['loc_num'] 
            data_dir = os.path.join(args.data_dir, map_name)

            edge_file = args.edge_file
            edge_GTs = np.load(os.path.join(data_dir, edge_file))
            
            A_GTs = torch.zeros(self.disease_num, loc_num, loc_num)
            for i in range(self.disease_num):
                A_GT = torch.tensor(edge_GTs)
                A_GTs[i] = A_GT
                           
            self.edge_GT_list.append(A_GTs)
            
            self.graph_learners.append(
                IndependentGraphGenerator(loc_num, self.disease_num)
            )            
            self.node_num_list.append(loc_num)

        print(f"Loaded {self.map_num} maps with configurations: {map_configs_sorted}")
    

    def forward(self, batched_data_dict):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        gamma, beta = None, None
        for dataset_id_str, batch_data in batched_data_dict.items():
            dataset_id = int(dataset_id_str.split('_')[1])
            batch_data = batch_data.to(self.device)
        
            cas = batch_data[:, :,0:-1,:]
            disease_id = batch_data[:, 0, -1, 0].to(torch.long)
            
            input = cas[..., :-1]
            target = cas[...,-1]
            node_num = self.node_num_list[dataset_id]

            g = self.graph_learners[dataset_id](disease_id)
            
            preds = self.backbone(input, g, gamma, beta)
            loss =  self.compute_loss(preds, target, g, node_num, loss_type='mix')
            losses["loss"] += loss
            losses["loss_mse"] += self.compute_loss(preds, target, loss_type='mse')
            losses["nll"] += self.compute_loss(preds, target, loss_type='nll')
        losses["loss"] /= len(batched_data_dict)
        losses["loss_mse"] /= len(batched_data_dict)
        losses["nll"] /= len(batched_data_dict)
        return losses

    def compute_loss(self, preds, targets, g=None, loss_type='nll'):

        if loss_type=='nll':
            loss_nll = utils.nll_gaussian(preds, targets)
            loss = loss_nll
        elif loss_type=='mix':
            loss_nll = utils.nll_gaussian(preds, targets)
            loss_sparsity = torch.norm(g, p=1) 
            loss = loss_nll + 0.01* loss_sparsity
        else:
            loss = F.mse_loss(preds,targets)
        return loss


class IndependentGraphGenerator(nn.Module):
    def __init__(self, num_nodes, num_diseases, init_scale=0.01):
        super(IndependentGraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        
        # 直接定义 (10, N, N) 的参数张量
        # 每一个 disease_id 对应一个完全独立的 N*N 矩阵
        self.adjs = nn.Parameter(torch.randn(num_diseases, num_nodes, num_nodes) * init_scale)
        
        # 注册 Buffer (不训练)
        self.register_buffer('identity', torch.eye(num_nodes))

    def forward(self, disease_ids, threshold=None):
        """
        Input: disease_ids (B,)
        Output: (B, N, N)
        """
        # 1. 查表 (Lookup)
        # 这一步非常快，直接取出当前 Batch 对应的那些图
        logits = self.adjs[disease_ids]
        
        # 2. 训练时加噪声 (依然建议保留，为了鲁棒性)
        if self.training:
            # 这里的噪声是为了防止落入局部最优
            noise = torch.rand_like(logits) # 或者用 gumbel
            noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10) # Gumbel
            logits = logits + 0.5 * noise 
            
        # 3. 激活
        g = F.softplus(logits)
        
        # 4. 去除自环
        g = g * (1 - self.identity)
        
        # 5. 推理截断
        if not self.training and threshold is not None:
            g = torch.where(g < threshold, torch.zeros_like(g), g)
            
        return g
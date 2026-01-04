import torch
from torch.optim import lr_scheduler
from util.utils import *
import argparse
from model.model_trans_v1 import Decoder,get_edge_prob
import numpy as np
import random
import time

parser = argparse.ArgumentParser()
# data
parser.add_argument('--suffix', type=str, default='MM_BA50_exp0', help='suffix for data')

parser.add_argument('--tr_num', type=int, default=50,
    help='No. of training trajectories, using all trajectories when None')

parser.add_argument('--va_num', type=int, default=30,
    help='No. of validation trajectories, using all trajectories when None')

parser.add_argument('--te_num', type=int, default=10,
    help='No. of test trajectories, using all trajectories when None')

parser.add_argument('--sample_freq', type=int, default=1,
    help='Sampling frequency of the trajectory')

parser.add_argument('--trajr_length', type=int, default=10,
    help='No. of time stamps in each trajectory, using all data when None')

parser.add_argument('--interlacing', type=bool, default=True,
    help='If the trajectories are interlacing when preparing dataset')

parser.add_argument('--Tstep', type=int, default=2, help='No. of steps for batched trajectories')

# model
parser.add_argument('--skip_first_edge_type', type=bool, default=False,
    help='If skip non-edges')
parser.add_argument('--dual_attention', type=bool, default=False,
    help='If use dual attention mechanism')
parser.add_argument('--gate', type=bool, default=False,
    help='If use gating mechanism')
parser.add_argument('--gumbel_noise', type=bool, default=True,help='If includes noise')
parser.add_argument('--init_logits', type=str, default='random',
    help='initialization of logtis, (uniform, random)')
parser.add_argument('--hidden_channels', type=int, default=512)

# training
parser.add_argument('--train', type=bool, default=True,
    help='If False, use test time adaption with a trained model')
parser.add_argument('--lr', type=float, default=0.0005, 
    help="Initial learning rate.")
parser.add_argument('--lr_z', type=float, default=0.1, 
    help="Learning rate for distribution estimation.")
parser.add_argument('--lr_logits', type=float, default=0.1,
    help="Learing rate for test time adaption")
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_tta_steps', type=int, default=100)
parser.add_argument('--lr_decay', type=int, default=200,help="After how epochs to decay LR by a factor of gamma.",)
parser.add_argument('--seed', type=int, default=0,help="random seed")
parser.add_argument('--num_head', type=int, default=4)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
args = parser.parse_args()
print(f"dual_attention:{args.dual_attention}")


seed=args.seed
if seed is None:
    seed=random.randint(100,10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:'+str(args.device_id) if torch.cuda.is_available() else 'cpu')
print('Device:', device)


if 'netsim' in args.suffix:
    x_tr, x_va, x_te, A = load_netsim_data(args)
else:
    x_tr, x_va, x_te, A = load_data(args) # loaded data has shape [batch, nodes, variables, time]

print(x_tr.shape,x_va.shape,x_te.shape)
def is_undirected_float(A, tolerance=1e-8):
    if A.shape[0] != A.shape[1]:
        return False
    # Use np.allclose to check floating-point symmetry
    return np.allclose(A, A.T, atol=tolerance)
print(f"A_weighted 是无向图吗? {is_undirected_float(A)}")
Tstep = args.Tstep
batch_size = args.batch_size
epochs = args.num_epoch
lr = args.lr
num_nodes = A.shape[0]
num_variables = x_tr.size(2)
num_edges = int(A.sum())
interlacing = args.interlacing
args.in_channels = num_variables

train_loader = torch.utils.data.DataLoader(TrajrData(x_tr,Tstep,interlacing), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(TrajrData(x_va,Tstep,interlacing), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(TrajrData(x_te,Tstep,interlacing), batch_size=batch_size, shuffle=False)

def train():

    def train_epoch(data_loader):
        model.train()
        loss_batch = 0
        num_datum = 0
        for x in data_loader:
            x = x.to(device)
            xpred = model(x[...,:-1],logits)
            edge_prob = get_edge_prob(logits, gumbel_noise=args.gumbel_noise, beta=1.0)

            kl_loss = kl_categorical_uniform(edge_prob, num_nodes)
            nll_loss = nll_gaussian(xpred,x[...,1:])
            
            
            loss = nll_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mse = criterion(x[...,1:], xpred)
            num_datum += xpred.numel()
            loss_batch += loss_mse.item()
        scheduler.step()
        loss_batch = loss_batch / num_datum
        return loss_batch

    def test_epoch(data_loader):
        model.eval()
        loss_batch = 0
        num_datum = 0
        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                xpred = model(x[...,:-1],logits)
                loss_mse = criterion(x[...,1:], xpred)
                loss_batch += loss_mse.item()
                num_datum += xpred.numel()
        loss_batch = loss_batch / num_datum
        return loss_batch

    optimizer = torch.optim.Adam(
            [{"params": logits, "lr": args.lr_z}] +
            [{"params": model.parameters(), "lr": args.lr}]
        )

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )
    data_path = './logs/nri' + args.suffix
    decoder_file = os.path.join(data_path+ "_decoder.pt")
    
    best_train_loss = np.inf
    best_val_loss = np.inf
    best_mes_from_va = 0
    best_mes_from_va = 0
    best_auc_from_va = 0
    best_acc_from_va = 0 
    epochs = 0
    patient=0
    start_time = time.perf_counter()
    for epoch in range(1, args.num_epoch+1):
        

        loss_tr = train_epoch(train_loader)
        if(epoch%1 == 0):
            loss_va = test_epoch(valid_loader)
            loss_te = test_epoch(test_loader)
            # edge_prob = get_edge_prob(logits,gumbel_noise=False, beta=1.0).clone().detach()
            g = logits.clone().detach()
            A_soft, A_hard = generate_prediction_adj(g)
            auc, acc, pre = cal_accuracy(A, A_soft, A_hard, num_edges)
            if loss_va < best_val_loss:
                best_train_loss= loss_tr
                best_val_loss = loss_va
                best_mes_from_va = loss_te
                best_auc_from_va = auc
                best_acc_from_va = acc
                
            print('Epoch: {:03d}'.format(epoch),
                'Train Loss: {:.8f}'.format(loss_tr),
                'Valid Loss: {:.8f}'.format(loss_va),
                'Picked AUC: {:.4f}'.format(best_auc_from_va),
                'Picked ACC: {:.4f}'.format(best_acc_from_va),
                'Current AUC: {:.4f}'.format(auc),
                'Current ACC: {:.4f}'.format(acc),
                'Current PRE: {:.4f}'.format(pre))
    end_time = time.perf_counter()
    epoch_duration = (end_time - start_time)/args.num_epoch
    print(f"Epoch duration: {epoch_duration:.4f} seconds)")
    return best_auc_from_va,best_acc_from_va,auc,best_train_loss,best_mes_from_va
    


if __name__ == '__main__':


    logits = torch.randn(torch.Size([2, num_nodes, num_nodes]),requires_grad=True,device=device)


    model = Decoder(args).to(device)
    criterion = torch.nn.MSELoss(reduction='sum')

    best_auc_from_va,best_acc_from_va,last_auc,best_train_loss,best_mes_from_va=train()
        
    import sys
    # Open a file for logging
    log_file = open('result/trans_'+args.suffix+'_'+str(args.sample_freq)+'_'+str(args.tr_num)+'_'+str(args.trajr_length)+'.txt', 'a')
    sys.stdout = log_file
    
    print(f'model:trans, seed:{seed}, auc:{best_auc_from_va}, acc:{best_acc_from_va}, last_auc:{last_auc}, train_loss:{best_train_loss}, test_loss:{best_mes_from_va}')
    sys.stdout = sys.__stdout__
    log_file.close()

            
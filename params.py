import argparse
import os
import yaml

def parse_args(config_name='config_demo'):
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_z', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--pwd', default='/home/zjy/project/AnyDiffusion', type=str, help='project base dir')
    parser.add_argument('--in_channels', default=1, type=int, help='information in node')
    parser.add_argument('--hidden_channels', default=128, type=int, help='model hidden layer width')
    parser.add_argument('--Tstep', default=2, type=int, help='rnn model pred steps')
    parser.add_argument('--burn_in_steps', default = 16, type=int, help='use pred after burn_in_steps')
    parser.add_argument('--train_batch', default=2, type=int, help='training batch size (number of cacade)')
    parser.add_argument('--test_batch', default=6, type=int, help='testing batch size (number of cacade)')
    parser.add_argument('--train_epoch', default=300, type=int, help='number of epochs')
    parser.add_argument('--test_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--load_model', default=False, help='if load model, True or False')
    parser.add_argument('--gpu', default='5', type=str, help='indicates which gpu to use')

    parser.add_argument('--eval_loss', default=True, type=bool, help='whether use CE loss to evaluate test performance')
    parser.add_argument('--direct', default=False, type=bool, help='whether use direct graph')
    parser.add_argument('--clamp', default=False, type=bool, help='whether clamp')
    parser.add_argument('--backbone', default='MLP', type=str, help='[RNN,MLP]')
    parser.add_argument('--train_size', default=0.6, type=float, help='train_size')
    parser.add_argument('--data_mode', default='mix', type=str, help='[lt,sir]')
    parser.add_argument('--disease_num', default=50, type=int, help='disease_num')
    parser.add_argument('--memory_size', default=16, type=int, help='promopt memory_net')
    parser.add_argument('--im_seed_rate', type=float, default=0.05,help='seed_rate')
    parser.add_argument('--im_mode', default='sir', type=str, help='[sir, ic]')
    parser.add_argument('--window_size', default=32, type=int, help='rnn computation graph length')
    parser.add_argument('--inner_update_steps', default=1, type=int, help='inner_update_steps')
    parser.add_argument('--test_inner_update_steps', default=5, type=int, help='inner_update_steps')
    parser.add_argument('--node_num', default=204, type=int, help='node_num')
    parser.add_argument('--num_fc', default=1, type=int, help='num_fc')
    parser.add_argument('--traj_file', default='traj_v1.npz', type=str, help='traj_file')
    parser.add_argument('--edge_file', default='edge_switch_v1.npz', type=str, help='edge_switch_v1')

    parser.add_argument('--data_dir', default='/home/lxx/disease_pred/disease_ACD/codebase/data/train_data_v11', type=str, help='Name of directory where data is stored.')
    parser.add_argument("--temp", default=0.5, type=float, help="Temperature for Gumbel softmax.")
    parser.add_argument("--hard", default=True, type=bool, help="Hard for Gumbel softmax.")
    parser.add_argument('--save_path', default='result', type=str, help='model save path')
    parser.add_argument('--load_model_path', default='train_data_v11_6_8/any_0.pth', type=str, help='model save path')
    parser.add_argument('--reinitialize_edge', default=True, type=bool, help='True or False')
    parser.add_argument('--pop_dim', default=5, type=int, help='num_fc')
    parser.add_argument('--risk_dim', default=85, type=int, help='num_fc')
    parser.add_argument('--sex_dim', default=3, type=int, help='num_fc')
    parser.add_argument('--age_dim', default=38, type=int, help='num_fc')
    parser.add_argument('--num_workers', default=1, type=int, help='num_fc')
    parser.add_argument('--port', default=29500, type=int, help='port')
    parser.add_argument('--decay_interval', default=10, type=int, help='decay_interval')
    parser.add_argument("--prior", default=0.06, type=float, help="prior")
    parser.add_argument("--final_r", default=0.002, type=float, help="final_r.")
    parser.add_argument("--init_r", default=0.1, type=float, help="final_r.")
    parser.add_argument("--decay_r", default=0.775, type=float, help="decay_r.")
    parser.add_argument("--lambda_switch", default=0.05, type=float, help="lambda_switch.")
    parser.add_argument('--loss_type', default='bic', type=str, help='loss_type')
    parser.add_argument('--switch_type', default='softmax', type=str, help='switch_type')
    parser.add_argument('--use_prompt', default=False, type=bool, help='use_prompt')
    parser.add_argument('--use_mask', default=False, type=bool, help='use_mask')
    parser.add_argument('--left', default=0, type=int, help='Left boundary')
    parser.add_argument('--right', default=34, type=int, help='right boundary')
    
    
    
    config_file_path = './configs/' + config_name + '.yaml'

    # ====== 在这里添加检查文件是否存在 ======
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            config_args = yaml.safe_load(f)
        parser.set_defaults(**config_args)
        print(f"Loaded configuration from {config_file_path}")
    else:
        print(f"Config file not found: {config_file_path}")
    
    return parser.parse_args()

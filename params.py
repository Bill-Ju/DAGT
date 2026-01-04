import argparse
import os
import yaml

def parse_args(config_name='config_covid19_v1'):
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_z', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--pwd', default='./DAGT', type=str, help='project base dir')
    parser.add_argument('--in_channels', default=1, type=int, help='information in node')
    parser.add_argument('--hidden_channels', default=128, type=int, help='model hidden layer width')
    parser.add_argument('--Tstep', default=2, type=int, help='rnn model pred steps')
    parser.add_argument('--train_batch', default=2, type=int, help='training batch size (number of cacade)')
    parser.add_argument('--test_batch', default=6, type=int, help='testing batch size (number of cacade)')
    parser.add_argument('--train_epoch', default=300, type=int, help='number of epochs')
    parser.add_argument('--test_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--load_model', default=False, help='if load model, True or False')
    parser.add_argument('--gpu', default='5', type=str, help='indicates which gpu to use')
    parser.add_argument('--train_size', default=0.6, type=float, help='train_size')
    parser.add_argument('--disease_num', default=50, type=int, help='disease_num')
    parser.add_argument('--node_num', default=204, type=int, help='node_num')
    parser.add_argument('--num_fc', default=1, type=int, help='num_fc')
    parser.add_argument('--traj_file', default='traj_v1.npz', type=str, help='traj_file')
    parser.add_argument('--edge_file', default='edge_switch_v1.npz', type=str, help='edge_switch_v1')

    parser.add_argument('--data_dir', default='./data/train_data_v11', type=str, help='Name of directory where data is stored.')
    parser.add_argument("--temp", default=0.5, type=float, help="Temperature for Gumbel softmax.")
    parser.add_argument("--hard", default=True, type=bool, help="Hard for Gumbel softmax.")
    parser.add_argument('--save_path', default='result', type=str, help='model save path')
    parser.add_argument('--num_head', default=4, type=int, help='num_head')
    parser.add_argument('--reinitialize_edge', default=True, type=bool, help='True or False')
    parser.add_argument('--loss_type', default='bic', type=str, help='loss_type')
    parser.add_argument('--left', default=0, type=int, help='Left boundary')
    parser.add_argument('--right', default=34, type=int, help='right boundary')
    config_file_path = './configs/' + config_name + '.yaml'

    # ====== Add file existence check here ======
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            config_args = yaml.safe_load(f)
        parser.set_defaults(**config_args)
        print(f"Loaded configuration from {config_file_path}")
    else:
        print(f"Config file not found: {config_file_path}")
    
    return parser.parse_args()

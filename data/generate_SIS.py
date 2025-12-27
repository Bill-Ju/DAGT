import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import argparse
import pickle
import pandas as pd


parser = argparse.ArgumentParser('Generate SIR simulation data')
parser.add_argument('--graph', type=str, default='BA')
parser.add_argument('--num-nodes', type=int, default=50,
                    help='Number of nodes in the simulation.')
parser.add_argument('--p', type=float, default=0.1, 
                    help='Connection/add connection probability In ER/NWS')
parser.add_argument('--k', type=int, default=2, 
                    help='Inital node degree in BA/NWS')

parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=200,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=30,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=30,
                    help='Number of test trajectories.')


parser.add_argument('--infection_rate', type=float, default=0.6)
parser.add_argument('--recovery_rate', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=300)


args = parser.parse_args()

beta = args.infection_rate
gamma = args.recovery_rate

def simulation(steps):
    seed = np.random.randint(0, n)
    # x = np.zeros(n)
    # x[seed] = 1.0
    # s = np.ones(n)
    # s[seed] = 0.0
    # r = np.zeros(n)
    x = np.random.rand(n)
    s = 1 -x
    r = np.zeros(n)
    x_trajr = x.reshape(1,-1)
    s_trajr = s.reshape(1,-1)

    for t in range(1,steps):
        infect = s*np.prod(1-beta*A*x,axis=1)
        recover = gamma * x
        x = x - recover + infect
        s = s - infect + recover
        x_trajr = np.concatenate([x_trajr,x.reshape(1,-1)],axis=0)
        s_trajr = np.concatenate([s_trajr,s.reshape(1,-1)],axis=0)

    x_trajr = np.expand_dims(x_trajr, axis = -1)
    s_trajr = np.expand_dims(s_trajr, axis = -1)  

    # x_trajr = np.concatenate((x_trajr,s_trajr,r_trajr),axis=-1) 
    # x_trajr = x_trajr   
    return x_trajr


# # iteration
# def simulation_sis(steps):
#     """
#     使用离散时间概率模型模拟网络上的 SIS 过程。
#     """
#     # --- 初始化 ---
#     # 随机初始化感染者比例
#     x = np.random.rand(n) * 0.1 # 初始感染比例较低
#     s = 1 - x

#     # 初始化轨迹记录
#     x_trajr = x.reshape(1, -1)
#     s_trajr = s.reshape(1, -1)


#     # --- 模拟主循环 ---
#     for t in range(1, steps):
#         infection_rate = 1 - np.exp(-beta * (A @ x))
#         infect = s * infection_rate
        
#         # 更新状态
#         x_new = x  + infect
#         s_new = s  - infect

#         # 保证比例在 [0, 1] 范围内
#         x = np.clip(x_new, 0, 1)
#         s = np.clip(s_new, 0, 1)

#         # 记录轨迹
#         x_trajr = np.concatenate([x_trajr, x.reshape(1, -1)], axis=0)
#         s_trajr = np.concatenate([s_trajr, s.reshape(1, -1)], axis=0)
        
#     # --- 格式化输出 ---
#     # 将 s 和 x 轨迹拼接在最后一个维度上
#     x_trajr = np.expand_dims(x_trajr, axis=-1)
#     s_trajr = np.expand_dims(s_trajr, axis=-1)
    
#     # 最终形状为 [steps, n, 2]，通道0是I(x)，通道1是S(s)
#     trajectory = np.concatenate((x_trajr, s_trajr), axis=-1)
    
#     return trajectory


if __name__ == '__main__':
    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'
    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k
        np.random.seed(exp_id)
        

        if args.graph in 'ER':
            G = nx.erdos_renyi_graph(n,p,seed=exp_id)
        elif args.graph in 'NWS':
            G = nx.newman_watts_strogatz_graph(n,k,p,seed=exp_id)
        elif args.graph in 'BA':
            G = nx.barabasi_albert_graph(n,k,seed=exp_id)

        A = nx.to_numpy_array(G)
        # print(A.sum())
        
        x_tr = np.zeros((args.tr_num,args.steps,n,1))
        for i in range(args.tr_num):
            print(f'Simulating training trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation(args.steps)
          
        x_va = np.zeros((args.va_num,args.steps,n,1))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            x_va[i] = simulation(args.steps)

        x_te = np.zeros((args.te_num,args.steps,n,1))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            x_te[i] = simulation(args.steps)

        A = nx.to_numpy_array(G)
        x_tr = x_tr.astype(np.float16)
        x_va = x_va.astype(np.float16)
        x_te = x_te.astype(np.float16)

        # save data, output data has shape [batch, time, nodes, variables]
        A = nx.to_numpy_array(G)
        result = [x_tr,x_va,x_te,A]
        data_path = 'SIS_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)


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
parser.add_argument('--steps', type=int, default=100)


args = parser.parse_args()

# beta = args.infection_rate
# gamma = args.recovery_rate
alpha = 0.22  # 本地感染率
beta = 0.2    # 本地恢复率
gamma = 0.1  # 迁移/扩散率

def simulation_metapop_sis(steps, P):
    """
    使用离散时间模型模拟集合种群SIS过程。
    """
    # --- 初始化 ---
    i = np.random.rand(n) * 0.1  # 初始感染比例
    s = 1 - i
    
    # 初始化轨迹记录
    i_trajr = i.reshape(1, -1)
    s_trajr = s.reshape(1, -1)

    # 预计算扩散项所需的部分
    P_out_strength = P.sum(axis=1) # D_out[i] = Σ_j P_ij
    
    # --- 模拟主循环 ---
    for t in range(1, steps):
        # --- 1. 反应步骤 (本地 SIS 动态) ---
        
        # a. 计算本地新增感染和恢复的个体比例
        #    这里我们使用一个更简单的本地感染项，以匹配微分方程的形式
        #    infect = s * (1 - np.exp(-alpha * i)) # 泊松模型
        #    或者更简单的均场近似：
        infect_local = alpha * s * i
        recover_local = beta * i
        
        # b. 计算反应后的中间状态
        i_after_reaction = i + infect_local - recover_local
        s_after_reaction = s - infect_local + recover_local

        # --- 2. 扩散步骤 (个体在节点间迁移) ---
        
        # a. 计算因扩散导致的感染者和易感者的净变化
        #    Δi_diffusion = gamma * (总流入 - 总流出)
        diffusion_net_change_i = gamma * (P.T @ i - P_out_strength * i)
        diffusion_net_change_s = gamma * (P.T @ s - P_out_strength * s)

        # b. 将扩散变化应用到反应后的状态上，得到最终的新状态
        i_new = i_after_reaction + diffusion_net_change_i
        s_new = s_after_reaction + diffusion_net_change_s
        
        # 更新状态变量 (使用旧的状态 i, s 进行扩散计算)
        i = i_new
        s = s_new
        
        # 保证比例在 [0, 1] 范围内
        i = np.clip(i, 0, 1)
        # 确保 s+i=1
        s = 1 - i
        
        # 记录轨迹
        i_trajr = np.concatenate([i_trajr, i.reshape(1, -1)], axis=0)
        s_trajr = np.concatenate([s_trajr, s.reshape(1, -1)], axis=0)
        
    # --- 格式化输出 ---
    i_trajr = np.expand_dims(i_trajr, axis=-1)

    s_trajr = np.expand_dims(s_trajr, axis=-1)
    
    # 最终形状为 [steps, n, 2]，通道0是I(i)，通道1是S(s)
    # trajectory = np.concatenate((i_trajr, s_trajr), axis=-1)
    
    return i_trajr


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
        
        # edge_path = '/home/zjy/project/AnyEpi/data/simulate/edges.csv'
        # air_edge = pd.read_csv(edge_path)
        # np.random.seed(exp_id)
        if args.graph in 'ER':
            G = nx.gnp_random_graph(n, p, directed=True)
        elif args.graph in 'BA':
            G = nx.scale_free_graph(
                n, 
                alpha=0.41, 
                beta=0.54, 
                gamma=0.05, 
                delta_in=0.2, 
                delta_out=0,
                seed=42
            )
            G = nx.DiGraph(G)
        min_w = 0.01 
        # 最大权重 (小于等于 1)
        max_w = 1.0   
        for u, v in G.edges():
            # 使用 np.random.uniform 在 [min_w, max_w) 范围内生成均匀随机数
            weight = np.random.uniform(min_w, max_w)
            
            # 将权重赋值给边的 'weight' 属性
            G[u][v]['weight'] = weight
            
        A = nx.to_numpy_array(G)
        
        print(A.sum())

        x_tr = np.zeros((args.tr_num,args.steps,n,1))
        for i in range(args.tr_num):
            print(f'Simulating training trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation_metapop_sis(args.steps, A)
          
        x_va = np.zeros((args.va_num,args.steps,n,1))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            x_va[i] = simulation_metapop_sis(args.steps, A)

        x_te = np.zeros((args.te_num,args.steps,n,1))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            x_te[i] = simulation_metapop_sis(args.steps, A)

        A = nx.to_numpy_array(G)
        x_tr = x_tr.astype(np.float16)
        x_va = x_va.astype(np.float16)
        x_te = x_te.astype(np.float16)

        # save data, output data has shape [batch, time, nodes, variables]
        A = nx.to_numpy_array(G)
        result = [x_tr,x_va,x_te,A]
        data_path = 'Meta_SIS_Direct_Weight_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)


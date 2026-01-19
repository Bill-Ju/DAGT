import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import pickle


parser = argparse.ArgumentParser('Generate FJ simulation data')
parser.add_argument('--graph', type=str, default='ER')
parser.add_argument('--num-nodes', type=int, default=50,
                    help='Number of nodes in the simulation.')
parser.add_argument('--p', type=float, default=0.1, 
                    help='Connection/add connection probability In ER/NWS')
parser.add_argument('--k', type=int, default=2, 
                    help='Inital node degree in BA/NWS')

parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=100,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=30,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=30,
                    help='Number of test trajectories.')

parser.add_argument('--steps', type=int, default=100)

args = parser.parse_args()

# def simulation(steps, A):

#     # initial condition
#     x = (np.random.rand(n) - .5) * 2
#     s = (np.random.rand(n) - .5) * 2
#     D = A.sum(axis=1)
#     D_inv = 1/ (D + 1)
#     x_trajr = x.reshape(1,-1)
#     s_trajr = s.reshape(1,-1)

#     for t in range(1,steps):
#         x = D_inv*(A@x + s)
#         x_trajr = np.concatenate([x_trajr,x.reshape(1,-1)],axis=0)
#         s_trajr = np.concatenate([s_trajr,s.reshape(1,-1)],axis=0)

#     x_trajr = np.expand_dims(x_trajr, axis = -1)
#     s_trajr = np.expand_dims(s_trajr, axis = -1)
#     x_trajr = np.concatenate((x_trajr,s_trajr),axis=-1)
#     return x_trajr
def simulation(steps, A, gamma=0.2):

    # initial condition
    x = (np.random.rand(n) - .5) * 2
    s = (np.random.rand(n) - .5) * 2
    D = A.sum(axis=1)
    D_inv = 1/ (D + 1)
    D_out = A.sum(axis=0) 
    x_trajr = x.reshape(1,-1)
    s_trajr = s.reshape(1,-1)

    for t in range(1,steps):
        out_penalty = gamma * D_out * x
        x = D_inv*(A@x + s - out_penalty)
        x_trajr = np.concatenate([x_trajr,x.reshape(1,-1)],axis=0)
        s_trajr = np.concatenate([s_trajr,s.reshape(1,-1)],axis=0)

    x_trajr = np.expand_dims(x_trajr, axis = -1)
    s_trajr = np.expand_dims(s_trajr, axis = -1)
    x_trajr = np.concatenate((x_trajr,s_trajr),axis=-1)
    return x_trajr

def plot_trajectory(x):
    t = np.arange(0,x.shape[0])
    for nid in range(n):
        plt.plot(t,x[:,nid,0])
        plt.xlabel('time')
        plt.ylabel('x(t)')
    plt.show()

if __name__ == '__main__':
    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'

    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k
        np.random.seed(exp_id)
        if args.graph in 'ER':
            G = nx.gnp_random_graph(n, p, directed=True)
            
        elif args.graph in 'BA':
            G = nx.scale_free_graph(
                n, 
                alpha=0.4, 
                beta=0.55, 
                gamma=0.05, 
                delta_in=2, 
                delta_out=1,
                seed=42
            )
            G = nx.DiGraph(G)
        
        min_w = 0.01 
        # Maximum weight (<= 1)
        max_w = 1.0   
        for u, v in G.edges():
            # Use np.random.uniform to generate uniform random numbers in [min_w, max_w)
            weight = np.random.uniform(min_w, max_w)
            
            # Assign weight to the 'weight' attribute of the edge
            G[u][v]['weight'] = weight

        A = nx.to_numpy_array(G)
        x_tr = np.zeros((args.tr_num,args.steps,n,2))
        for i in range(args.tr_num):
            print(f'Simulating train trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation(args.steps, A)
            #plot_trajectory(x_tr[i])
        x_va = np.zeros((args.va_num,args.steps,n,2))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            x_va[i] = simulation(args.steps, A)

        x_te = np.zeros((args.te_num,args.steps,n,2))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            x_te[i] = simulation(args.steps, A)

        x_tr = x_tr.astype(np.float16)
        x_va = x_va.astype(np.float16)
        x_te = x_te.astype(np.float16)

        # save data, output data has shape [batch, time, nodes, variables]
        result = [x_tr,x_va,x_te,A]
        data_path = 'FJ_Direct_Weight_' + args.graph + str(args.num_nodes) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)


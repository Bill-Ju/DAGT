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
parser.add_argument('--num-nodes', type=int, default=200,
                    help='Number of nodes in the simulation.')
parser.add_argument('--p', type=float, default=0.1, 
                    help='Connection/add connection probability In ER/NWS')
parser.add_argument('--k', type=int, default=2, 
                    help='Inital node degree in BA/NWS')

parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=1000,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=30,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=30,
                    help='Number of test trajectories.')


parser.add_argument('--infection_rate', type=float, default=0.6)
parser.add_argument('--recovery_rate', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=300)

args = parser.parse_args()

# Local infection rate
alpha = 0.5
# Local recovery rate
beta = 0.1
# Migration/diffusion rate
gamma = 0.05

def simulation_metapop_sis(steps, P):
    """
    Simulate metapopulation SIS process using discrete-time model.
    """
    # --- Initialize ---
    i = np.random.rand(n) * 0.1  # Initial infection proportion
    s = 1 - i
    
    # Initialize trajectory records
    i_trajr = i.reshape(1, -1)
    s_trajr = s.reshape(1, -1)

    # Pre-compute out-strength required for diffusion term
    P_out_strength = P.sum(axis=1) # D_out[i] = Σ_j P_ij
    
    # --- Main simulation loop ---
    for t in range(1, steps):
        # --- 1. Reaction step (local SIS dynamics) ---
        
        # a. Calculate local new infections and recoveries as proportions
        #    Using a simpler local infection term matching differential equation form
        #    infect = s * (1 - np.exp(-alpha * i)) # Poisson model
        #    or simpler mean-field approximation:
        infect_local = alpha * s * i
        recover_local = beta * i
        
        # b. Calculate intermediate state after reaction
        i_after_reaction = i + infect_local - recover_local
        s_after_reaction = s - infect_local + recover_local

        # --- 2. Diffusion step (individuals migrate between nodes) ---
        
        # a. Calculate net change in infected and susceptible due to diffusion
        #    Δi_diffusion = gamma * (total inflow - total outflow)
        diffusion_net_change_i = gamma * (P.T @ i - P_out_strength * i)
        diffusion_net_change_s = gamma * (P.T @ s - P_out_strength * s)

        # b. Apply diffusion changes to reaction state, get final new state
        i_new = i_after_reaction + diffusion_net_change_i
        s_new = s_after_reaction + diffusion_net_change_s
        
        # Update state variables (use old states i, s for diffusion calculation)
        i = i_new
        s = s_new
        
        # Ensure proportions stay within [0, 1] range
        i = np.clip(i, 0, 1)
        # Ensure s+i=1
        s = 1 - i
        
        # Record trajectory
        i_trajr = np.concatenate([i_trajr, i.reshape(1, -1)], axis=0)
        s_trajr = np.concatenate([s_trajr, s.reshape(1, -1)], axis=0)
        
    # --- Format output ---
    i_trajr = np.expand_dims(i_trajr, axis=-1)

    s_trajr = np.expand_dims(s_trajr, axis=-1)
    
    # Final shape is [steps, n, 1], channel 0 is I(i)
    # trajectory = np.concatenate((i_trajr, s_trajr), axis=-1)
    
    return i_trajr


# # iteration
# def simulation_sis(steps):
#     """
#     Simulate SIS process on network using discrete-time probabilistic model.
#     """
#     # --- Initialization ---
#     # Randomly initialize infection proportion
#     x = np.random.rand(n) * 0.1 # Initial infection proportion is low
#     s = 1 - x

#     # Initialize trajectory recording
#     x_trajr = x.reshape(1, -1)
#     s_trajr = s.reshape(1, -1)


#     # --- Main simulation loop ---
#     for t in range(1, steps):
#         infection_rate = 1 - np.exp(-beta * (A @ x))
#         infect = s * infection_rate
        
#         # Update state
#         x_new = x  + infect
#         s_new = s  - infect

#         # Ensure proportion is in [0, 1] range
#         x = np.clip(x_new, 0, 1)
#         s = np.clip(s_new, 0, 1)

#         # Record trajectory
#         x_trajr = np.concatenate([x_trajr, x.reshape(1, -1)], axis=0)
#         s_trajr = np.concatenate([s_trajr, s.reshape(1, -1)], axis=0)
        
#     # --- Format output ---
#     # Concatenate s and x trajectories on the last dimension
#     x_trajr = np.expand_dims(x_trajr, axis=-1)
#     s_trajr = np.expand_dims(s_trajr, axis=-1)
    
#     # Final shape is [steps, n, 2], channel 0 is I(x), channel 1 is S(s)
#     trajectory = np.concatenate((x_trajr, s_trajr), axis=-1)
    
#     return trajectory


if __name__ == '__main__':
    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'
    for exp_id in range(args.exp_num):
        n = 223
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
        
        print(A.sum())

        x_tr = np.zeros((args.tr_num,args.steps,n,1))
        for i in range(args.tr_num):
            print(f'Simulating training trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation_metapop_sis(args.steps, A)
          
        x_va = np.zeros((args.va_num,args.steps,n,1))
        for i in range(args.va_num):
            print(f'Simulating validation trajectory: {i+1:3d}/{args.va_num:3d}')
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
        data_path = 'meta_SIS_weight_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)


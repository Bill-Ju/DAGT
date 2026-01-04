import os
import numpy as np
import torch
import random
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import pickle


def load_data(map_configs, args):
    train_dataset = None
    map_configs_sorted = sorted(map_configs, key=lambda x: x['map_index'])
    all_train_datasets = []
    for config in map_configs_sorted:
        train_dataset = load_disease_data(
            config, args)
        print(f"train_dataset shape:{len(train_dataset)}")
        all_train_datasets.append(train_dataset)
        
    combined_train_dataset = CombinedDataset(all_train_datasets)
    
    
    print(f"batch_size_per_dataset:{args.train_batch}")
    
    sampler = RoundRobinBatchSampler(combined_train_dataset.datasets, args.train_batch)
    
    train_loader = DataLoader(
        combined_train_dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,
    )
        
    return train_loader


def load_disease_data(config, args):
    map_index = config['map_index']
    map_name = config['map_name']
    loc_num = config['loc_num']
    data_dir = os.path.join(args.data_dir, map_name)

    print(f"Loading disease data from {data_dir}")
    
    train_data_file = args.traj_file
    
    train_data = np.load(os.path.join(data_dir, train_data_file))
    
    print(f"数据维度: [batch={train_data.shape[0]}, timesteps={train_data.shape[1]}, locations={train_data.shape[2]}]")
    

    train_data = torch.from_numpy(train_data).to(torch.float32)
    
    train_data = train_data[:, args.left: args.right, :, :]
    
    
    
    i = 0
    
    # train_data[:,:,:,i] = torch.clamp(train_data[:,:,:,i], min=0)
    
    # xmax = train_data[:,:,:,i].max()
    # xmin = train_data[:,:,:,i].min()    
    # train_data[:,:,:,i] = (train_data[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
    # train_data[:,:,:,i] = (train_data[:,:,:,i] - xmin) / (xmax - xmin)

    # input data has shape [batch, time, nodes, variables]
    train_data = train_data.permute(0,2,3,1)
    # data has shape [batch, nodes, variables, time]
    
    # Create data loader
    train_dataset = TrajrData(map_index, train_data, args.Tstep)

    return train_dataset



class TrajrData(Dataset):
    def __init__(self, map_index, data, Tstep, interlacing=True):
        self.data_index = map_index
        self.data = data
        # data has shape [batch, nodes, variables, time]
        self.interlacing = interlacing
        if interlacing:
            self.Tout = self.data.shape[-1] - Tstep+1  #steps for reccurent output
        else:
            assert self.data.shape[-1]%Tstep == 0, 'Trajectory length must be integer multiple of Tstep'
            self.Tout = int(np.ceil(self.data.shape[-1]/Tstep))
        self.Tstep = Tstep
        self.batch = self.data.shape[0]
        self.datalen = self.batch*self.Tout
        print(f"datalen{self.datalen}")
        
        
    def __len__(self):
        return self.datalen
    def __getitem__(self, idx):
        i, j = idx//self.Tout, idx%self.Tout #i: batch, j: start time step
        
        if self.interlacing:
            # print(f"i:{i}, j:{j}")
            sample = self.data[i,:,:,j:j+self.Tstep]
        else:
            start_ind = j*self.Tstep
            sample = self.data[i,:,:,start_ind:start_ind+self.Tstep]
        # print(f"sample:{sample.shape}")
        return sample, self.data_index

  
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.lengths))
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        sub_idx = idx - self.cumulative_lengths[dataset_idx]
        if not (0 <= dataset_idx < len(self.datasets)):
            raise IndexError(f"Calculated dataset_idx {dataset_idx} is out of bounds for self.datasets (length {len(self.datasets)}). Input idx was {idx}. Cumulative lengths: {self.cumulative_lengths}")
        
        return self.datasets[dataset_idx][sub_idx] 
    
def custom_collate_fn(batch):
    grouped_samples = {}
    for data, dataset_id in batch:
        if dataset_id not in grouped_samples:
            grouped_samples[dataset_id] = {'data': []}
        grouped_samples[dataset_id]['data'].append(data)

    batched_output = {}
    for dataset_id, samples in grouped_samples.items():
        data_list = samples['data']
        batched_data = torch.stack(data_list)
        batched_output[f'dataset_{dataset_id}'] = batched_data
    return batched_output

class RoundRobinBatchSampler(Sampler):
    def __init__(self, datasets, batch_size_per_dataset, shuffle=True):
        self.datasets = datasets
        self.batch_size_per_dataset = batch_size_per_dataset
        self.shuffle = shuffle
        self.num_datasets = len(datasets)
        self.lengths = [len(d) for d in datasets]
        self.cumulative = [0] + list(np.cumsum(self.lengths))
        self.max_batches = min(
            l // batch_size_per_dataset for l in self.lengths
        )

    def __iter__(self):
        # Create index list for each dataset
        indices = [list(range(l)) for l in self.lengths]

        if self.shuffle:
            for arr in indices:
                random.shuffle(arr)

        for t in range(self.max_batches):
            batch = []
            for d in range(self.num_datasets):
                start = t * self.batch_size_per_dataset
                end = start + self.batch_size_per_dataset
                local_indices = indices[d][start:end]
                global_indices = [self.cumulative[d] + idx for idx in local_indices]
                batch.extend(global_indices)
            yield batch

    def __len__(self):
        return self.max_batches


import os
import logging
import torch
import torch.distributed as dist
from datetime import datetime
from torch import nn 
import numpy as np

def save_graph(args, map_configs, map_dict, logger):
    data_dir = os.path.join(args.save_path, 'temp_graph')
    os.makedirs(data_dir, exist_ok=True)
    for config in map_configs:
        map_index = config['map_index']
        map_name = config['map_name']
        loc_num = config['loc_num']
        edge_file = f'{map_name}_pred.npy'
        np.save(os.path.join(data_dir, edge_file), map_dict[map_index].cpu().numpy())
        logger.info(f"Saved graph for {map_name}")

def setuplogger(args, main_log_filename="experiment.log", 
                     console_level_rank0=logging.INFO, console_level_other_ranks=logging.ERROR,
                     file_level_rank0=logging.INFO):

    logger = logging.getLogger(f"Main")
    # Set the global level of logger, handlers can have their own higher levels
    logger.setLevel(logging.DEBUG) # Capture all levels of messages, filtered by handler

    # Prevent duplicate addition of handlers (if this function is accidentally called multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        f'%(asctime)s - Main - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    main_log_filename = f"experiment_{hour}.log"

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level_rank0)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    datadir_basename = args.data_dir.split('/')[-1] # Extract 'v5'
    subdir_name = f"{datadir_basename}_{month}_{day}"
    
    base_save_dir = os.path.join(args.pwd, f'{args.save_path}/logs/{args.left}_{args.right}/{args.backbone}/{subdir_name}/')
    os.makedirs(base_save_dir, exist_ok=True)
    
    main_log_file_path = os.path.join(base_save_dir, main_log_filename)
    file_handler = logging.FileHandler(main_log_file_path, mode="a") # Append mode
    file_handler.setLevel(file_level_rank0)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Rank 0 Logger setup complete. Logging to console and file.")
   

    return logger


def save_model(epoch_idx, model, args,  logger):
    """
    Only Rank 0 saves the model, optimizer, scheduler, and training history.
    optimizers_dict: {'name1': optimizer1, 'name2': optimizer2}
    schedulers_dict: {'name1': scheduler1, 'name2': scheduler2}
    training_history: Object or dictionary containing training information, such as loss lists
    """

    logger.info(f'Saving model and states at epoch {epoch_idx+1}...')
    
    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    state_dict_to_save = model.state_dict()
    
    content = {
        'epoch': epoch_idx + 1,
        'model_state_dict': state_dict_to_save, # Key: Save .module's state_dict
    }
        
    datadir_basename = args.data_dir.split('/')[-1] # Extract 'v5'
    subdir_name = f"{datadir_basename}_{month}_{day}"
    base_save_dir = os.path.join(args.pwd, f'{args.save_path}/best_models/{args.left}_{args.right}/{args.backbone}/{subdir_name}/')
    os.makedirs(base_save_dir, exist_ok=True)
    
    filename = f'any_{hour}_{minute}.pth'


    save_path = os.path.join(base_save_dir, filename)

    try:
        torch.save(content, save_path)
        logger.info(f'Model and states saved to: {save_path}')
    except Exception as e:
        logger.error(f"Error saving model to {save_path}: {e}", exc_info=True)
    
    load_model_path = f'{subdir_name}/{filename}'
    return load_model_path


def load_model(model_to_load_into, load_model_path, args):

    logger = logging.getLogger(f"Model") # 获取当前rank的logger
    loaded_epoch = 0

    base_save_dir = os.path.join(args.pwd, f'{args.save_path}/best_models/')
    filename = load_model_path
    load_path = os.path.join(base_save_dir, filename)        
    
    if os.path.exists(load_path):
        logger.info(f"Attempting to load model and states from: {load_path}")
        try:
            # Load to CPU, then move to rank 0's device, DDP will handle synchronization
            checkpoint = torch.load(load_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                saved_state_dict = checkpoint['model_state_dict']
        
                # Get whether the current model is DDP wrapped
                is_model_ddp = isinstance(model_to_load_into, torch.nn.parallel.DistributedDataParallel)

                # Adjust state_dict keys
                    
                for k, v in saved_state_dict.items():
                    if is_model_ddp:
                        # If current model is DDP, but saved keys don't have "module." prefix, add it
                        if not k.startswith('module.'):
                            name = 'module.' + k
                        else:
                            name = k
                        if name =='module.edge_weights' and args.reinitialize_edge:
                            v = 0.5*torch.ones(args.disease_num, args.node_num ** 2 - args.node_num, 1)
                            logger.info('=====reinitialize edge====')
                    else:
                        # If current model is not DDP, but saved keys have "module." prefix, remove it
                        if k.startswith('module.'):
                            name = k[7:] # Remove 'module.'
                        else:
                            name = k
                    new_state_dict[name] = v
                    
                # Load using the adjusted state_dict
                model_to_load_into.load_state_dict(new_state_dict)
                logger.info("Model state_dict loaded successfully on rank 0.")
            else:
                logger.error(f"Unrecognized checkpoint format at {load_path}.")
                dist.barrier()
                return loaded_epoch

            if 'epoch' in checkpoint:
                loaded_epoch = checkpoint['epoch']
                logger.info(f"Resuming from epoch: {loaded_epoch}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {load_path}: {e}", exc_info=True)
    else:
        logger.info(f"Checkpoint file not found at: {load_path}. Starting from scratch.")
        
    # DDP will handle parameter synchronization
    return loaded_epoch



def setup_ddp_logger(rank, args, main_log_filename="experiment.log", 
                     console_level_rank0=logging.INFO, console_level_other_ranks=logging.ERROR,
                     file_level_rank0=logging.INFO):
    """
    为DDP环境设置logger。
    Rank 0 会有一个主日志文件，并且控制台输出更详细。
    其他Rank的控制台输出更简洁，可以选择性地将它们的错误也记录到主日志文件或单独文件。
    """
    logger = logging.getLogger(f"DDP_Rank_{rank}")
    # Set the global level of logger, handlers can have their own higher levels
    logger.setLevel(logging.DEBUG) # Capture all levels of messages, filtered by handler

    # Prevent duplicate addition of handlers (if this function is accidentally called multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    main_log_filename = f"experiment_{hour}.log"

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level_rank0 if rank == 0 else console_level_other_ranks)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    datadir_basename = args.data_dir.split('/')[-1] # Extract 'train_data_v5'
    subdir_name = f"{datadir_basename}_{month}_{day}"
    
    # File Handler (create main log file only for Rank 0)
    if rank == 0:
        base_save_dir = os.path.join(args.pwd, f'{args.save_path}/logs/{args.left}_{args.right}/{args.backbone}/{args.loss_type}/{subdir_name}/')
        os.makedirs(base_save_dir, exist_ok=True)
        
        main_log_file_path = os.path.join(base_save_dir, main_log_filename)
        file_handler = logging.FileHandler(main_log_file_path, mode="a") # Append mode
        file_handler.setLevel(file_level_rank0)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Rank 0 Logger setup complete. Logging to console and file.")
    else:
        logger.info(f"Rank {rank} Logger setup complete. Logging to console (level: {console_level_other_ranks}).")

    return logger


def save_ddp_model(rank, epoch_idx, ddp_model, training_history, args, mse=False, test=False):
    """
    仅在 Rank 0 保存模型、优化器、调度器和训练历史。
    optimizers_dict: {'name1': optimizer1, 'name2': optimizer2}
    schedulers_dict: {'name1': scheduler1, 'name2': scheduler2}
    training_history: 包含训练信息的对象或字典，例如损失列表
    """
    if rank == 0:
        logger = logging.getLogger(f"DDP_Rank_{rank}") # 获取当前rank的logger
        logger.info(f'Saving model and states at epoch {epoch_idx+1}...')
        
        now = datetime.now()
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute

        content = {
            'epoch': epoch_idx + 1,
            'model_state_dict': ddp_model.module.state_dict(), # 关键：保存 .module 的 state_dict
        }
        # # Save optimizer state
        # for name, optim in optimizers_dict.items():
        #     content[f'optimizer_{name}_state_dict'] = optim.state_dict()
        # # Save scheduler state
        # for name, sched in schedulers_dict.items():
        #     content[f'scheduler_{name}_state_dict'] = sched.state_dict()
        
        if training_history is not None:
            content['training_history'] = training_history
            
        datadir_basename = args.data_dir.split('/')[-1] # 提取 'train_data_v5'
        subdir_name = f"{datadir_basename}_{month}_{day}"
        if test:
            base_save_dir = os.path.join(args.pwd, f'{args.save_path}/best_models/test/{args.backbone}/{args.loss_type}/{subdir_name}/')
        else :
            base_save_dir = os.path.join(args.pwd, f'{args.save_path}/best_models/{args.left}_{args.right}/{args.backbone}/{args.loss_type}/{subdir_name}/')
        os.makedirs(base_save_dir, exist_ok=True)
        
        filename = f'any_{hour}_{minute}.pth'
        if mse:
            filename = f'any_{hour}_{minute}_mse.pth'

        save_path = os.path.join(base_save_dir, filename)

        try:
            torch.save(content, save_path)
            logger.info(f'Model and states saved to: {save_path}')
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}", exc_info=True)
        
        load_model_path = f'{subdir_name}/{filename}'
        return load_model_path


def load_ddp_model(rank, model_to_load_into, load_model_path, args):
    """
    在 Rank 0 加载模型、优化器和调度器状态，DDP 会自动同步模型参数。
    优化器和调度器状态只在 rank 0 加载，如果需要在其他 rank 上恢复它们（通常不需要），
    则需要额外的广播逻辑。
    返回加载的 epoch 数（如果有）和其他训练历史。
    """
    logger = logging.getLogger(f"DDP_Rank_{rank}") # 获取当前rank的logger
    loaded_epoch = 0
    training_history_loaded = None

    is_model_ddp = False
    new_state_dict = {}
    
    if rank == 0:
        base_save_dir = os.path.join(args.pwd, f'{args.save_path}/best_models/')
        filename = load_model_path
        load_path = os.path.join(base_save_dir, filename)        
        
        if os.path.exists(load_path):
            logger.info(f"Attempting to load model and states from: {load_path}")
            try:
                # Load to CPU, then move to rank 0's device, DDP will handle synchronization
                checkpoint = torch.load(load_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    saved_state_dict = checkpoint['model_state_dict']
        
                    # Get whether the current model is DDP wrapped
                    is_model_ddp = isinstance(model_to_load_into, torch.nn.parallel.DistributedDataParallel)

                    # Adjust state_dict keys
                    
                    for k, v in saved_state_dict.items():
                        if is_model_ddp:
                            # If current model is DDP, but saved keys don't have "module." prefix, add it
                            if not k.startswith('module.'):
                                name = 'module.' + k
                            else:
                                name = k
                            if name =='module.edge_weights' and args.reinitialize_edge:
                                v = 0.5*torch.ones(args.disease_num, args.node_num ** 2 - args.node_num, 1)
                                logger.info('=====reinitialize edge====')
                        else:
                            # If current model is not DDP, but saved keys have "module." prefix, remove it
                            if k.startswith('module.'):
                                name = k[7:] # Remove 'module.'
                            else:
                                name = k
                        new_state_dict[name] = v
                    
                    # Load using the adjusted state_dict
                    model_to_load_into.load_state_dict(new_state_dict)
                    logger.info("Model state_dict loaded successfully on rank 0.")
                else:
                    logger.error(f"Unrecognized checkpoint format at {load_path}.")
                    dist.barrier()
                    return loaded_epoch, training_history_loaded

                # # Load optimizer state (rank 0 only)
                # for name, optim in optimizers_dict.items():
                #     key = f'optimizer_{name}_state_dict'
                #     if key in checkpoint and optim is not None:
                #         optim.load_state_dict(checkpoint[key])
                #         logger.info(f"Optimizer '{name}' state loaded.")
                
                # # Load scheduler state (rank 0 only)
                # for name, sched in schedulers_dict.items():
                #     key = f'scheduler_{name}_state_dict'
                #     if key in checkpoint and sched is not None:
                #         sched.load_state_dict(checkpoint[key])
                #         logger.info(f"Scheduler '{name}' state loaded.")

                if 'epoch' in checkpoint:
                    loaded_epoch = checkpoint['epoch']
                    logger.info(f"Resuming from epoch: {loaded_epoch}")
                
                if 'training_history' in checkpoint:
                    training_history_loaded = checkpoint['training_history']

            except Exception as e:
                logger.error(f"Failed to load checkpoint from {load_path}: {e}", exc_info=True)
        else:
            logger.info(f"Checkpoint file not found at: {load_path}. Starting from scratch.")
    
    # All processes wait here to ensure rank 0 completes loading (or skips)
    # DDP synchronizes model parameters from rank 0 to all other ranks during its initialization (or after the first forward/backward)
    if dist.is_initialized():
        dist.barrier()
    
    # Broadcast the state dict to all other ranks
    state_dict = model_to_load_into.state_dict()

    for key, value in state_dict.items():
        if is_model_ddp:
            dist.broadcast(value, src=0)
        else:
            dist.broadcast(value, src=0)

    # Update the model's state dict on all ranks
    model_to_load_into.load_state_dict(state_dict)

    logger.info(f"Model state_dict broadcasted and loaded on rank {dist.get_rank()}.")
    
    # model_to_load_into should have been .to(device) at creation
    # DDP will handle parameter synchronization
    return loaded_epoch, training_history_loaded
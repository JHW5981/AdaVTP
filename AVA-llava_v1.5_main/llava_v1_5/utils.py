import os
import torch
import torch.distributed as dist
import numpy as np
import random
import json
from pathlib import Path

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
    else:    
        print("不是分布式训练...")
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
        print(f"| MASTER_ADDR: {os.environ['MASTER_ADDR']}| MASTER_PORT: {os.environ['MASTER_PORT']} | RANK: {args.rank} | WORLD_SIZE: {args.world_size} | LOCAL RANK: {args.gpu} |")
    else:
        print(f"| RANK: {args.rank} | WORLD_SIZE: {args.world_size} | LOCAL RANK: {args.gpu} |")
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url, 
        world_size=args.world_size, 
        rank=args.rank,
        device_id=torch.device(f'cuda:{args.gpu}')
        )
        
    torch.distributed.barrier()
    
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def load_pretrained_weight(model_path):
    if os.path.isdir(model_path):
        index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
        
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            shard_files = set(index_data['weight_map'].values())            
            # 加载所有分片并合并
            merged_state_dict = {}
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(model_path, shard_file)
                shard_weights = torch.load(shard_path, map_location='cpu')
                merged_state_dict.update(shard_weights)
            pretrained_weight = merged_state_dict
            
        else:
            model_files = []
            for file in os.listdir(model_path):
                if file.endswith(('.bin', '.pth', '.safetensors')):
                    model_files.append(file)
            if not model_files:
                raise FileNotFoundError(f"在文件夹 {model_path} 中未找到模型权重文件")

            if len(model_files) == 1:
                model_file = os.path.join(model_path, model_files[0])
                pretrained_weight = torch.load(model_file, map_location='cpu')
            else:
                pretrained_weight = {}
                for model_file in sorted(model_files):
                    file_path = os.path.join(model_path, model_file)
                    component_name = os.path.splitext(model_file)[0]
                    pretrained_weight[component_name] = torch.load(file_path, map_location='cpu')       
    else:
        pretrained_weight = torch.load(model_path, map_location='cpu')
    
    return pretrained_weight

def save_model(args, model, model_without_ddp, optimizer, loss_scaler, epoch, best_acc):
    output_dir = Path(args.save_ckpt_path)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'best_acc': best_acc,
        }
        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)



def select_hidden_states(hidden_states: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 hidden_states 中筛选，保持 batch 维度。仅支持 batch=1。
    hidden_states: (1, T, D), mask: (1, T), 1 表示保留
    返回: (1, T', D)
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    return hidden_states[0, indices, :].unsqueeze(0)


def select_position_ids(position_ids: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 position_ids 中筛选，保持 batch 维度。仅支持 batch=1。
    position_ids: (1, T) 或 (3, 1, T), mask: (1, T)
    返回: (1, T') 或 (3, 1, T')
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    if position_ids.dim() == 3:
        return position_ids[:, 0, indices].unsqueeze(1)
    return position_ids[0, indices].unsqueeze(0)

def select_causal_mask(causal_mask: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 causal_mask 中筛选，保持 batch 维度。仅支持 batch=1。
    支持两种形状：
    - causal_mask: (1, 1, T, T)，第二维为 head 广播维，mask: (1, T)，返回: (1, 1, T', T')
    - causal_mask: (1, 1, 1, T)，mask: (1, T)，返回: (1, 1, 1, T')
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    if causal_mask.dim() == 4 and causal_mask.shape[2] == 1:
        # (1, 1, 1, T) -> (1, 1, 1, T')
        return causal_mask[:, :, :, indices]
    # (1, 1, T, T) -> (1, 1, T', T')
    return causal_mask[0, :, indices, :][:, :, indices].unsqueeze(0)

def select_position_embeddings(position_embeddings: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
    """
    根据 mask 从 position_embeddings 中筛选，保持 batch 维度。仅支持 batch=1。
    position_embeddings: (cos, sin)，每个形状为 (3, 1, T, D), mask: (1, T)
    返回: (cos_selected, sin_selected)，每个形状为 (3, 1, T', D)
    """
    cos, sin = position_embeddings
    indices = mask[0].nonzero(as_tuple=True)[0]
    return cos[:, :, indices, :], sin[:, :, indices, :]

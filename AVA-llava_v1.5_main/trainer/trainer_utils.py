"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoImageProcessor
from llava_v1_5.modeling_llava import LlavaLlamaForCausalLM
from llava_v1_5.configuration_llava import LlavaLlamaConfig
from llava_v1_5.utils import load_pretrained_weight


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e9
    Logger(f'Model Params: {total:.4f}B')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}.pth'
    resume_path = f'{save_dir}/{weight}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

def init_model(args, from_weight="none", save_dir='../out', device='cuda', dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(args.llava_v1_5_weight_path)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_tower_path)
    config = LlavaLlamaConfig.from_pretrained(args.llava_v1_5_weight_path)
    config.gumbel_temperature = args.gumbel_temperature
    config.train_drop_rate = args.train_drop_rate
    
    disable_torch_init()
    model = LlavaLlamaForCausalLM(config)
    
    model_param_names = model.state_dict().keys()
    pretrained_weight_language_model = load_pretrained_weight(args.llava_v1_5_weight_path)
    pretrained_weight_vision_model = load_pretrained_weight(args.vision_tower_path)
    for k, v in pretrained_weight_vision_model.items():
        if 'model.vision_tower.vision_tower.' + k in model_param_names:
            pretrained_weight_language_model['model.vision_tower.vision_tower.' + k] = v.to(dtype)
    model.load_state_dict(pretrained_weight_language_model, strict=False)

    
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}.pth'
        weights = torch.load(weight_path, map_location=device)
        # weights中只包含model star模块的参数（只加载star模块的权重）
        model_state_dict = model.state_dict()
        for k, v in weights.items():
            if k in model_state_dict:
                model_state_dict[k].copy_(v)
            else:
                print(f"Warning: Key {k} from weights not found in model.state_dict(). Skipped.")
        print(f'Loaded star module weights from {from_weight} at {weight_path}')
    # else:
    #     # 加载未训练的语言模型参数
    #     from safetensors.torch import load_file
    #     safetensor1 = args.first_load_qwen_2_5_l_weight_path + "/model-00001-of-00002.safetensors"
    #     safetensor2 = args.first_load_qwen_2_5_l_weight_path + "/model-00002-of-00002.safetensors"
    #     language_model_weights = {}
    #     language_model_weights.update(load_file(safetensor1))
    #     language_model_weights.update(load_file(safetensor2))
    #     new_language_model_weights = {}
    #     for n, p in language_model_weights.items():
    #         n = n.replace('model.', '')
    #         new_language_model_weights[n] = p
    #     model.language_model.load_state_dict(new_language_model_weights, strict=True)
    return model.to(device, dtype=dtype), tokenizer, image_processor


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


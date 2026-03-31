import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import json
import math
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from qwen2_5vl.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from dataset.dataset import MyDataset, DataCollatorForSupervisedDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, get_model_params

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, dtype=None):
    start_time = time.time()
    last_grad_norm = 0.0
    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch['input_ids'].to(args.device)
        visual_mask = batch['visual_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        pixel_values = batch['pixel_values'].to(args.device, dtype=dtype) if batch['pixel_values'] is not None else None #避免纯文本训练报错
        image_grid_thw = batch['image_grid_thw'].to(args.device) if batch['image_grid_thw'] is not None else None
        position_ids = batch['position_ids'].to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            visual_mask=visual_mask,
        )
        loss = res.loss
        loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 只对可训练参数进行梯度裁剪
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            last_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).item()

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, grad_norm: {last_grad_norm:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if swanlab: swanlab.log({"loss": current_loss, "logits_loss": current_logits_loss, "grad_norm": last_grad_norm, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items() if 'star' in k}
            torch.save(state_dict, ckp)
            lm_checkpoint(weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='/mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/checkpoints/main_checkpoints')
            model.train()
            del state_dict

        del res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full SFT")
    parser.add_argument("--first_load_qwen_2_5_vl_weight_path", type=str, default="/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/Qwen/Qwen2.5-VL-3B-Instruct", help="Qwen2.5-VL的权重路径")
    parser.add_argument("--first_load_qwen_2_5_l_weight_path", type=str, default="/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/Qwen/Qwen2.5-3B", help="Qwen2.5的权重路径")
    parser.add_argument("--save_dir", type=str, default="/mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/out/main_out", help="模型保存目录")
    parser.add_argument('--save_weight', default='main-finetune', type=str, help="保存权重的前缀名")
    parser.add_argument('--name', default='main-finetune', type=str, help="训练名称")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--data_path", type=str, default="/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json", help="训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_false", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="Main-Finetune", help="wandb项目名")
    # Qwen数据的一些限制
    parser.add_argument('--max_pixels', default=28 * 28 * 1024, type=int, metavar='MAX_PIXELS', help="Max pixels (default: 28 * 28 * 576)")
    parser.add_argument('--min_pixels', default=28 * 28 * 128, type=int, metavar='MIN_PIXELS', help="Min pixels (default: 28 * 28 * 16)")
    parser.add_argument('--video_max_frames', default=8, type=int, metavar='VIDEO_MAX_FRAMES', help="Video max frames (default: 8)")
    parser.add_argument('--video_min_frames', default=4, type=int, metavar='VIDEO_MIN_FRAMES', help="Video min frames (default: 4)")
    parser.add_argument('--video_max_pixels', default=1024 * 28 * 28, type=int, metavar='VIDEO_MAX_PIXELS', help="Video max pixels (default: 1024 * 28 * 28)")
    parser.add_argument('--video_min_pixels', default=256 * 28 * 28, type=int, metavar='VIDEO_MIN_PIXELS', help="Video min pixels (default: 256 * 28 * 28)")
    parser.add_argument('--video_fps', default=2, type=float, metavar='VIDEO_FPS', help="Video fps (default: 2)")
    
    parser.add_argument("--gumbel_temperature", type=float, default=1, help="Gumbel-Sigmoid温度")
    parser.add_argument("--train_drop_rate", type=float, default=0.89, help="训练drop rate")


    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = Qwen2_5_VLConfig.from_pretrained(args.first_load_qwen_2_5_vl_weight_path)
    ckp_data = lm_checkpoint(weight=args.save_weight, save_dir='/mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/checkpoints/main_checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # ========== 4. 配wandb ==========
    swanlab = None
    if args.use_wandb and is_main_process():
        import swanlab
        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        resume = 'must' if swanlab_id else None
        swanlab_run_name = f"Qwen-Main-{args.name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        swanlab.init(
            project=args.wandb_project, 
            name=swanlab_run_name, 
            id=swanlab_id, 
            resume=resume)
    
    # ========== 5. 定义模型、数据 ==========
    model, processor = init_model(args, from_weight=args.from_weight, save_dir=args.save_dir, device=args.device)
    processor.image_processor.max_num_frames = args.video_max_frames
    processor.image_processor.min_num_frames = args.video_min_frames
    processor.image_processor.max_num_pixels = args.video_max_pixels
    processor.image_processor.min_num_pixels = args.video_min_pixels
    processor.image_processor.fps = args.video_fps
    
    model.to(dtype)
    with open(args.data_path, "r") as f:
        data = json.load(f)
    train_ds = MyDataset(data, processor, args)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
     
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)


    # ========== 7. 解冻merge模块的参数 ==========    
    for name, param in model.named_parameters():
        if 'star' in name:
            param.requires_grad = True
            # if param.dim() > 1:  # 通常权重（Linear, Conv等）
            #     nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            # else:  # 通常偏置
            #     nn.init.zeros_(param)
            if name == "model.language_model.star.q_proj.weight":
                param.data.copy_(model.model.language_model.layers[0].self_attn.q_proj.weight.data)
            elif name == "model.language_model.star.q_proj.bias":
                param.data.copy_(model.model.language_model.layers[0].self_attn.q_proj.bias.data)
            elif name == "model.language_model.star.k_proj.weight":
                param.data.copy_(model.model.language_model.layers[0].self_attn.k_proj.weight.data)
            else:
                param.data.copy_(model.model.language_model.layers[0].self_attn.k_proj.bias.data)
        else:
            param.requires_grad = False
    get_model_params(model)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.4f}B')
    
    # ========== 8. 创建优化器（只包含可训练参数） ==========
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    if ckp_data and 'optimizer' in ckp_data:
        # 尝试加载optimizer状态，如果参数不匹配（比如之前训练的是所有参数）则跳过
        try:
            optimizer.load_state_dict(ckp_data['optimizer'])
            Logger('成功加载optimizer状态')
        except (ValueError, KeyError) as e:
            Logger(f'无法加载optimizer状态（可能因为参数不匹配），将使用新的optimizer: {e}')

    # ========== 9. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # ========== 10. 开始训练 ==========
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer))
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, swanlab, dtype)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer))
            train_epoch(epoch, loader, len(loader), 0, swanlab, dtype)
    # ========== 11. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
   
import sys
sys.path.insert(0, "/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7")

import os
import json
import argparse
import torch
import time
import shortuuid
import pandas as pd
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.dataset import MyDataset, DataCollatorForSupervisedDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, get_model_params
from evaluation.eval_utils import calculate_choice_accuracy, calculate_relaxed_accuracy, calculate_yes_no_accuracy
from evaluation.logger import MetricLogger

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def eval_benchmark(model, tokenizer, eval_dataloader, args, metric_logger, dtype):
    model_for_inference = model.module if isinstance(model, DistributedDataParallel) else model
    model_for_inference.eval()
    header = 'Test:'
    
    # 多卡并行时，每个 rank 写入独立文件，避免并发写入导致 jsonl 损坏
    if dist.is_initialized():
        rank = dist.get_rank()
        ans_file_path = f"{args.answer_file}.rank{rank}"
    else:
        ans_file_path = args.answer_file
    ans_file = open(ans_file_path, "w+")

    for batch in metric_logger.log_every(eval_dataloader, 10, header):
        current_num = len(batch['input_ids'])
        batch['input_ids'] = batch['input_ids'].to(args.device)
        batch['images'] = [img.to(args.device, dtype=dtype) for img in batch['images']]
        cur_prompt = batch['prompt']
        idx = batch['question_id']
        
        # 记录生成开始时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        generated_answers_ids = model_for_inference.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            images=batch['images'],
            image_sizes=batch['image_sizes'],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        generated_answers = tokenizer.batch_decode(generated_answers_ids, skip_special_tokens=True)
        generated_answers = [answer.strip() for answer in generated_answers]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx[0],
                                "round_id": 0,
                                "prompt": cur_prompt[0],
                                "text": generated_answers[0],
                                "answer_id": ans_id,
                                "model_id": "v1-298M-data-epoch-3",
                                "metadata": {}}) + "\n")

        # 记录生成结束时间并计算吞吐量
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ans_file.close()
    
    # 确保所有进程都完成了这个数据集的评估
    if dist.is_initialized():
        torch.distributed.barrier()
        # rank 0 合并各 rank 的答案文件到最终 answer_file（按 question_id 排序保证顺序一致）
        if dist.get_rank() == 0:
            world_size = dist.get_world_size()
            all_answers = []
            for r in range(world_size):
                rank_path = f"{args.answer_file}.rank{r}"
                if os.path.exists(rank_path):
                    with open(rank_path, "r") as f:
                        for line in f:
                            if line.strip():
                                all_answers.append(json.loads(line))
                    os.remove(rank_path)
            all_answers.sort(key=lambda x: x["question_id"])
            with open(args.answer_file, "w") as merged:
                for ans in all_answers:
                    merged.write(json.dumps(ans, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--llava_v1_5_weight_path", type=str, default="/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/liuhaotian/llava-v1.5-7b", help="Qwen2.5-VL的权重路径")
    parser.add_argument("--vision_tower_path", default='/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/openai/clip-vit-large-patch14-336', type=str, metavar='MM_VISION_TOWER', help="Name of vision tower to use")
    parser.add_argument('--save_dir', default='/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/out/main_out', type=str, help="测试结果保存目录")
    parser.add_argument('--eval_dir', default='/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/eval_results', type=str, help="测试结果保存目录")
    parser.add_argument('--from_weight', default='v1-298M-data-epoch-3', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument('--max_new_tokens', default=128, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument("--question_file", type=str, default="/mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/mmbench/mmbench_dev_20230712.jsonl", help="评测数据路径")
    parser.add_argument("--image_folder", type=str, default="/mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/mmbench/images", help="图片文件夹路径")
    parser.add_argument("--answer_file", type=str, default="/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/mmbench/answers/mmbench_dev_20230712/v1-298M-data-epoch-3.jsonl", help="答案路径")
    
    parser.add_argument("--gumbel_temperature", type=float, default=1, help="Gumbel-Sigmoid温度")
    parser.add_argument("--train_drop_rate", type=float, default=0.89, help="训练drop rate")

    args = parser.parse_args()
    
    disable_torch_init()
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置测试结果保存目录 ==========
    os.makedirs(args.eval_dir, exist_ok=True)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # ========== 4. 定义模型、数据 ==========
    model, tokenizer, image_processor = init_model(args, from_weight=args.from_weight, save_dir=args.save_dir, device=args.device, dtype=dtype)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    mydataset = MyDataset(questions, tokenizer, image_processor, args)
    eval_sampler = DistributedSampler(mydataset) if dist.is_initialized() else None

    # ========== 5. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True,)
    
    # ========== 6. 开始评测 ==========
    metric_logger = MetricLogger(delimiter=" ")
    eval_dataloader = DataLoader(
        mydataset, 
        sampler=eval_sampler, 
        batch_size=1,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        )
    Logger(f'Dataset is evaluating...')
    eval_benchmark(model, tokenizer, eval_dataloader, args, metric_logger, dtype)

    metric_logger.synchronize_between_processes()

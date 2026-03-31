export CUDA_VISIBLE_DEVICES=4,5,7
export PYTHONPATH=/mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7:/mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV:$PYTHONPATH

cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/trainer
python -m torch.distributed.run --nproc_per_node=3 --master_port=29502 train_finetune.py \
    --name v1-298M-data-epoch-3 \
    --learning_rate 3e-5 \
    --batch_size 4 \
    --epochs 30 \
    --save_weight v1-298M-data-epoch-3 \
    --save_interval 1000 \
    --wandb_project LLaVA-Main \
    --from_resume 1 \
    --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=2 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME/answers/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.67

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME
# python convert_answer_to_mme.py --experiment v1-298M-data-epoch-3


# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME/eval_tool
# echo "298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.78" >> /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME/eval_tool/results.txt
# python calculation.py --results_dir /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME/eval_tool/answers/v1-298M-data-epoch-3 >> /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MME/eval_tool/results.txt

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=8 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/train_images \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.67

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/textvqa
# python eval_textvqa.py \
#     --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-3.jsonl


# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=8 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/data/images \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.89

# python /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/gqa/convert_gqa_for_eval.py --src /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-3.jsonl --dst /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/gqa/data/testdev_balanced_predictions.json

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/gqa/data
# python eval.py --tier testdev_balanced


# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 eval_benchmarks_mmbench.py \
#     --question_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/mmbench/mmbench_dev_20230712.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/mmbench/images \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/mmbench/answers/mmbench_dev_20230712/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.67

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/mmbench
# python convert_mmbench_for_submission.py \
#     --annotation-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
#     --result-dir /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/mmbench/answers/mmbench_dev_20230712 \
#     --upload-dir /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712 \
#     --experiment v1-298M-data-epoch-3

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/pope/val2014 \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/answers/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.67

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/pope
# python eval_pope.py \
#     --annotation-dir /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/coco \
#     --question-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/answers/v1-298M-data-epoch-3.jsonl

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/PyramidDrop/playground/data/eval/seed_bench/llava-seed-bench-existing.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/seed_bench \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-3.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.89

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/seed_bench
# python convert_seed_for_submission.py \
#     --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/seed_bench/SEED-Bench.json \
#     --result-file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-3.jsonl \
#     --result-upload-file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/seed_bench/answers_upload/v1-298M-data-epoch-3.jsonl

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu
# python run_llava.py \
#     --train_drop_rate 0.67 \
#     --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.67.json

# python main_eval_only.py --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.67.json


# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu
# python run_llava.py \
#     --train_drop_rate 0.78 \
#     --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.78.json

# python main_eval_only.py --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.78.json

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu
# python run_llava.py \
#     --train_drop_rate 0.89 \
#     --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.89.json

# python main_eval_only.py --output_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/MMMU/mmmu/v1-298M-data-epoch-3/v1-298M-data-epoch-3-0.89.json

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=8 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/test2015 \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2/answers/v1-298M-data-epoch-3-0.67.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.67
# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2
# python convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt v1-298M-data-epoch-3-0.67

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=5 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/test2015 \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2/answers/v1-298M-data-epoch-3-0.78.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.78
# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2
# python convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt v1-298M-data-epoch-3-0.78

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# python -m torch.distributed.run --nproc_per_node=5 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/test2015 \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2/answers/v1-298M-data-epoch-3-0.89.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.89

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2
# python convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt v1-298M-data-epoch-3-0.89

# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation
# /mnt/inaisfs/home/test4/test3/miniconda3/envs/qwen/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29502 eval_benchmarks.py \
#     --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/vqav2/test2015 \
#     --answer_file /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2/answers/v1-298M-data-epoch-3-0.78.jsonl \
#     --from_weight v1-298M-data-epoch-3 \
#     --train_drop_rate 0.78
# cd /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-7/evaluation/playground/data/eval/vqav2
# python convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt v1-298M-data-epoch-3-0.78

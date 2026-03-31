export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/root/mysapce/workspace/AVA-qwen2_5_vl_main:$PYTHONPATH

# cd /root/mysapce/workspace/AVA-qwen2_5_vl_main/trainer
python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
    --name v1-298M-data-epoch-3-3e-5 \
    --learning_rate 3e-5 \
    --batch_size 4 \
    --epochs 3 \
    --save_weight v1-298M-data-epoch-3-3e-5 \
    --save_interval 1000 \
    --wandb_project Qwen2.5-VL-Main \
    --from_resume 1 \
    --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name v1-298M-data-epoch-1-3e-5 \
#     --learning_rate 3e-5 \
#     --batch_size 4 \
#     --epochs 1 \
#     --save_weight v1-298M-data-epoch-1-3e-5 \
#     --save_interval 1000 \
#     --wandb_project Qwen2.5-VL-Main \
#     --from_resume 1 \
#     --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name v1-298M-data-epoch-3-6e-5 \
#     --learning_rate 6e-5 \
#     --batch_size 4 \
#     --epochs 3 \
#     --save_weight v1-298M-data-epoch-3-6e-5 \
#     --save_interval 1000 \
#     --wandb_project Qwen2.5-VL-Main \
#     --from_resume 1 \
#     --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name v1-298M-data-epoch-3-1e-4 \
#     --learning_rate 1e-4 \
#     --batch_size 4 \
#     --epochs 3 \
#     --save_weight v1-298M-data-epoch-3-1e-4 \
#     --save_interval 1000 \
#     --wandb_project Qwen2.5-VL-Main \
#     --from_resume 1 \
#     --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name v1-298M-data-epoch-3-1e-5 \
#     --learning_rate 1e-5 \
#     --batch_size 4 \
#     --epochs 3 \
#     --save_weight v1-298M-data-epoch-3-1e-5 \
#     --save_interval 1000 \
#     --wandb_project Qwen2.5-VL-Main \
#     --from_resume 1 \
#     --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json

# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name v1-298M-data-epoch-3-3e-6 \
#     --learning_rate 3e-6 \
#     --batch_size 4 \
#     --epochs 3 \
#     --save_weight v1-298M-data-epoch-3-3e-6 \
#     --save_interval 1000 \
#     --wandb_project Qwen2.5-VL-Main \
#     --from_resume 1 \
#     --data_path /mnt/inaisfs/home/test3/jihuawei/workspace/AVA-llava_v1.5-jvis-arch-2/created_datasets/llava_v1_5_mix665k_new_with_images_short_1_1.json



# cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
# python -m torch.distributed.run --nproc_per_node=8 eval_benchmarks.py \
#     --save_dir /root/mysapce/workspace/AVA-qwen2_5_vl_main/out/main_out \
#     --eval_dir /root/mysapce/workspace/AVA-qwen2_5_vl_main/eval_results \
#     --from_weight main-finetune-ablate1

# cd /root/mysapce/workspace/AVA-qwen2_5_vl_main/trainer
# python -m torch.distributed.run --nproc_per_node=8 train_finetune.py \
#     --name main-finetune-ablate2 \
#     --learning_rate 6e-6 \
#     --batch_size 8 \
#     --epochs 4 \
#     --save_weight main-finetune-ablate2

# cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
# python -m torch.distributed.run --nproc_per_node=8 eval_benchmarks.py \
#     --save_dir /root/mysapce/workspace/AVA-qwen2_5_vl_main/out/main_out \
#     --eval_dir /root/mysapce/workspace/AVA-qwen2_5_vl_main/eval_results \
#     --from_weight main-finetune-ablate2








export CUDA_VISIBLE_DEVICES=4,6,7


cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/data/images \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-1-3e-5.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.50

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa
python convert_gqa_for_eval.py --src /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-1-3e-5.jsonl --dst /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/data/testdev_balanced_predictions.json

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/data
echo "GQA, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.50" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/results.txt
python eval.py --tier testdev_balanced >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/results.txt


cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/gqa/data/images \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-1-3e-5.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.75

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa
python convert_gqa_for_eval.py --src /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/answers/v1-298M-data-epoch-1-3e-5.jsonl --dst /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/data/testdev_balanced_predictions.json

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/data
echo "GQA, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.75" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/results.txt
python eval.py --tier testdev_balanced >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/gqa/results.txt








cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/pope/val2014 \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.30

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope
echo "POPE, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.30" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt
python eval_pope.py --annotation-dir /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/coco --question-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/pope/val2014 \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.50

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope
echo "POPE, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.50" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt
python eval_pope.py --annotation-dir /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/coco --question-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/pope/val2014 \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.75

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope
echo "POPE, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.75" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt
python eval_pope.py --annotation-dir /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/coco --question-file /mnt/inaisfs/home/test3/jihuawei/projects/LLaVA-FastV/playground/data/eval/pope/llava_pope_test.jsonl --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/answers/v1-298M-data-epoch-1-3e-5.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/pope/results.txt










cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/PyramidDrop/playground/data/eval/seed_bench/llava-seed-bench-existing.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/seed_bench \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.75.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.75

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench
python convert_seed_for_submission.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.75.jsonl \
    --result-upload-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers_upload/v1-298M-data-epoch-1-3e-5-0.75.jsonl


cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/PyramidDrop/playground/data/eval/seed_bench/llava-seed-bench-existing.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/seed_bench \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.50.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.50

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench
python convert_seed_for_submission.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.50.jsonl \
    --result-upload-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers_upload/v1-298M-data-epoch-1-3e-5-0.50.jsonl

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/PyramidDrop/playground/data/eval/seed_bench/llava-seed-bench-existing.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/visionzip/LLaVA/playground/data/eval/seed_bench \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.30.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.30

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench
python convert_seed_for_submission.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers/v1-298M-data-epoch-1-3e-5-0.30.jsonl \
    --result-upload-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/seed_bench/answers_upload/v1-298M-data-epoch-1-3e-5-0.30.jsonl












cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/train_images \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.30.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.30
cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa
echo "TextVQA, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.30" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt
python eval_textvqa.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.30.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/train_images \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.50.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.50
cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa
echo "TextVQA, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.50" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt
python eval_textvqa.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.50.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt

cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation
python -m torch.distributed.run --nproc_per_node=3 eval_benchmarks.py \
    --question_file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image_folder /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/train_images \
    --answer_file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.75.jsonl \
    --from_weight v1-298M-data-epoch-1-3e-5 \
    --train_drop_rate 0.75
cd /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa
echo "TextVQA, 298M data, 3 epoch, lr 3e-5, batch 4, train_drop_rate 0.75" >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt
python eval_textvqa.py \
    --annotation-file /mnt/inaisfs/home/test3/40618/LLaVA-PruMerge/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/answers/v1-298M-data-epoch-1-3e-5-0.75.jsonl >> /mnt/inaisfs/home/test2/163879/workspace/AVA-qwen2_5_vl_main/evaluation/playground/data/eval/textvqa/results.txt

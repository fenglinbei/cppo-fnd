export SWANLAB_PROJECT="factcheck-grpo"
export NCCL_CUMEM_HOST_ENABLE=0 
export NCCL_CUMEM_ENABLE=0 
accelerate launch  --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=3  \
    --main_process_port 10909 \
    src/open_r1/grpo_fnd.py \
    --config recipes/fnd/Qwen3-0.6B.yaml \
    --output_dir=output/fnd/ \
    --save_strategy='best' \
    --eval_steps=100 --max_completion_length=1024 \
    --model_name_or_path=./models/Qwen3-0.6B \
    --dataset_name=data/liar-raw \
    --num_generations=4 \
    --log_level=debug \
    --dataset_train_split=train
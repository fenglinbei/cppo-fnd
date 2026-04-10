export WANDB_PROJECT="factcheck-grpo"
export WANDB_API_KEY="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"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL,GRAPH
# export NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_ASYNC_ERROR_HANDLING
unset CUDA_LAUNCH_BLOCKING
unset TORCH_DISTRIBUTED_DEBUG
unset TORCH_SHOW_CPP_STACKTRACES
unset NCCL_IB_DISABLE
unset NCCL_SHM_DISABLE
export NCCL_P2P_DISABLE=1
accelerate launch  --config_file recipes/accelerate_configs/ddp.yaml \
    --num_processes=3  \
    --main_process_port 10909 \
    src/open_r1/grpo_fnd_v2.py \
    --config recipes/fnd/Qwen2.5-3B-Instruct_v2.yaml \
    --output_dir=output/fnd/Qwen-2.5-3B_v2/ \
    --save_strategy='best' \
    --eval_steps=500 --max_completion_length=1024 \
    --model_name_or_path=./models/Qwen2.5-3B-Instruct \
    --dataset_name=data/liar-raw \
    --num_generations=12 \
    --log_level=info \
    --dataset_train_split=train
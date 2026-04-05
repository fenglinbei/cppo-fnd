export VLLM_SERVER_DEV_MODE=1
export CUDA_VISIBLE_DEVICES=3

vllm serve ./models/Qwen3-0.6B \
  --served-model-name live-policy \
  --host 0.0.0.0 \
  --port 10909 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --weight-transfer-config '{"backend":"nccl"}' \
  --load-format dummy
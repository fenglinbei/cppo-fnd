export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  --nproc-per-node=3 \
  test/test_nccl.py 
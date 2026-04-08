# save as test_nccl_broadcast.py
import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    # 约 445 MiB 的 uint8 tensor，接近你日志里的大 broadcast 量级
    x = torch.empty(466_747_392, dtype=torch.uint8, device=f"cuda:{local_rank}")
    if rank == 0:
        x.fill_(1)

    print(f"[before broadcast] rank={rank}, device={torch.cuda.current_device()}", flush=True)
    dist.broadcast(x, src=0)
    torch.cuda.synchronize()
    print(f"[after broadcast] rank={rank}, sum={x.sum().item()}", flush=True)

if __name__ == "__main__":
    main()
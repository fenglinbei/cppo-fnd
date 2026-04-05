import os
import torch
from vllm import LLM, SamplingParams

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i))
    print("mem_get_info(i) =", torch.cuda.mem_get_info(i))

vllm_device = "cuda:2"   # 这里一定要写“进程内本地索引”
torch.cuda.set_device(vllm_device)
print("current_device =", torch.cuda.current_device())
print("current mem_get_info =", torch.cuda.mem_get_info())

llm = LLM(
    model="./models/Qwen3-0.6B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    device=vllm_device,
    model_impl="transformers",
    gpu_memory_utilization=0.9
)

outs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
print(outs[0].outputs[0].text)
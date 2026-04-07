import os
import torch
from vllm import LLM, SamplingParams

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())

assert torch.cuda.device_count() == 2
print("device 0 =", torch.cuda.get_device_name(0))
print("mem_get_info(0) =", torch.cuda.mem_get_info(0))

torch.cuda.set_device(0)
x = torch.zeros(1, device="cuda:0")
print("tensor device =", x.device)
print("current_device =", torch.cuda.current_device())
print("current mem_get_info =", torch.cuda.mem_get_info())

try:
    llm = LLM(
        model="./models/Qwen3-0.6B",
        tensor_parallel_size=1,
        trust_remote_code=True,
        device="cuda:0",
        model_impl="transformers",
        gpu_memory_utilization=0.5,
        distributed_executor_backend="uni",
        enforce_eager=True,
    )
    
    outs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    print(outs[0].outputs[0].text)
    print("指定GPU成功")
except:
    print("指定GPU失败")

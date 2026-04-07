import os
import torch
from vllm import LLM, SamplingParams

llm = LLM(
        model="./models/Qwen2.5-1.5B-Instruct",
        tensor_parallel_size=1,
        trust_remote_code=True,
        device="cuda",
        dtype=self.args.vllm_dtype,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        max_model_len=self.args.vllm_max_model_len
    )

outs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
print(outs[0].outputs[0].text)
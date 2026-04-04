from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/Qwen3-0.6B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_impl="transformers",
    gpu_memory_utilization=0.5
)

outs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
print(outs[0].outputs[0].text)
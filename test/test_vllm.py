from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/Qwen3-0.6B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_impl="transformers",
)

outs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
print(outs[0].outputs[0].text)
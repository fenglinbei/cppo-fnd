python - <<'PY'
import torch, vllm, transformers, accelerate
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("vllm =", vllm.__version__)
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
PY

nvidia-smi
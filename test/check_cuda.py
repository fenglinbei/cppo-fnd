import os
import re
import sys
import glob
import ctypes
import subprocess
from ctypes.util import find_library


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def format_cuda_version(v):
    """
    CUDA version integer -> string
    e.g. 12040 -> 12.4
         11080 -> 11.8
    """
    if v <= 0:
        return "unknown"
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"


def find_cudart_candidates():
    candidates = []

    # 1) ctypes 默认查找
    lib = find_library("cudart")
    if lib:
        candidates.append(lib)

    # 2) 常见环境变量路径
    env_vars = ["CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "VIRTUAL_ENV"]
    for env_name in env_vars:
        base = os.environ.get(env_name)
        if not base:
            continue
        patterns = [
            os.path.join(base, "lib64", "libcudart.so*"),
            os.path.join(base, "lib", "libcudart.so*"),
            os.path.join(base, "Library", "bin", "cudart64*.dll"),  # Windows
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))

    # 3) LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    for p in ld_library_path.split(":"):
        if p:
            candidates.extend(glob.glob(os.path.join(p, "libcudart.so*")))

    # 4) 系统常见路径
    common_patterns = [
        "/usr/local/cuda/lib64/libcudart.so*",
        "/usr/local/cuda-*/lib64/libcudart.so*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so*",
    ]
    for pattern in common_patterns:
        candidates.extend(glob.glob(pattern))

    # 去重，保留顺序
    uniq = []
    seen = set()
    for x in candidates:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def load_cudart():
    candidates = find_cudart_candidates()
    last_err = None

    for libpath in candidates:
        try:
            lib = ctypes.CDLL(libpath)
            return lib, libpath
        except Exception as e:
            last_err = e

    return None, f"Failed to load libcudart. Last error: {last_err}"


def get_cudart_versions():
    """
    通过 CUDA runtime API 获取：
    - cudaRuntimeGetVersion: 当前进程实际加载的 cudart 版本
    - cudaDriverGetVersion: 当前驱动 API 版本
    """
    lib, info = load_cudart()
    if lib is None:
        return {
            "cudart_path": None,
            "runtime_version_raw": None,
            "runtime_version": None,
            "driver_version_raw": None,
            "driver_version": None,
            "error": info,
        }

    runtime_v = ctypes.c_int()
    driver_v = ctypes.c_int()

    try:
        f_runtime = lib.cudaRuntimeGetVersion
        f_runtime.argtypes = [ctypes.POINTER(ctypes.c_int)]
        f_runtime.restype = ctypes.c_int

        f_driver = lib.cudaDriverGetVersion
        f_driver.argtypes = [ctypes.POINTER(ctypes.c_int)]
        f_driver.restype = ctypes.c_int

        err1 = f_runtime(ctypes.byref(runtime_v))
        err2 = f_driver(ctypes.byref(driver_v))

        return {
            "cudart_path": info,
            "runtime_version_raw": runtime_v.value if err1 == 0 else None,
            "runtime_version": format_cuda_version(runtime_v.value) if err1 == 0 else None,
            "driver_version_raw": driver_v.value if err2 == 0 else None,
            "driver_version": format_cuda_version(driver_v.value) if err2 == 0 else None,
            "error": None if (err1 == 0 and err2 == 0) else f"cuda api call failed: err1={err1}, err2={err2}",
        }
    except Exception as e:
        return {
            "cudart_path": info,
            "runtime_version_raw": None,
            "runtime_version": None,
            "driver_version_raw": None,
            "driver_version": None,
            "error": str(e),
        }


def parse_nvidia_smi():
    code, out, err = run_cmd(["nvidia-smi"])
    if code != 0:
        return {"ok": False, "error": err or out}

    # 典型格式里会出现 "CUDA Version: 12.4"
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    cuda_version = m.group(1) if m else None

    m2 = re.search(r"Driver Version:\s*([\d.]+)", out)
    driver_version = m2.group(1) if m2 else None

    return {
        "ok": True,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "raw": out,
    }


def parse_nvcc():
    code, out, err = run_cmd(["nvcc", "--version"])
    text = out if out else err
    if code != 0:
        return {"ok": False, "error": text}

    # 典型格式: release 12.4, V12.4.131
    m = re.search(r"release\s+([\d.]+)", text)
    version = m.group(1) if m else None

    return {
        "ok": True,
        "nvcc_version": version,
        "raw": text,
    }


def check_torch_cuda():
    result = {
        "torch_installed": False,
        "torch_version": None,
        "torch_compiled_cuda": None,
        "cuda_available": None,
        "device_count": None,
        "current_device": None,
        "device_name": None,
        "capability": None,
        "test_op_ok": False,
        "error": None,
    }

    try:
        import torch

        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["torch_compiled_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            result["device_count"] = torch.cuda.device_count()
            result["current_device"] = torch.cuda.current_device()
            result["device_name"] = torch.cuda.get_device_name(result["current_device"])
            result["capability"] = torch.cuda.get_device_capability(result["current_device"])

            # 强制做一次实际 CUDA 运算，确认当前环境真的在跑 CUDA
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            z = y.mean()
            torch.cuda.synchronize()
            result["test_op_ok"] = True
            result["test_value"] = float(z.item())
        else:
            result["error"] = "torch.cuda.is_available() == False"

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("=" * 80)
    print("Python / CUDA environment inspection")
    print("=" * 80)
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")
    print()

    # 1) PyTorch
    torch_info = check_torch_cuda()
    print("[PyTorch]")
    if not torch_info["torch_installed"]:
        print("  torch: not installed")
    else:
        print(f"  torch.__version__         : {torch_info['torch_version']}")
        print(f"  torch.version.cuda        : {torch_info['torch_compiled_cuda']}")
        print(f"  torch.cuda.is_available() : {torch_info['cuda_available']}")
        print(f"  device_count              : {torch_info['device_count']}")
        print(f"  current_device            : {torch_info['current_device']}")
        print(f"  device_name               : {torch_info['device_name']}")
        print(f"  capability                : {torch_info['capability']}")
        print(f"  test_op_ok                : {torch_info['test_op_ok']}")
        if torch_info.get("error"):
            print(f"  error                     : {torch_info['error']}")
    print()

    # 2) 当前进程实际加载的 cudart
    cudart_info = get_cudart_versions()
    print("[CUDA Runtime API via libcudart]")
    print(f"  libcudart path            : {cudart_info['cudart_path']}")
    print(f"  cudaRuntimeGetVersion     : {cudart_info['runtime_version']} ({cudart_info['runtime_version_raw']})")
    print(f"  cudaDriverGetVersion      : {cudart_info['driver_version']} ({cudart_info['driver_version_raw']})")
    if cudart_info["error"]:
        print(f"  error                     : {cudart_info['error']}")
    print()

    # 3) nvidia-smi
    smi_info = parse_nvidia_smi()
    print("[nvidia-smi]")
    if smi_info["ok"]:
        print(f"  Driver Version            : {smi_info['driver_version']}")
        print(f"  CUDA Version              : {smi_info['cuda_version']}")
    else:
        print(f"  error                     : {smi_info['error']}")
    print()

    # 4) nvcc
    nvcc_info = parse_nvcc()
    print("[nvcc --version]")
    if nvcc_info["ok"]:
        print(f"  nvcc release              : {nvcc_info['nvcc_version']}")
    else:
        print(f"  error                     : {nvcc_info['error']}")
    print()

    print("=" * 80)
    print("Interpretation")
    print("=" * 80)
    print("1) 最值得看的是 cudaRuntimeGetVersion：它表示当前进程实际加载到的 CUDA Runtime 版本。")
    print("2) torch.version.cuda 表示 PyTorch 编译时绑定的 CUDA 版本，不一定等于你机器安装的 toolkit 版本。")
    print("3) nvidia-smi 里的 CUDA Version 通常表示驱动最高支持的 CUDA API 版本，不等于实际运行时版本。")
    print("4) nvcc --version 只是本机编译器/toolkit 版本；很多推理/训练环境即使没有 nvcc 也能正常跑。")
    print("5) 如果 test_op_ok=True，说明当前 Python 进程确实完成了一次 CUDA 张量计算。")


if __name__ == "__main__":
    main()
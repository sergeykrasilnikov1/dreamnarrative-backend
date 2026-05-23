"""
Проверка совместимости PyTorch CUDA с GPU (Tesla V100 / sm_70 vs torch cu130).
"""
from __future__ import annotations


def torch_cuda_kernel_usable() -> bool:
    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability(0)
    if hasattr(torch.cuda, "get_arch_list"):
        archs = torch.cuda.get_arch_list()
        sm = f"sm_{major}{minor}"
        if sm not in archs and major == 7 and minor == 0:
            return False

    try:
        x = torch.ones(2, 2, device="cuda")
        x.mul_(2)
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def resolve_torch_device(preference: str = "auto") -> str:
    pref = (preference or "auto").lower().strip()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not torch_cuda_kernel_usable():
            raise RuntimeError(_cuda_unusable_message())
        return "cuda"
    if torch_cuda_kernel_usable():
        return "cuda"
    return "cpu"


def _cuda_unusable_message() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            ver = torch.__version__
            archs = (
                ", ".join(torch.cuda.get_arch_list())
                if hasattr(torch.cuda, "get_arch_list")
                else "n/a"
            )
            return (
                f"PyTorch {ver} не содержит CUDA-ядер для {name} (CC {cap[0]}.{cap[1]}). "
                f"Сборка: {archs}. "
                "На Tesla V100: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
    except ImportError:
        pass
    return "CUDA недоступна или PyTorch собран без поддержки этого GPU."


def cuda_device_summary() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False, "kernels_usable": False}

    summary = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "kernels_usable": torch_cuda_kernel_usable(),
    }
    if torch.cuda.is_available():
        summary["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        summary["compute_capability"] = f"{cap[0]}.{cap[1]}"
        if hasattr(torch.cuda, "get_arch_list"):
            summary["torch_arch_list"] = torch.cuda.get_arch_list()
    if summary["cuda_available"] and not summary["kernels_usable"]:
        summary["hint"] = _cuda_unusable_message()
    return summary

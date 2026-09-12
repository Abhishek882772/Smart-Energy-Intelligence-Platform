"""Device Management and Environment Logging for NILM Training Pipeline."""

from typing import Any, Dict, Optional, Union
import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve computation device with automatic CUDA detection.

    Args:
        device: Explicit device string or torch.device (optional).

    Returns:
        torch.device: Resolved device ('cuda' if available else 'cpu').
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader_kwargs(
    device: Optional[Union[str, torch.device]] = None,
    num_workers_cuda: int = 2,
    num_workers_cpu: int = 0,
) -> Dict[str, Any]:
    """Return device-optimized DataLoader keyword arguments.

    When CUDA is used:
        pin_memory=True, num_workers=2
    When CPU is used:
        pin_memory=False, num_workers=0

    Args:
        device: Optional device identifier.
        num_workers_cuda: Worker processes for CUDA (default: 2).
        num_workers_cpu: Worker processes for CPU (default: 0).

    Returns:
        dict: Keyword arguments for torch.utils.data.DataLoader.
    """
    target_device = get_device(device)
    if target_device.type == "cuda":
        return {
            "pin_memory": True,
            "num_workers": num_workers_cuda,
        }
    return {
        "pin_memory": False,
        "num_workers": num_workers_cpu,
    }


def log_environment_info(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """Log and return system, PyTorch, and CUDA environment metadata.

    Args:
        device: Optional target device.

    Returns:
        dict: Detailed environment metadata dictionary.
    """
    target_device = get_device(device)
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda if cuda_available else None
    device_count = torch.cuda.device_count() if cuda_available else 0

    info = {
        "selected_device": str(target_device),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "pytorch_version": torch.__version__,
    }

    print("=" * 80)
    print("COMPUTE ENVIRONMENT & DEVICE DIAGNOSTICS")
    print("=" * 80)
    print(f"  Selected Device : {info['selected_device']}")
    print(f"  CUDA Available  : {info['cuda_available']}")
    if cuda_available:
        print(f"  GPU Name        : {info['gpu_name']}")
        print(f"  CUDA Version    : {info['cuda_version']}")
        print(f"  Device Count    : {info['device_count']}")
    else:
        print("  GPU Name        : N/A (CPU execution mode)")
    print(f"  PyTorch Version : {info['pytorch_version']}")
    print("=" * 80)

    return info

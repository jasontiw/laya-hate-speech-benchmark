"""Hardware and runtime introspection (PRD section 18).

Every number in the report that depends on the machine is recorded here, so a
reader can tell whether a latency figure is reproducible on their own hardware.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import platform
import sys
from typing import Any, Dict, Optional


def _ram_bytes() -> Optional[int]:
    """Total physical RAM in bytes, best effort, no third-party dependency."""
    try:
        if sys.platform == "win32":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _module_version(name: str) -> Optional[str]:
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:
        return None


def runtime_info() -> Dict[str, Any]:
    """A JSON-serialisable snapshot of the environment a run executed in."""
    ram = _ram_bytes()
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram / 1e9, 2) if ram else None,
        "packages": {
            "torch": _module_version("torch"),
            "transformers": _module_version("transformers"),
            "laya": _module_version("laya"),
            "scikit_learn": _module_version("sklearn"),
            "pandas": _module_version("pandas"),
            "numpy": _module_version("numpy"),
        },
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        info.setdefault("cuda_available", None)
    return info


def resolve_device(preferred: Optional[str]) -> str:
    """Pick an inference device: explicit value, else CUDA, else MPS, else CPU."""
    if preferred:
        return preferred
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

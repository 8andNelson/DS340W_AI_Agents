"""
This computer's hardware, and whether a reconstructed baseline can run on it
(Replication Agent).

There is no time limit on a replication: if the Parent Paper's method needs
hours, it gets hours. What can make a run impossible is the computer itself:
not enough memory, no GPU when the method truly needs one, or not enough
free disk. Before a reconstructed script runs, the agent compares what the
script needs (the model's own estimate, `resources`, plus a deterministic
estimate of the data's size in memory) with profile(). When it does not fit,
the agent does not run the script: it stops early and says why
(feasibility), and Master Agent does not retry it, since a retry cannot add
hardware. A long estimated run time is recorded and shown, never a reason
to stop.

Every function here is fail-soft: an unknown value is None, and an unknown
value never makes a run infeasible.
"""
import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3
# Libraries that can use an NVIDIA GPU on native Windows (gpu_use). PyTorch can
# because the Replication Agent installs its CUDA build (torch_cuda_index).
GPU_LIBRARIES_ON_WINDOWS = ("xgboost", "catboost", "torch", "pytorch")
# PyTorch's CUDA builds, newest first: (index name, CUDA version). A build is
# used only when the driver supports its CUDA version. Checked live
# (2026-11-18): cu130 and cu126 have torch 2.14.1 for Python 3.14 on Windows.
TORCH_CUDA_BUILDS = (("cu130", 13.0), ("cu128", 12.8), ("cu126", 12.6), ("cu124", 12.4), ("cu121", 12.1),
                     ("cu118", 11.8))
TORCH_INDEX = "https://download.pytorch.org/whl/{build}"
# The data in memory is copied several times (features, a scaled copy, the
# resampled copy). Only when this many copies do not fit is it infeasible.
DATA_COPIES = 3
BYTES_PER_VALUE = 8


def _ram_gb() -> tuple[float | None, float | None]:
    """(total, available) memory in GB, or None for an unknown value."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / GIB, 1), round(status.ullAvailPhys / GIB, 1)
            return None, None
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
        available = None
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024 / GIB
        return round(total, 1), round(available, 1) if available is not None else None
    except Exception:
        return None, None


def _gpu() -> dict:
    """An NVIDIA GPU's name, memory (GB) and the highest CUDA version its
    driver supports, from nvidia-smi, or {} for none found. Other GPUs are
    not detected, so "none found" is not proof."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        line = (out.stdout or "").strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
        name, _, memory = line.partition(",")
        if not name.strip():
            return {}
        gpu = {"name": name.strip(), "memory_gb": round(float(memory) / 1024, 1)}
        header = subprocess.run([smi], capture_output=True, text=True, timeout=15).stdout or ""
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header)
        if match:
            gpu["cuda_version"] = float(match.group(1))
        return gpu
    except Exception:
        return {}


def torch_cuda_off() -> bool:
    """TORCH_CUDA=off (.env): always install PyTorch's CPU build, e.g. to skip
    the 2-3 GB CUDA download. The default, auto, uses CUDA when it fits."""
    return os.environ.get("TORCH_CUDA", "auto").strip().lower() in ("off", "false", "0", "no", "cpu")


def has_gpu(hw: dict) -> bool:
    """An NVIDIA GPU was found (the REPLICATION_GPU a script is given)."""
    return bool((hw.get("gpu") or {}).get("name"))


def torch_cuda_index(hw: dict) -> str:
    """The PyTorch index URL of the newest CUDA build the driver supports,
    or "" when there is no NVIDIA GPU, its CUDA version is unknown or too
    old, or TORCH_CUDA=off."""
    if torch_cuda_off():
        return ""
    cuda = (hw.get("gpu") or {}).get("cuda_version")
    if not isinstance(cuda, (int, float)):
        return ""
    build = next((name for name, version in TORCH_CUDA_BUILDS if version <= cuda), "")
    return TORCH_INDEX.format(build=build) if build else ""


def profile(work_dir: Path | None = None) -> dict:
    """This computer: CPU cores, memory, NVIDIA GPU, free disk where the run writes."""
    total, available = _ram_gb()
    try:
        disk = round(shutil.disk_usage(str(work_dir or Path.cwd())).free / GIB, 1)
    except Exception:
        disk = None
    return {"cpu_cores": os.cpu_count(), "ram_total_gb": total, "ram_available_gb": available,
            "gpu": _gpu(), "disk_free_gb": disk}


def gpu_use(hw: dict) -> str:
    """Which of the pipeline's libraries can use this computer's GPU. The
    pip packages it installs decide this: on native Windows, TensorFlow
    (2.11 and later) has no GPU support, and PyTorch from PyPI is the
    CPU-only build; XGBoost's and CatBoost's wheels include CUDA."""
    if not (hw.get("gpu") or {}).get("name"):
        return "no GPU: everything runs on the CPU"
    if os.name == "nt":
        torch = ("PyTorch can (its CUDA build is installed: device = \"cuda\" if torch.cuda.is_available() "
                 "else \"cpu\")" if torch_cuda_index(hw) else
                 "PyTorch cannot (TORCH_CUDA=off: its CPU build is installed)" if torch_cuda_off() else
                 "PyTorch cannot (no CUDA build fits this driver)")
        return ("XGBoost can use it (device=\"cuda\"), CatBoost can (task_type=\"GPU\") and " + torch
                + "; TensorFlow cannot on Windows. When the paper does not require a specific library, write "
                  "neural networks in PyTorch (the same architecture), so they use the GPU")
    return ("XGBoost (device=\"cuda\"), CatBoost (task_type=\"GPU\"), TensorFlow and PyTorch can use it when their "
            "CUDA libraries are present")


def describe(hw: dict) -> str:
    """The profile in one line, for prompts and reports."""
    gpu = hw.get("gpu") or {}
    return (f"{hw.get('cpu_cores') or 'unknown'} CPU cores; "
            f"{hw.get('ram_total_gb') or 'unknown'} GB RAM ({hw.get('ram_available_gb') or 'unknown'} GB free now); "
            + (f"GPU: {gpu['name']} ({gpu.get('memory_gb')} GB)" if gpu.get("name") else "no NVIDIA GPU found")
            + f"; {hw.get('disk_free_gb') or 'unknown'} GB free disk. GPU use: {gpu_use(hw)}")


def data_memory_gb(cleaning_result: dict | None) -> float | None:
    """The master dataset in memory as 8-byte values, in GB, or None."""
    rows = (cleaning_result or {}).get("rows_total")
    cols = (cleaning_result or {}).get("columns_total")
    if not isinstance(rows, int) or not isinstance(cols, int):
        return None
    return round(rows * cols * BYTES_PER_VALUE / GIB, 3)


def _number(value) -> float | None:
    try:
        return float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def feasibility(resources: dict | None, hw: dict, cleaning_result: dict | None = None) -> list:
    """Why this computer cannot run the script, or [] when it can (or when a
    limit is unknown). resources is the model's estimate: estimated_ram_gb,
    needs_gpu with gpu_reason and gpu_library, estimated_disk_gb."""
    resources = resources if isinstance(resources, dict) else {}
    reasons = []
    total_ram = hw.get("ram_total_gb")
    data_gb = data_memory_gb(cleaning_result)
    if total_ram and data_gb is not None and data_gb * DATA_COPIES > total_ram:
        reasons.append(f"the data alone needs about {round(data_gb * DATA_COPIES, 1)} GB of memory "
                       f"({DATA_COPIES} copies of {data_gb} GB), and this computer has {total_ram} GB")
    ram = _number(resources.get("estimated_ram_gb"))
    if total_ram and ram is not None and ram > total_ram:
        reasons.append(f"the method needs about {ram} GB of memory, and this computer has {total_ram} GB")
    if resources.get("needs_gpu") is True:
        why = str(resources.get("gpu_reason") or "").strip()
        library = str(resources.get("gpu_library") or "").strip().lower()
        if not (hw.get("gpu") or {}).get("name"):
            reasons.append("the method needs a GPU" + (f" ({why})" if why else "") + ", and no GPU was found")
        elif os.name == "nt" and library not in GPU_LIBRARIES_ON_WINDOWS:
            reasons.append("the method needs a GPU" + (f" ({why})" if why else "")
                           + f" for {library or 'its neural networks'}, and on Windows TensorFlow and the PyTorch "
                             "installed here cannot use this computer's GPU")
    disk = _number(resources.get("estimated_disk_gb"))
    if hw.get("disk_free_gb") is not None and disk is not None and disk > hw["disk_free_gb"]:
        reasons.append(f"the run needs about {disk} GB of disk, and {hw['disk_free_gb']} GB is free")
    return reasons

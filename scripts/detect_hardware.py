#!/usr/bin/env python3
"""Detect CPU count, RAM, and GPU/CUDA availability on this machine.
Writes verification/hardware_report.json. Pure inspection -- does not
change any scientific configuration, does not launch training."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import VERIFICATION, python_exe, write_json_atomic  # noqa: E402


def main() -> int:
    import multiprocessing
    try:
        import psutil  # optional; not in the frozen requirements, so guard it
        ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        ram_total_gb = None
        ram_available_gb = None

    report = {
        "cpu_count_logical": multiprocessing.cpu_count(),
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "python_exe": python_exe(),
    }

    try:
        import torch
        report["torch_version"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            report["cuda_total_vram_gb"] = props.total_memory / (1024 ** 3)
            report["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        report["torch_import_error"] = "torch not installed yet -- run 00_SETUP first"

    write_json_atomic(VERIFICATION / "hardware_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

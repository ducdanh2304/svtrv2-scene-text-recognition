#!/usr/bin/env python3
"""Preinitialize CUDA with temporary CPU affinity.

The original benchmark script and its SHA256 remain unchanged.
"""

import hashlib
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "evaluate_unified_e2e_v1.py"


def main():
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)

    print("=== PYTORCH CUDA STARTUP GUARD ===", flush=True)

    allowed = os.sched_getaffinity(0)
    cpu = min(allowed)

    print("Allowed CPUs:", len(allowed), flush=True)
    print("Temporary startup CPU:", cpu, flush=True)

    try:
        os.sched_setaffinity(0, {cpu})

        import torch

        print("Torch:", torch.__version__, flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; benchmark not started"
            )

        torch.cuda.init()

        x = torch.ones(
            (1,), device="cuda:0"
        )
        torch.cuda.synchronize()

        assert float(x.item()) == 1.0

        print(
            "GPU:", torch.cuda.get_device_name(0),
            flush=True
        )
        print("CUDA INITIALIZATION: PASS", flush=True)

    finally:
        os.sched_setaffinity(0, allowed)

    # Restore CPU parallelism for inference.
    import torch

    threads = min(8, len(allowed))
    torch.set_num_threads(threads)

    print("CPU affinity: RESTORED", flush=True)
    print("PyTorch CPU threads:", threads, flush=True)

    print(
        "Original benchmark SHA256:",
        hashlib.sha256(
            RUNNER.read_bytes()
        ).hexdigest(),
        flush=True
    )

    print(
        "Launcher SHA256:",
        hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        flush=True
    )

    print("Starting original benchmark...", flush=True)

    # Preserve the benchmark's original __file__,
    # checkpoint/config identity and resume logic.
    sys.argv = [str(RUNNER), *sys.argv[1:]]

    runpy.run_path(
        str(RUNNER),
        run_name="__main__"
    )


if __name__ == "__main__":
    main()

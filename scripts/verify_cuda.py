#!/usr/bin/env python3
"""GPU-acceleration smoke test, per the migration's own explicit
requirement: never assume CUDA is faster for this workload without
measuring, and verify device handling changes NOTHING about the
scientific configuration (architecture, batch size, optimizer,
determinism-relevant behaviour).

Runs the SAME frozen QNetwork(18 -> 64 -> 64 -> 3) on CPU and (if
available) GPU:
  1. forward + backward pass, batch_size=64 (the frozen batch size --
     this smoke test does NOT change it to make GPU look better);
  2. checkpoint save/load round-trip on each device;
  3. NaN/Inf check;
  4. wall-clock timing for N repeated update steps, honestly reported.

Writes verification/cuda_report.json. Does not touch any real
training checkpoint. Safe to run any time."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from _common import PROJECT_ROOT, VERIFICATION, load_frozen_config, write_json_atomic  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _build_net(cfg, device):
    import torch
    from thesis.agents.independent_dqn_v2 import QNetwork
    net = QNetwork(obs_dim=cfg["dqn"]["obs_dim"], n_actions=cfg["dqn"]["n_actions"],
                   hidden_sizes=tuple(cfg["dqn"]["hidden_sizes"])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["dqn"]["lr_start"])
    return net, opt


def _smoke_one_device(cfg, device_str: str, n_iters: int = 200) -> dict:
    import torch
    device = torch.device(device_str)
    net, opt = _build_net(cfg, device)
    batch_size = cfg["dqn"]["batch_size"]
    obs_dim = cfg["dqn"]["obs_dim"]
    n_actions = cfg["dqn"]["n_actions"]

    torch.manual_seed(0)
    obs = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randint(0, n_actions, (batch_size,), device=device)
    targets = torch.randn(batch_size, device=device)

    # forward + backward, exactly the frozen batch size -- no vectorization/
    # batch-size change to "use the GPU better"
    t0 = time.perf_counter()
    nan_or_inf = False
    for _ in range(n_iters):
        opt.zero_grad()
        q = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(q, targets)
        loss.backward()
        opt.step()
        if not torch.isfinite(loss).item():
            nan_or_inf = True
    if device_str == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # checkpoint save/load round-trip on this device
    ckpt_path = VERIFICATION / f"_cuda_smoke_ckpt_{device_str.replace(':', '_')}.pt"
    torch.save({"online": net.state_dict(), "optimiser": opt.state_dict(), "step": 12345}, ckpt_path)
    reloaded = torch.load(ckpt_path, map_location=device)
    net2, _ = _build_net(cfg, device)
    net2.load_state_dict(reloaded["online"])
    with torch.no_grad():
        out_before = net(obs)
        out_after = net2(obs)
    checkpoint_roundtrip_ok = bool(torch.allclose(out_before, out_after, atol=1e-6))
    ckpt_path.unlink(missing_ok=True)

    return {
        "device": device_str,
        "iterations": n_iters,
        "batch_size": batch_size,
        "wall_seconds": elapsed,
        "seconds_per_iter": elapsed / n_iters,
        "nan_or_inf_detected": nan_or_inf,
        "checkpoint_roundtrip_ok": checkpoint_roundtrip_ok,
        "observation_dim": obs.shape[1],
        "action_dim": n_actions,
    }


def main() -> int:
    import torch
    cfg = load_frozen_config()
    report: dict = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}

    cpu_result = _smoke_one_device(cfg, "cpu")
    report["cpu"] = cpu_result
    print(json.dumps(cpu_result, indent=2))

    if torch.cuda.is_available():
        gpu_result = _smoke_one_device(cfg, "cuda")
        report["gpu"] = gpu_result
        print(json.dumps(gpu_result, indent=2))
        speedup = cpu_result["seconds_per_iter"] / gpu_result["seconds_per_iter"]
        report["gpu_speedup_factor_vs_cpu"] = speedup
        report["recommendation"] = (
            "GPU is faster for this workload -- consider device=cuda for training." if speedup > 1.2 else
            "GPU is NOT meaningfully faster for this tiny (18->64->64->3) network at batch_size="
            f"{cfg['dqn']['batch_size']} -- the highway simulation itself is CPU-bound and kernel-launch "
            "overhead likely dominates any GPU matmul benefit. Default recommendation: stay on CPU and use "
            "process-level parallelism (see launch_formal.py) instead."
        )
        print(f"\nGPU speedup factor vs CPU: {speedup:.3f}x")
        print(report["recommendation"])
    else:
        report["gpu"] = None
        report["recommendation"] = "No CUDA device detected by torch. Either no GPU/driver, or the CPU-only " \
            "torch wheel is installed (see wheelhouse/gpu/ and README.md Section 3 for the CUDA install path)."
        print(report["recommendation"])

    write_json_atomic(VERIFICATION / "cuda_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

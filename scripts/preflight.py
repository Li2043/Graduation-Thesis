#!/usr/bin/env python3
"""Preflight validation: imports, environment reset/step, observation
shape, action space, and the study_b test suite. Does NOT run formal
training. Safe to run repeatedly. Writes verification/preflight_report.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from _common import PROJECT_ROOT, VERIFICATION, load_frozen_config, python_exe, write_json_atomic  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def check_imports() -> dict:
    errors = []
    modules = [
        "thesis.study_b.local_observation",
        "thesis.study_b.envs.highwayenv_wrapper",
        "thesis.study_b.envs.highwayenv_merge",
        "thesis.study_b.welfare_reward",
        "thesis.study_b.utility",
        "thesis.agents.independent_dqn_v2",
        "thesis.agents.stage10_shared_dqn",
        "thesis.study_b.shared_local_dqn",
        "thesis.study_b.q_ensemble",
        "highway_env", "gymnasium", "torch", "numpy",
    ]
    for m in modules:
        try:
            __import__(m)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{m}: {e!r}")
    return {"modules_checked": len(modules), "import_errors": errors, "ok": not errors}


def check_env_and_observation(cfg: dict) -> dict:
    from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
    from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig

    r = cfg["observation"]["local_sensing_range_m"]
    env_cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=cfg["environment"]["episode_max_steps"],
                                           action_representation=cfg["environment"]["action_representation"])
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_cfg, local_sensing_range_m=r))
    obs, _info = env.reset(seed=0)
    dims = {vid: int(o.shape[0]) for vid, o in obs.items()}
    expected = cfg["observation"]["local_obs_dim"]
    ok_dims = all(d == expected for d in dims.values())

    actions = {vid: 0 for vid in env.active_vehicle_ids}
    obs2, reward, terminated, truncated, step_info = env.step(actions)
    step_ok = isinstance(reward, dict) or True  # reward shape not asserted here, just that step() didn't crash

    return {
        "observation_dims": dims, "expected_dim": expected, "observation_dim_ok": ok_dims,
        "n_active_vehicles_at_reset": len(env.active_vehicle_ids),
        "one_step_ok": step_ok, "local_sensing_range_m_used": r,
    }


def check_q_network(cfg: dict) -> dict:
    import torch
    from thesis.agents.independent_dqn_v2 import QNetwork
    net = QNetwork(obs_dim=cfg["dqn"]["obs_dim"], n_actions=cfg["dqn"]["n_actions"],
                   hidden_sizes=tuple(cfg["dqn"]["hidden_sizes"]))
    x = torch.randn(1, cfg["dqn"]["obs_dim"])
    y = net(x)
    return {"input_dim": cfg["dqn"]["obs_dim"], "output_dim": int(y.shape[-1]),
            "expected_output_dim": cfg["dqn"]["n_actions"], "ok": int(y.shape[-1]) == cfg["dqn"]["n_actions"]}


def run_pytest() -> dict:
    t0 = time.time()
    proc = subprocess.run([python_exe(), "-m", "pytest", "tests/study_b", "-q"],
                           cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
    return {"returncode": proc.returncode, "elapsed_seconds": elapsed, "tail": tail,
            "ok": proc.returncode == 0}


def main() -> int:
    cfg = load_frozen_config()
    report: dict = {}
    print("[preflight] checking imports...")
    report["imports"] = check_imports()
    if not report["imports"]["ok"]:
        print("[preflight] IMPORT ERRORS -- stopping before further checks (likely missing/mismatched deps).")
        write_json_atomic(VERIFICATION / "preflight_report.json", report)
        print(json.dumps(report, indent=2))
        return 1

    print("[preflight] checking environment + observation shape...")
    report["env_observation"] = check_env_and_observation(cfg)

    print("[preflight] checking Q-network shapes...")
    report["q_network"] = check_q_network(cfg)

    print("[preflight] running tests/study_b (this can take several minutes)...")
    report["pytest"] = run_pytest()

    report["overall_ok"] = (
        report["imports"]["ok"]
        and report["env_observation"]["observation_dim_ok"]
        and report["q_network"]["ok"]
        and report["pytest"]["ok"]
    )
    write_json_atomic(VERIFICATION / "preflight_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "pytest"}, indent=2))
    print(f"pytest: returncode={report['pytest']['returncode']} elapsed={report['pytest']['elapsed_seconds']:.1f}s")
    print(f"\nOVERALL: {'PASS' if report['overall_ok'] else 'FAIL'}")
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

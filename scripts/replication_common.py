"""Shared paths and frozen stage table for the independent-seed
replication (protocol/new_protocol.md). Engineering only -- does not
change DQN/reward/observation hyperparameters. Reuses the existing
train_curriculum_stage_highwayenv.py entry point."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _common import BUNDLE_ROOT, CHECKPOINTS_ROOT, OUTPUTS, SCENARIO_BANKS, find_latest_checkpoint

REPL_CFG_PATH = BUNDLE_ROOT / "configs" / "REPLICATION_RUN_CONFIG.json"
MANIFEST_CSV = BUNDLE_ROOT / "new_seed_manifest.csv"
REPL_CKPT_ROOT = CHECKPOINTS_ROOT / "seed_replication_v1"
CURRICULUM_ROOT = REPL_CKPT_ROOT / "curriculum"
WELFARE_ROOT = REPL_CKPT_ROOT / "welfare"
SMOKE_ROOT = REPL_CKPT_ROOT / "smoke_929999"
REPL_RUN_STATE = OUTPUTS / "replication_run_state"
SEEDS = (920101, 920102, 920103, 920104, 920105, 920106)
SMOKE_SEED = 929999
CURRICULUM_END_STEP = 1_200_000


@dataclass(frozen=True)
class Stage:
    name: str
    start_step: int
    end_step: int
    scenario_bank: str
    scenario_ids: tuple[str, ...]
    checkpoint_every: int


def _ids_from_bank(filename: str) -> tuple[str, ...]:
    data = json.loads((SCENARIO_BANKS / filename).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return tuple(s["scenario_id"] for s in data)
    return tuple(data.keys())


def load_stages() -> tuple[Stage, ...]:
    cfg = json.loads(REPL_CFG_PATH.read_text(encoding="utf-8"))
    stages = []
    for raw in cfg["curriculum_stages"]:
        ids = raw["scenario_ids"]
        if ids == "from_bank":
            ids = _ids_from_bank(raw["scenario_bank"])
        else:
            ids = tuple(ids)
        stages.append(Stage(
            name=raw["name"],
            start_step=int(raw["start_step"]),
            end_step=int(raw["end_step"]),
            scenario_bank=raw["scenario_bank"],
            scenario_ids=ids,
            checkpoint_every=int(raw["checkpoint_every"]),
        ))
    return tuple(stages)


def stage_ckpt_dir(seed: int, stage_name: str, *, curriculum: bool = True) -> Path:
    root = CURRICULUM_ROOT if curriculum else SMOKE_ROOT
    return root / str(seed) / stage_name / f"seed_{seed}_{stage_name}"


def stage_output_root(seed: int, stage_name: str, *, curriculum: bool = True) -> Path:
    root = CURRICULUM_ROOT if curriculum else SMOKE_ROOT
    return root / str(seed) / stage_name


def latest_verified_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Newest ckpt_step_*.pt that torch.load()s cleanly. Skips a corrupt
    in-progress write so pause/crash resume never loads a half-written file."""
    if not ckpt_dir.exists():
        return None
    ranked: list[tuple[int, Path]] = []
    for f in ckpt_dir.glob("ckpt_step_*.pt"):
        try:
            ranked.append((int(f.stem.split("_")[-1]), f))
        except ValueError:
            continue
    ranked.sort(reverse=True)
    if not ranked:
        return None
    try:
        import torch
    except ImportError:
        return ranked[0]
    for step, path in ranked:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(ckpt, dict) or "online" not in ckpt:
            continue
        return step, path
    return None


def seed_progress(seed: int) -> dict:
    """Inspect on-disk checkpoints for one replication seed."""
    stages = load_stages()
    progress = {}
    current = None
    for st in stages:
        latest = latest_verified_checkpoint(stage_ckpt_dir(seed, st.name))
        progress[st.name] = None if latest is None else latest[0]
        if latest is None or latest[0] < st.end_step:
            if current is None:
                current = st.name
    done = progress.get("C64_R50") == CURRICULUM_END_STEP
    return {
        "seed": seed,
        "stages": progress,
        "current_stage": None if done else current or stages[0].name,
        "curriculum_complete": done,
        "latest_step": max((s or 0) for s in progress.values()) if any(progress.values()) else 0,
    }


def next_stage_job(seed: int) -> dict | None:
    """Return the next incomplete curriculum stage job, or None if the
    seed's task curriculum is finished at C64_R50 step 1_200_000."""
    stages = load_stages()
    prev_end_ckpt: Path | None = None
    for st in stages:
        ckpt_dir = stage_ckpt_dir(seed, st.name)
        latest = latest_verified_checkpoint(ckpt_dir)
        if latest is not None and latest[0] >= st.end_step:
            prev_end_ckpt = latest[1]
            continue
        if latest is None:
            resume_from = prev_end_ckpt
            start_step = st.start_step
        else:
            resume_from = latest[1]
            start_step = latest[0]
        remaining = st.end_step - start_step
        if remaining <= 0:
            prev_end_ckpt = latest[1] if latest else prev_end_ckpt
            continue
        return {
            "seed": seed,
            "stage": st,
            "resume_from": resume_from,
            "start_step": start_step,
            "max_additional_steps": remaining,
        }
    return None


CONDITIONS = ("mean", "ggi", "maximin")
CONDITION_DIR = {"mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}
WELFARE_START = 1_200_000
WELFARE_END = 2_000_000
WELFARE_STAGE_PREFIX = "Formal"


def task_c64_checkpoint(seed: int) -> Path | None:
    latest = latest_verified_checkpoint(stage_ckpt_dir(seed, "C64_R50"))
    if latest is None or latest[0] < CURRICULUM_END_STEP:
        return None
    return latest[1]


def welfare_ckpt_dir(seed: int, condition: str) -> Path:
    stage = f"{WELFARE_STAGE_PREFIX}_{condition}"
    return WELFARE_ROOT / str(seed) / CONDITION_DIR[condition] / f"seed_{seed}_{stage}"


def welfare_output_root(seed: int, condition: str) -> Path:
    return WELFARE_ROOT / str(seed) / CONDITION_DIR[condition]


def welfare_run_id(seed: int, condition: str) -> str:
    return f"{condition}_{seed}"


def next_welfare_job(seed: int, condition: str) -> dict | None:
    """Next incomplete Mean/GGI/Maximin fine-tune for one seed, or None if
    already at step 2_000_000. All three conditions start from the same
    C64_R50 task checkpoint (new_protocol.md §13)."""
    init = task_c64_checkpoint(seed)
    if init is None:
        return None
    latest = latest_verified_checkpoint(welfare_ckpt_dir(seed, condition))
    if latest is not None and latest[0] >= WELFARE_END:
        return None
    if latest is None:
        resume_from, start_step = init, WELFARE_START
    else:
        resume_from, start_step = latest[1], latest[0]
    remaining = WELFARE_END - start_step
    if remaining <= 0:
        return None
    return {
        "run_id": welfare_run_id(seed, condition),
        "seed": seed,
        "condition": condition,
        "resume_from": resume_from,
        "start_step": start_step,
        "max_additional_steps": remaining,
    }

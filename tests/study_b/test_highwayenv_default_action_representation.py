"""Pre-formal audit finding (Gate Q, 2026-08-16): the training script's
``--action-representation`` CLI flag defaulted to ``direct_accel`` (the
deprecated representation) instead of ``meta_speed`` (the actually
accepted one, per Amendment 4 / M6-R3). No launched run was silently
affected since every invocation this session explicitly passed the
flag, but an omitted flag in a future session would have silently
regressed to the wrong representation. Locks in the corrected default
for both the training script and the evaluation script (which already
defaulted correctly)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_train_curriculum_stage_highwayenv_cli_defaults_to_meta_speed():
    module = _load_script("train_curriculum_stage_highwayenv")
    src = inspect.getsource(module.main)
    assert '"--action-representation", type=str, default="meta_speed"' in src


def test_train_curriculum_stage_highwayenv_helper_defaults_to_meta_speed():
    module = _load_script("train_curriculum_stage_highwayenv")
    assert inspect.signature(module._make_env).parameters["action_representation"].default == "meta_speed"
    assert inspect.signature(module.collect_fixed_oracle_batch_multi).parameters["action_representation"].default == "meta_speed"


def test_evaluate_policy_highwayenv_cli_defaults_to_meta_speed():
    module = _load_script("evaluate_policy_highwayenv")
    src = inspect.getsource(module.main)
    assert '"--action-representation", type=str, default="meta_speed"' in src

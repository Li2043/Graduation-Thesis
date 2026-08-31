"""Stage 6B formal analysis package (100K multi-seed results).

Does not retrain policies. Reconstructs missing episode-level evaluation
records at the preregistered primary endpoint (step 100000) from published
final network weights when Stage 6A only retained evaluation summaries.
"""

from __future__ import annotations

ANALYSIS_PROTOCOL_VERSION = "stage6b_h1_r1_100k"
PRIMARY_ENDPOINT_STEP = 100_000
EVALUATION_STEPS = (0, 10_000, 25_000, 50_000, 75_000, 100_000)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 91_001
CONTRASTS = (
    ("mean_pbrs", "baseline", "mean_pbrs - baseline"),
    ("min_pbrs", "baseline", "min_pbrs - baseline"),
    ("min_pbrs", "mean_pbrs", "min_pbrs - mean_pbrs"),
)
PRIMARY_ENDPOINTS = (
    "evaluation_success_rate",
    "stakeholder_collision_rate",
    "mean_stakeholder_episode_utility",
    "minimum_stakeholder_episode_utility",
    "convention_consistency",
)
STAKEHOLDERS = ("A", "B", "B_front", "B_rear")
EXPECTED_ENV_LOCK = "d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12"
EXPECTED_COMFORT_LOCK = "1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061"
EXPECTED_PBRS_LOCK = "032080d29a34e47fb15c79d46a1575276a7b944eaafad0886193c3a8b6b183f2"
EXPECTED_PROTOCOL_LOCK = "44d5e647cf97bf9c6ce6e863320b669e8e69e1200871155e12929b60682dddc2"
EXPECTED_RUNNER_COMMIT = "a89256db879f04d1e02782ff8dc1af00ff1d75b9"

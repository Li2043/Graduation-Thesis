# STAGE 7C-Q1 DECISION

## 1. Experiment identity

- Stage: `stage7c_q1`
- Purpose: Baseline competence qualification pilot
- Algorithm: Double DQN
- Condition: Baseline
- Base reward: V2 active-time (`0.0005` / step)
- Seeds: `64001`–`64020`
- Max steps: `400000` joint environment steps

## 2. Protocol provenance

- Protocol tag: `stage7c-q1-protocol-v1`
- Tagged/code commit: `c8c75207c06c6a0511cac5fb24b644a61def8d14`
- Config SHA-256: `df64cc71c3c221e22b1abdb714ff6a45850ae32162e1b8ac8672cf23dc20e248`
- Results branch tip (analysis parent): `6f19e5a68e4d62912b74a2199ecf6e18f0c9b2b4` prior to analysis commit

## 3. Integrity verdict

- Integrity status: **VALID**
- Checks: 27; failed: 0
- All critical provenance and completeness checks passed.

## 4. Complete checkpoint table

Extended view (≥200K uses 64 episodes/seed-checkpoint; early standard 16).

| ckpt | ep_success | seed_mean_success | collision | truncation | swap | n_ep |
|------|------------|-------------------|-----------|------------|------|------|
| 0 | 0.165625 | 0.165625 | 0.15625 | 0.678125 | 0.0625 | 320 |
| 25000 | 0.6125 | 0.6125 | 0.065625 | 0.321875 | 0.16875 | 320 |
| 50000 | 0.65625 | 0.65625 | 0.0625 | 0.28125 | 0.2 | 320 |
| 75000 | 0.734375 | 0.734375 | 0.065625 | 0.2 | 0.225 | 320 |
| 100000 | 0.778125 | 0.778125 | 0.103125 | 0.11875 | 0.31875 | 320 |
| 125000 | 0.775 | 0.775 | 0.0375 | 0.1875 | 0.2875 | 320 |
| 150000 | 0.784375 | 0.784375 | 0.075 | 0.140625 | 0.2375 | 320 |
| 175000 | 0.803125 | 0.803125 | 0.071875 | 0.125 | 0.25 | 320 |
| 200000 | 0.778125 | 0.778125 | 0.05625 | 0.165625 | 0.25 | 1280 |
| 225000 | 0.771875 | 0.771875 | 0.034375 | 0.19375 | 0.25625 | 1280 |
| 250000 | 0.790625 | 0.790625 | 0.053125 | 0.15625 | 0.26875 | 1280 |
| 275000 | 0.834375 | 0.834375 | 0.046875 | 0.11875 | 0.21875 | 1280 |
| 300000 | 0.803125 | 0.803125 | 0.025 | 0.171875 | 0.3 | 1280 |
| 325000 | 0.871875 | 0.871875 | 0.05625 | 0.071875 | 0.30625 | 1280 |
| 350000 | 0.8375 | 0.8375 | 0.04375 | 0.11875 | 0.35625 | 1280 |
| 375000 | 0.84375 | 0.84375 | 0.040625 | 0.115625 | 0.30625 | 1280 |
| 400000 | 0.7625 | 0.7625 | 0.09375 | 0.14375 | 0.31875 | 1280 |

Episode-pooled and seed-equal means coincide when all seeds share equal episode counts.

## 5. Learning-curve assessment

- Net change 200K→400K: `-0.015625`
- Spearman ρ(checkpoint, success): `0.3166666667` (p=`0.4063970145`; descriptive only)
- Max adjacent success drop (200K–400K): `0.08125` at `(375000, 400000, 0.08125000000000004)`
- All adjacent drops ≤ 0.03: `False`
- 350–400K platform range: `0.08125`

## 6. Seed-level stability

- Stable qualified seed intersection (|S|≥61/64 at 350∩375∩400): `1` / 20
- Seeds: `[64002]`
- Material-regression seeds (350–400, drop>0.20): `[64001, 64003, 64004, 64008, 64009, 64013, 64019]` (n=7)
- Late-collapse seeds: `[64009]` (n=1)

## 7. Safety assessment

- unilateral_stall_trend: 200K=0 → 400K=0; already near-absent in late extended evaluations (no material late decline signal).
- mutual_yielding_trend: 200K=0.03125 → 400K=0.009375; declines
- collision_late_rise: 350K=0.04375, 375K=0.040625, 400K=0.09375; yes, collision rises at 400K relative to 350/375
- low_truncation_high_collision: 350K truncation=0.11875, collision=0.04375; 400K truncation=0.14375, collision=0.09375; no clean truncation-down/collision-up conversion from 350K to 400K; both remain above gate.
- downstream_failure_note: Dominant non-success truncation-related category is downstream_failure (200K=0.134375, 400K=0.134375).
- road_role_collision_asymmetry: passing-order at 400K is near-balanced (mainline_first=0.495902, ramp_first=0.504098); mean_exit_mainline=58.2316, mean_exit_ramp=55.2587.
- passing_order_bias_400K: mainline_first=0.495902, ramp_first=0.504098; not highly biased to a single direction.
- controller_role_swap_changes_outcomes: swap_disagreement_rate=0.2625 over n_pairs=640; assignment-stratified rates in role_swap_analysis.csv.

## 8. Failure-mode assessment

Collision and truncation are reported separately; they are not merged.
See `failure_taxonomy.csv` and figure `07_failure_taxonomy.png`.

## 9. Role and passing-order audit

See `role_swap_analysis.csv`. This pilot is not a PBRS confirmatory test.
Any base-reward-induced role bias is interpretive risk only unless a frozen hard threshold exists (none beyond swap eligibility).

## 10. Historical Stage 7B comparison

- Historical Stage 7B Double DQN @300K: success=`0.75625`, collision=`0.08125`, truncation=`0.1625`
- Stage 7C-Q1 @300K (extended): success=`0.803125`, collision=`0.025`, truncation=`0.171875`
- Disclaimer: Historical, non-paired comparison. Different master seeds and evaluation protocol. Not a causal estimate of the active-time reward effect.
- No causal claim that V2 is significantly better than V1.

## 11. Gate table

| checkpoint | mean_success | collision | truncation | swap | values |
|------------|--------------|-----------|------------|------|--------|
| 350000 | False | False | False | False | s=0.8375, c=0.04375, t=0.11875, swap=0.35625 |
| 375000 | False | False | False | False | s=0.84375, c=0.040625, t=0.115625, swap=0.30625 |
| 400000 | False | False | False | False | s=0.7625, c=0.09375, t=0.14375, swap=0.31875 |

- intersection_ok: `False` count=`1`
- learning_curve_ok: `False` violations=`['275000->300000 drop=0.0312', '325000->350000 drop=0.0344', '375000->400000 drop=0.0813']`
- material_regression_seeds: `[64001, 64003, 64004, 64008, 64009, 64013, 64019]`
- late_collapse_seeds: `[64009]`

## 12. Final status

# **FAIL**

## 13. Permitted next action

Do not start the final three-condition experiment. Stop further algorithm and reward modifications. Report competence-limited conclusion.


# V1 Decision Log

> Append-only record of V1 design decisions. Every notable choice (scope,
> parameters, seeds, metrics, freezes) must be logged here before it takes
> effect in a final run. Do not delete rows; supersede them with a new dated
> row and set the old row `status` to `superseded`.

| date | decision | reason | alternatives considered | approved by | status |
| --- | --- | --- | --- | --- | --- |
| 2026-06-25 | Initialise V1 documentation framework and experiment registry | Establish a controlled, reproducible, auditable V1 baseline before any V1 code or runs | Continue ad hoc as in V0; defer documentation | _TBD_ | active |
| 2026-06-25 | Replace diagnostic raw-min Rawlsian reward (`objective_scale * min_i E_i`) with proposal-aligned delta-min shaping: `R_rawls = base_individual_reward + r_rawls_t`, where `r_rawls_t ∈ {+λ_R, 0, −λ_R}` from `ΔE_min` vs `ε_R` | Raw-min level reward was an early diagnostic device; the revised proposal defines Rawlsian shaping as a change-based signal on the least-advantaged agent. Keeps the controlled comparison: the only difference from egoistic is the added delta-min signal | (a) keep raw-min scaling; (b) ignore deprecated flag entirely | _TBD_ | active |
| 2026-06-25 | Shared merge-task adjustment and shared terminal collision penalty remain applied identically to both conditions; `λ_R`/`ε_R` exposed via CLI and logged; `--rawlsian-objective-scale` kept as a deprecated alias mapped to `λ_R` | Preserve single-factor comparison and reproducibility/auditability; avoid breaking existing commands | Drop the deprecated flag (would break old commands) | _TBD_ | active |

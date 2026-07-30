# Analysis-ready data dictionary

- locks/: H1-R1 authoritative protocol and PBRS locks with hashes
- jobs/<job_id>/: per-run resolved config, status, manifests, episode/eval traces, final network weights
- aggregates/run_status.csv: one row per formal job terminal status
- aggregates/evaluation_episodes.csv: evaluation traces across jobs
- aggregates/episode_summaries.csv: training episode summaries
- aggregates/checkpoint_manifest.csv: SHA-256 for local checkpoints and published final weights
- reports/stage6a_summary.json / stage6a_report.md: Stage 6A execution summary
- formal_publish_manifest.json: SHA-256 inventory of published files

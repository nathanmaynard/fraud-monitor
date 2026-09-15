## What

## Why

## Evidence
<!-- metrics before/after, report paths, screenshots -->

## Review checklist
- [ ] Time-ordered split preserved — nothing from months ≥ 5 leaks into fitting
- [ ] Metrics reported at fixed FPR, not accuracy
- [ ] Fairness ratio recomputed if features or threshold changed
- [ ] Model card updated if the deployed model or its inputs changed
- [ ] Tests cover the new code path; `uv run pytest` and `uv run ruff check .` pass

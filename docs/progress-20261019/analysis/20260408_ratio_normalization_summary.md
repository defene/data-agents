## Experiment 4: ratio answer normalization

- Commit: `9fb1d71`
- Previous retained baseline: `028127c` (`finish validator hardening`)
- Candidate runs:
  - `20260408T044001Z`
  - `20260408T044601Z`
  - `20260408T045149Z`

### Aggregate result

- Candidate scores: `21/50`, `25/50`, `21/50`
- Mean candidate score: `22.33/50`
- Previous retained baseline mean: `22.00/50`
- Mean delta vs retained baseline: `+0.33`
- Mean avg steps: `6.55`
- Answered tasks: `50/50` in all three runs

### Hypothesis check

This experiment only targeted ratio/percentage final-answer formatting:

- normalize `0.375 (3 posts vs 8 votes)` -> `0.375`
- normalize `100.0%` -> `100.0`
- reject explanatory text for single-cell ratio answers

The direct targeted signal is positive:

- `task_243` moved from `0/3` correct in the previous retained baseline to `2/3` correct here

### Stable fixes vs original v2 baseline

- `task_22` fixed in `3/3`
- `task_214` fixed in `3/3`
- `task_259` fixed in `3/3`
- `task_420` fixed in `3/3`

### Stable regressions vs original v2 baseline

- `task_38` regressed in `3/3`
- `task_249` regressed in `3/3`

### Interpretation

The overall mean improved slightly over the retained baseline, and the targeted ratio-formatting
behavior improved in the expected direction.

`task_249` should not be causally affected by this experiment because the code path only runs for
ratio/percentage questions. Its regression is therefore more likely model variance or an unrelated
upstream reasoning issue than a direct effect of the patch.

### Decision

Keep `9fb1d71` as the new retained baseline and continue with the next experiment from this commit.

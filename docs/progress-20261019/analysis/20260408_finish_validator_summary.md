## Experiment 2: finish validator hardening

- Commit: `028127c`
- Base run: `20260403T055832Z`
- Candidate runs:
  - `20260408T034147Z`
  - `20260408T034803Z`
  - `20260408T035508Z`

### Aggregate result

- Base score: `21/50`
- Candidate scores: `22/50`, `23/50`, `21/50`
- Mean candidate score: `22/50`
- Mean delta vs base: `+1`
- Mean avg steps: `6.73` vs base `6.16`
- Answered tasks: `50/50` in all three candidate runs

### Hypothesis check

This experiment aimed to reduce premature answer submission for:

- empty answer tables
- NaN-like scalar answers
- percentage answers containing `%`

The observed gains are consistent with the hypothesis:

- `task_67` fixed in `2/3`
- `task_283` fixed in `2/3`
- `task_420` fixed in `3/3`
- `task_11` fixed in `1/3`

### Stable fixes

- `task_22` fixed in `3/3`
- `task_259` fixed in `3/3`
- `task_330` fixed in `3/3`
- `task_420` fixed in `3/3`

### Stable regressions

- `task_38` regressed in `3/3`
- `task_243` regressed in `3/3`
- `task_349` regressed in `3/3`

### Interpretation

The finish validator appears to be a real improvement and should be kept as the active baseline.
Most regressions are not direct evidence against the validator idea itself; they point to a missing
answer-contract layer:

- `task_243` regresses because the answer becomes explanatory text instead of a plain numeric scalar.
- `task_349` regresses because the answer becomes a more verbose surface form than the gold label.
- `task_38` still reaches submission with a poor contract or malformed row extraction output.

### Decision

Keep `028127c` as the new experiment baseline and make the next experiment focus on answer-contract
control and final-output normalization.

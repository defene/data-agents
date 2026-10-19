## Experiment 3: answer contract control

- Commit: `09e74df`
- Base run: `20260403T055832Z`
- Candidate runs:
  - `20260408T041216Z`
  - `20260408T041901Z`
  - `20260408T042506Z`

### Aggregate result

- Base score: `21/50`
- Candidate scores: `19/50`, `21/50`, `22/50`
- Mean candidate score: `20.67/50`
- Mean delta vs base: `-0.33`
- Mean avg steps: `6.35` vs base `6.16`
- Answered tasks: `49/50`, `50/50`, `50/50`

One run hit an external OpenAI 400 request-body parse failure on `task_283`. This should not be
treated as algorithmic signal.

### Hypothesis check

The hypothesis was that a lightweight answer-contract layer would:

- normalize explanatory numeric cells into raw numeric values
- block scratch-file-path answers for table questions
- reduce malformed final submissions without hurting stable tasks

The hypothesis did not hold up at the benchmark level.

### Stable fixes

- `task_22` fixed in `3/3`
- `task_214` fixed in `3/3`
- `task_259` fixed in `3/3`
- `task_420` fixed in `3/3`

### Stable regressions

- `task_38` regressed in `3/3`
- `task_218` regressed in `3/3`
- `task_249` regressed in `3/3`

### Root cause

The answer-contract inference was too coarse:

- It treated many numeric questions as requiring a single scalar cell.
- This broke legitimate multi-metric outputs such as `task_249`, which requires two aggregate values.
- It also over-normalized some scalar outputs, e.g. phone-number formatting in `task_218`.

### Additional observations

- `task_243` improved from a stable regression to only `1/3` regression, which suggests the raw-value
  guard has some value.
- `task_349` remained problematic, so the contract layer alone did not solve verbose label answers.
- `task_38` remained unstable, so the path/placeholder-row guard was not enough to recover the row-extraction workflow.

### Decision

Do not keep `09e74df` as the active baseline. Revert it and keep `028127c` (`finish validator hardening`)
as the current stable line.

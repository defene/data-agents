# Agent Experiment Protocol

This repository should evaluate agent ideas with single-variable benchmark experiments.

## Core Rules

1. Read the relevant code and representative failure traces before proposing a change.
2. Keep one hypothesis per experiment branch/commit whenever possible.
3. Save every meaningful state in git before starting the next hypothesis.
4. Run the full benchmark after each change.
5. Compare every new run against a fixed reference run and record both gains and regressions.
6. Prefer extensible changes over narrow hacks that only fix one task family.

## Required Inputs Before Any Change

For each experiment, write down:

- Hypothesis: what should improve and why
- Code read: which files and functions were reviewed
- Evidence: which tasks or traces motivated the change
- Scope: what files will change
- Risks: what behavior may regress

If the answer to any of the above is unclear, do more reading before editing.

## Versioning Discipline

Use one saved version per milestone:

- Baseline snapshot commit
- One commit for each hypothesis implementation
- One benchmark run per hypothesis commit
- One comparison report per benchmark run

Suggested naming:

- Branch: keep working on `codex/agent-v2-overhaul`
- Commit: `exp: <slug>`
- Run id: `exp_<date>_<slug>`
- Comparison report: `artifacts/analysis/<date>_<slug>.json`
- Notes: `artifacts/analysis/<date>_<slug>.md`

## Experiment Loop

1. Read code
2. Read traces for representative wins/failures
3. Write a short hypothesis note
4. Implement exactly one main idea
5. Save commit
6. Run full benchmark
7. Compare against reference run
8. Decide: keep, revise, or revert

## Decision Rules

Keep an experiment only if it satisfies one of:

- Improves overall score with acceptable regressions
- Holds overall score but clearly improves a strategic task family with lower step cost
- Improves robustness or answer rate without harming accuracy materially

Revert or branch away if:

- Overall score drops and gains are not concentrated in an important target family
- The change only fixes a handful of tasks through brittle formatting tricks
- The implementation makes future route/controller extension harder

## Minimum Analysis to Record

For every run comparison, record:

- Overall score delta
- Per-difficulty delta
- Fixed task ids
- Regressed task ids
- Average step delta
- Whether the hypothesis behaved as expected
- Final decision: keep / revise / discard

## Reference Commands

Run benchmark:

```powershell
uv run dabench run-benchmark --config configs\react_baseline.local.yaml
```

Compare runs:

```powershell
.\.venv\Scripts\python.exe scripts\compare_runs.py `
  --base-run-dir artifacts\runs\<base_run_id> `
  --candidate-run-dir artifacts\runs\<candidate_run_id> `
  --json-out artifacts\analysis\<date>_<slug>.json
```

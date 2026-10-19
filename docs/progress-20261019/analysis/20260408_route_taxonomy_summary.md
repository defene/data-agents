# Route Taxonomy Experiment Summary

## Metadata

- Experiment commit: `0a6717e`
- Base run: `artifacts/runs/20260403T055832Z`
- Candidate runs:
  - `artifacts/runs/20260408T025637Z`
  - `artifacts/runs/20260408T030304Z`
  - `artifacts/runs/20260408T030853Z`

## Hypothesis

Refining route taxonomy should improve task framing by:

- separating `ratio_percentage`, `average`, `count`, and `multi_row_extract`
- reducing the pathological `doc_link` default caused by `knowledge.md`
- steering percentage and aggregate tasks toward `sql_first` or `tabular_first`

## Result

- Base score: `21/50`
- Candidate scores: `16/50`, `23/50`, `18/50`
- Mean candidate score: `19/50`
- Mean delta vs base: `-2.0`

Per-run deltas:

- Run 1: `-5`
- Run 2: `+2`
- Run 3: `-3`

## Stable Improvements

Tasks fixed in multiple runs:

- `task_259` fixed in 3/3 runs
- `task_67` fixed in 2/3 runs
- `task_214` fixed in 2/3 runs

One-run improvements:

- `task_22`
- `task_26`
- `task_283`
- `task_379`
- `task_408`

## Stable Regressions

Tasks regressed in all 3 runs:

- `task_38`
- `task_249`
- `task_350`

Tasks regressed in 2/3 runs:

- `task_196`
- `task_303`
- `task_349`

## Operational Notes

- Run 2 produced only `49/50` answers because `task_27` hit a model-side 400 request parse error.
- Average steps stayed close to base, so the main effect was answer quality, not runtime behavior.

## Decision

Discard as the next default baseline.

The route taxonomy change improved several percentage and counting tasks, but the gains were not robust enough across three full benchmark runs. The regressions on previously stable tasks were too frequent to justify keeping this as the active base.

## Follow-up

Keep the experiment commit for reference, but continue the next hypothesis from the prior stable baseline instead of stacking on top of this change.

## Full Task Sweep

- Analysis source: `artifacts/analysis/20260408_full_task_sweep.json`
- Retained baseline runs analyzed:
  - `20260408T044001Z`
  - `20260408T044601Z`
  - `20260408T045149Z`

## Overall

- Total tasks: `50`
- Stable correct across all 3 runs: `15`
- Stable fail across all 3 runs: `21`
- Swing tasks: `14`

This means the current framework is still dominated by structural instability and recurring failure
families rather than isolated one-off misses.

## Where The Framework Is Strong

Stable wins cluster in:

- direct doc-link lookups after a clean structured join
- simple count / average tasks when the expression is obvious
- some JSON+DB+doc single-row outputs
- a subset of row-extraction tasks where projection is straightforward

Representative stable-correct tasks:

- `task_214`, `task_420`, `task_257`, `task_292`, `task_415`

## Main Failure Families

### 1. Row Extraction And Projection Failures

This is the biggest stable-failure bucket.

- Stable fail: `7/13`
- Swing: `1/13`

Representative tasks:

- `task_11`: IDs are found, but projected columns (`SEX`, `Diagnosis`) are blank.
- `task_19`: answer is returned as `full_name` instead of separate `first_name`, `last_name`.
- `task_38`: agent writes the result CSV to scratch and answers with the file path.
- `task_173`: returns a narrative string instead of the actual country rows.
- `task_180`: finds a large matching set but outputs the wrong projection / filtering result.
- `task_379`: returns a tally table (`element`, `count`) while gold expects the element list only.

Interpretation:

- The framework often knows the target slice, but fails to preserve the exact projection the task wants.
- Multi-row outputs are still too loosely controlled at finalization time.

### 2. Ratio / Percentage Tasks

- Stable fail: `3/7`
- Swing: `3/7`

Representative tasks:

- `task_283`: resolves to `0.0`, suggesting a join or label-resolution failure.
- `task_396`: wrong numerator / denominator selection.
- `task_408`: numerically close, but output precision / formatting does not match gold.
- `task_243`, `task_303`, `task_352` are swing tasks, which means this family is improving but still unstable.

Interpretation:

- This family is no longer purely a formatting problem.
- The remaining errors split into:
  - wrong entity resolution / counting
  - wrong final expression
  - insufficient precision preservation

### 3. Multi-Metric Aggregate Output Shape

- Stable fail: `3/6`
- Stable correct: `2/6`
- Swing: `1/6`

Representative tasks:

- `task_27`: `full_name` is merged and numeric value leaks as `np.float64(...)`.
- `task_163`: returns per-category totals, while gold expects the approved expense type and total in a different grouping.
- `task_355`: returns `full_name, cost` instead of `first_name, last_name, cost`.

Interpretation:

- These are not raw reasoning failures only.
- A large part of the loss is in output contract fidelity: split vs merged names, grouped vs final aggregate, source formatting leakage.

### 4. Average / Aggregate Semantics

- Stable fail: `3/6`

Representative tasks:

- `task_169`: monthly average is computed with the wrong denominator or wrong aggregation grain.
- `task_199`: finds many valid rows but applies a too-broad school filter.
- `task_249`: computes average upvotes over posts rather than over users.

Interpretation:

- The framework still confuses row-level aggregation with entity-level aggregation.
- This is a solver semantics issue, not just formatting.

### 5. “Knowledge Missing” False Negative Tasks

Representative tasks:

- `task_344`
- `task_418`

Interpretation:

- The agent sometimes decides a task is under-specified even though the benchmark expects it to derive thresholds or criteria from the available docs.
- This points to weak doc-range extraction and weak confidence calibration.

## Modality Pattern

The hardest modality buckets are:

- `csv+json+doc`
- `csv+db+doc`
- `json+doc`

These are exactly the mixed-source settings where the agent must:

- resolve an entity in structured data
- use docs to interpret a label / threshold / linked field
- return a tightly constrained final output

Purely structured tasks are not the main bottleneck now. Mixed-source tasks are.

## Practical Reading Of The Benchmark

The benchmark is not asking for one universal improvement next. It is exposing three different agent weaknesses:

1. final table contract fidelity
2. aggregate grain control
3. mixed-source entity / label resolution

## Best Next Targets

If we optimize by leverage, the next experiments should target these in order:

1. Multi-row table finalization for row-extraction tasks
   - Especially `task_38`, `task_19`, `task_379`, `task_173`
2. Aggregate grain control
   - Especially `task_169`, `task_199`, `task_249`
3. Mixed-source label / threshold resolution
   - Especially `task_75`, `task_89`, `task_344`, `task_418`

## Recommendation

Do not continue with broad prompt-only changes.

The next worthwhile iteration should be a narrow controller-side experiment on multi-row table answers:

- when the gold-style answer is clearly row extraction
- and the agent has already materialized rows in scratch or in a query result
- the framework should prefer returning the projected values directly instead of paths, merged labels, tally tables, or partially populated rows

That is the cleanest next leverage point exposed by the full-task sweep.

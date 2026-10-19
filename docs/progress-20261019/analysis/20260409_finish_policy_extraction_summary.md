# 2026-04-09 Finish Policy Extraction Summary

- Commit: `9341a7b`
- Reference retained baseline mean: `22.33/50` from runs `20260408T044001Z`, `20260408T044601Z`, `20260408T045149Z`
- Candidate runs: `20260409T015546Z`, `20260409T020409Z`, `20260409T021106Z`
- Candidate scores: `23`, `22`, `22`
- Candidate mean: `22.33/50`
- Candidate answered counts: `50`, `49`, `49`

## Stable changes vs fixed reference run

- Stable fixed: ['task_243', 'task_250', 'task_261', 'task_305']
- Stable regressed: ['task_349', 'task_352', 'task_86']

## Interpretation

- This experiment was designed as a behavior-preserving structural refactor.
- The candidate mean matches the retained baseline mean (`22.33/50`), so the extraction appears score-neutral overall.
- Two runs answered `49/50` because of external OpenAI request-limit failures rather than explicit local guard rejections.
- Because score held while policy boundaries improved, this experiment is a candidate to keep as the new structural baseline.

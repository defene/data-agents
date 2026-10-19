# 2026-04-09 Answer Policy Extraction Summary

- Commit: `e10a7b2`
- Previous retained baseline mean: `22.33/50` from runs `20260409T015546Z`, `20260409T020409Z`, `20260409T021106Z`
- Candidate runs: `20260409T023900Z`, `20260409T024653Z`, `20260409T025453Z`
- Candidate scores: `23`, `25`, `23`
- Candidate mean: `23.67/50`
- Candidate answered counts: `50`, `49`, `50`

## Stable changes vs fixed reference run

- Stable fixed: ['task_243', 'task_250', 'task_283']
- Stable regressed: ['task_26']

## Interpretation

- This experiment was intended to be mostly behavior-preserving while moving family-specific answer logic out of controller.py.
- The result is better than expected: mean score increased from `22.33/50` to `23.67/50`.
- The extracted boundary appears to reduce some cross-family interference, especially on tasks where answer normalization and validation were previously coupled to controller logic.
- One run answered `49/50` because of an external OpenAI 400 request-body failure on `task_344`, not because of a local answer-policy rejection.
- Decision: keep this commit as the new retained baseline.

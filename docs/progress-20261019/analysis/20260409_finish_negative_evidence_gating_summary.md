# 2026-04-09 Finish Negative Evidence Gating Summary

- Commit: `ba10601`
- Previous retained baseline mean: `23.67/50` from runs `20260409T023900Z`, `20260409T024653Z`, `20260409T025453Z`
- Candidate runs: `20260409T035017Z`, `20260409T035653Z`, `20260409T040429Z`
- Candidate scores: `23`, `22`, `24`
- Candidate mean: `23.00/50`
- Candidate answered counts: `50`, `50`, `49`

## Stable changes vs fixed reference run

- Stable fixed: ['task_24', 'task_243', 'task_283', 'task_305']
- Stable regressed: ['task_86']

## Interpretation

- The change successfully prevented some premature ready states on negative evidence, but it did not translate into enough benchmark gain.
- Mean score moved from `23.67/50` down to `23.00/50`.
- The target tasks `task_22`, `task_38`, and `task_80` were not recovered reliably enough to offset regressions elsewhere.
- Decision: discard and revert.

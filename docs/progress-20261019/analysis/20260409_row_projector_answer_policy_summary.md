# 2026-04-09 Row Projector Answer Policy Summary

- Commit: `1afa3bd`
- Previous retained baseline mean: `23.67/50` from runs `20260409T023900Z`, `20260409T024653Z`, `20260409T025453Z`
- Candidate runs: `20260409T031542Z`, `20260409T032252Z`, `20260409T033253Z`
- Candidate scores: `22`, `20`, `26`
- Candidate mean: `22.67/50`
- Candidate answered counts: `50`, `48`, `49`

## Stable changes vs fixed reference run

- Stable fixed: ['task_24', 'task_243']
- Stable regressed: ['task_26', 'task_349', 'task_86']

## Interpretation

- The hypothesis was conceptually sound: local checks showed the row projector can turn a full_name column into a scoreable multi-column answer, and proxy answers are correctly rejected.
- In full benchmark runs, the behavior was too noisy. `task_19` was fixed in one run, but the overall mean fell from `23.67/50` to `22.67/50`.
- Stable regressions against the fixed reference include `task_26`, `task_86`, and `task_349`.
- Because the mean dropped relative to the retained baseline, this experiment should be discarded and reverted despite the positive targeted signal.

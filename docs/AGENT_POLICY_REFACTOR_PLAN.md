# Agent Policy Refactor Plan

## Goal

Refactor the current agent strategy layer so that heuristics act only as cheap priors and hard safety guards, not as the main semantic engine.

This plan is designed for the current `agent-v2` line and follows the repository experiment protocol:

1. Read code and traces first.
2. Change one main idea at a time.
3. Save every meaningful state in git.
4. Run the full benchmark three times on a fixed `gpt-5-mini` configuration.
5. Keep only changes that improve score, robustness, or extensibility.

## Why This Refactor Is Needed

The least extensible part of the current system is the cross-file strategy layer, not the tool layer.

Current duplication points:

- `src/data_agent_baseline/agents/router.py`
  - `_infer_task_type`
  - `_infer_solver_mode`
  - `_infer_output_hint`
- `src/data_agent_baseline/agents/controller.py`
  - `_is_row_extraction_question`
  - `_is_ratio_or_percentage_question`
  - `_is_average_question`
  - `update_finish_memory`
  - `normalize_answer_submission`
  - `validate_answer_submission`
- `src/data_agent_baseline/agents/react.py`
  - `_LOW_VALUE_EXPLORATION_ACTIONS`
  - `_should_block_low_value_exploration`
  - inline answer-guard orchestration
- `src/data_agent_baseline/agents/prompt.py`
  - route and finish behavior restated in prompt instructions

Today, the same policy knowledge is encoded in four places:

- route inference
- finish readiness
- answer normalization / validation
- prompt behavior instructions

That makes extension expensive and fragile. Adding a new task family often means editing the same concept in multiple files and hoping the pieces stay aligned.

## Role of Heuristics

### Heuristics Should Stay In The System

Heuristics are still useful when they are:

- cheap
- deterministic
- auditable
- low-risk when wrong

Approved heuristic roles:

1. Source bootstrap
   - File suffixes, directory patterns, and source ordering are acceptable for building a source manifest.
2. Route prior
   - Heuristics may propose candidate routes, but should not decide the final solver path alone.
3. Hard guardrails
   - Empty answer tables, NaN-like outputs, path-like answers, and obvious formatting violations should still be blocked by deterministic rules.
4. Repair triggers
   - If an observed result contradicts the initial route prior, heuristics can trigger reroute or repair.

### Heuristics Should Not Own Core Semantics

Heuristics should not:

- decide task semantics from keywords alone
- determine finish readiness from question text patterns
- infer final answer shape from the last answer payload
- be duplicated across prompt, controller, and run loop

Rule of thumb:

- `heuristic = prior`
- `evidence = truth`
- `validator = gate`

## Research Basis

This refactor is grounded in the task definition and related literature, not only repo-local intuition.

### Official Competition Setting

- DataAgent official site: [https://dataagent.top/](https://dataagent.top/)
  - Tasks require multi-step reasoning over heterogeneous data packages.
  - Reasoning topology is explicitly non-linear: branching, loops, and merge.
  - Evaluation is column-as-vector matching, so final projection matters as much as intermediate reasoning.
  - Phase 2 introduces additional modalities, so the strategy layer must stay modality-extensible.

### Papers That Inform The Refactor

1. DataMind: Scaling Generalist Data-Analytic Agents
   - OpenReview: [https://openreview.net/forum?id=5PxFqpIYWC](https://openreview.net/forum?id=5PxFqpIYWC)
   - Why it matters:
     - uses fine-grained task taxonomy
     - stresses stable code-based multi-turn rollout
     - supports replacing coarse route labels with richer task-family priors

2. TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning
   - ACL Anthology: [https://aclanthology.org/2025.emnlp-main.710/](https://aclanthology.org/2025.emnlp-main.710/)
   - Why it matters:
     - breaks heterogeneous reasoning into query decomposition, retrieval, SQL, and intermediate answer composition
     - supports separating solver policy from final answer projection

3. HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data
   - ACL Anthology: [https://aclanthology.org/2020.findings-emnlp.91/](https://aclanthology.org/2020.findings-emnlp.91/)
   - Why it matters:
     - shows that heterogeneous QA requires both structured and textual evidence
     - supports keeping route priors weak until evidence confirms the actual join path

4. ARC: Active and Reflection-driven Context Management for Long-Horizon Information Seeking Agents
   - OpenReview: [https://openreview.net/forum?id=j2PfD05yHt](https://openreview.net/forum?id=j2PfD05yHt)
   - Why it matters:
     - supports separating context management from action generation
     - motivates explicit policy state instead of prompt-only control

5. HELM: Steering Long-Horizon Agents with Learned Hierarchical Memory and Epistemic Governance
   - OpenReview: [https://openreview.net/forum?id=W1vKCYeAM1](https://openreview.net/forum?id=W1vKCYeAM1)
   - Why it matters:
     - supports explicit memory slots and evidence governance
     - motivates provenance-aware validators rather than string-pattern finish checks

6. Memp: Exploring Agent Procedural Memory
   - OpenReview: [https://openreview.net/forum?id=aaij11qBCl](https://openreview.net/forum?id=aaij11qBCl)
   - Why it matters:
     - supports storing reusable solver recipes as procedural memory
     - motivates route-specific policies instead of one global behavior prompt

7. TALE: Tool-using Agent Learning from Error
   - OpenReview: [https://openreview.net/forum?id=4V0cZkWT1A](https://openreview.net/forum?id=4V0cZkWT1A)
   - Why it matters:
     - treats error recovery as a first-class capability
     - motivates repair controllers instead of one-pass answer guards

## Target Architecture

### Policy Registry

Introduce a policy registry layer between route inference and the main ReAct loop.

Proposed modules:

- `src/data_agent_baseline/agents/policies/base.py`
- `src/data_agent_baseline/agents/policies/registry.py`
- `src/data_agent_baseline/agents/policies/route.py`
- `src/data_agent_baseline/agents/policies/finish.py`
- `src/data_agent_baseline/agents/policies/answer.py`
- `src/data_agent_baseline/agents/policies/prompting.py`

### Core Interfaces

#### `RoutePolicy`

Input:

- question
- source manifest
- lightweight task context

Output:

- candidate route ids
- route confidence
- expected evidence types
- required state slots

Notes:

- keyword and suffix heuristics are allowed here
- the output is a prior, not a final truth label

#### `FinishPolicy`

Input:

- runtime state
- evidence slots
- current route id
- remaining steps

Output:

- ready to answer
- missing requirements
- repair hints
- optional reroute signal

Notes:

- should operate on state and evidence, not keyword guesses
- one policy per task family is preferred

#### `AnswerPolicy`

Input:

- task
- runtime state
- submitted answer payload

Output:

- normalized answer payload
- validation error or success
- projection hints

Notes:

- universal guards stay deterministic
- answer-family-specific normalization is pluggable

#### `PromptPolicy`

Input:

- selected route
- current finish state
- current evidence state

Output:

- short route-specific behavioral instructions

Notes:

- prompt becomes a renderer of selected policy, not the source of truth

## Minimal Data Model Extension

The current `StructuredMemory` in `src/data_agent_baseline/agents/runtime.py` is enough for a first refactor if we add only a few narrow fields later.

First-pass policy state can stay lightweight:

- `route.task_type`
- `route.solver_mode`
- `route.output_hint`
- `finish.ready_to_answer`
- `finish.missing_requirements`
- `evidence`
- `answer_shape`
- `output_mapping`

Second-pass extensions, only if experiments justify them:

- `route_candidates`
- `route_confidence`
- `answer_family`
- `repair_state`
- `evidence_kinds`

## What Moves Out Of Each Current File

### `controller.py`

Keep:

- universal hard guards
- answer payload sanitation helpers

Move out:

- family-specific finish logic
- family-specific normalization logic
- family-specific semantic validation rules

### `router.py`

Keep:

- source manifest construction
- cheap source-type priors

Move out:

- final task-family semantics
- output-shape semantics
- route rationale text

### `react.py`

Keep:

- loop execution
- tool invocation
- state recording

Move out:

- hard-coded low-value action semantics
- inline finish strategy
- inline answer policy orchestration

### `prompt.py`

Keep:

- shared response contract
- task and observation rendering

Move out:

- policy logic duplicated from route/controller

## First Refactor Principle

Do not start by changing behavior.

The first refactor should extract one policy boundary while keeping semantics close to current behavior, so that later benchmark deltas can be attributed to actual policy ideas rather than to the structural extraction itself.

## Recommended Experiment Sequence

### Experiment A: Extract Finish Policy Boundary

Goal:

- move finish logic behind a policy interface
- preserve current behavior as much as possible

Change:

- introduce `FinishPolicy`
- replace direct `update_finish_memory(...)` logic with a default policy implementation

Success condition:

- near-parity benchmark score
- cleaner future extension path

### Experiment B: Separate Universal Guards From Family Policies

Goal:

- keep universal guardrails in one place
- move ratio / row-extraction specific rules into dedicated answer policies

Change:

- split `normalize_answer_submission(...)` and `validate_answer_submission(...)`

Success condition:

- no regression on stable wins
- fewer cross-family side effects when fixing a single family

### Experiment C: Downgrade Router To Prior-Only

Goal:

- let route inference propose, not decide

Change:

- router returns route candidates or route features
- finish and answer policies rely on evidence state, not route keywords alone

Success condition:

- fewer route-induced regressions
- better extensibility for new modalities or task families

### Experiment D: Route-Specific Prompt Renderer

Goal:

- eliminate repeated strategy text from the global prompt

Change:

- prompt assembles short instructions from selected policy

Success condition:

- prompt edits become local to a policy family

## Immediate Recommendation

The next experiment should be:

- `exp: finish policy extraction`

Why:

- `controller.py` is the current strategy choke point
- it has the highest concentration of duplicated semantic rules
- it is the smallest slice that meaningfully improves extensibility without forcing a full architecture rewrite

## Non-Goals For The Next Iteration

Do not do these yet:

- multi-agent manager/specialist orchestration
- learned router replacement
- large memory schema expansion
- new tool interfaces
- benchmark-specific output hacks

Those can come later if the policy boundary refactor proves stable.

## Decision Standard

The policy refactor is worth continuing only if:

- benchmark score stays competitive
- family-specific fixes stop causing unrelated regressions
- new policy logic can be added in one place instead of four

If extraction adds complexity without reducing coupled edits, stop and simplify.

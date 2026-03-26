# data-agents

English | [中文](README.zh.md)

This repository is my competition project for **KDD Cup 2026 DataAgent-Bench**.  
It is based on the official starter kit and adapted for my own experiments, runs, and iterations.

## What This Repository Is

- A personal competition codebase built on top of the official baseline
- A local runner for task inspection, single-task execution, and benchmark runs
- A place to organize prompts, runtime logic, configs, and experiment outputs

## Project Layout

```text
.
├── configs/
├── src/data_agent_baseline/
├── artifacts/
├── README.md
├── README.zh.md
└── pyproject.toml
```

Main code areas:

- `src/data_agent_baseline/agents/`: model client, prompts, and ReAct loop
- `src/data_agent_baseline/tools/`: filesystem, SQLite, Python execution, and answer tool
- `src/data_agent_baseline/run/`: single-task and benchmark runners
- `src/data_agent_baseline/benchmark/`: task schema and dataset loading
- `configs/react_baseline.example.yaml`: example runtime configuration

## Quick Start

1. Install `uv` by following the official guide: [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies:

```bash
uv sync
```

3. Prepare your local config from `configs/react_baseline.example.yaml`
4. Check project and dataset status:

```bash
uv run dabench status --config configs/react_baseline.example.yaml
```

5. Run one task or a benchmark:

```bash
uv run dabench run-task task_1 --config configs/react_baseline.example.yaml
uv run dabench run-benchmark --config configs/react_baseline.example.yaml
```

## Configuration

Example config:

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id: example_run_id
  max_workers: 8
  task_timeout_seconds: 600
```

Key fields:

- `dataset.root_path`: local dataset root
- `agent.model`: model name
- `agent.api_base`: OpenAI-compatible endpoint
- `agent.api_key`: model API key
- `run.output_dir`: output directory for runs
- `run.run_id`: optional run name
- `run.max_workers`: parallel worker count

## CLI

```bash
uv run dabench <command> --config PATH
```

Available commands:

- `status`: show project paths and dataset status
- `inspect-task`: inspect one task and list files under `context/`
- `run-task`: execute one task
- `run-benchmark`: execute multiple tasks

## Outputs

Run outputs are written under `artifacts/runs/<run_id>/`.

Typical files include:

- `summary.json`
- `<task_id>/trace.json`
- `<task_id>/prediction.csv`

## Competition Links

- Official website: [dataagent.top](https://dataagent.top)
- Discord: [Join the community](https://discord.gg/vRr7uyK9)
- Official starter kit: [HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit)

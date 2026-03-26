# data-agents

[English](README.md) | 中文

这个仓库是我参加 **KDD Cup 2026 DataAgent-Bench** 的比赛项目。  
整体代码基于官方 starter kit 做了整理和修改，用来承载我自己的实验、运行流程和迭代记录。

## 这个仓库是什么

- 一个基于官方 baseline 改出来的个人参赛代码仓库
- 一个用于查看任务、运行单题和批量跑 benchmark 的本地工程
- 一个集中管理 prompt、runtime、配置和实验产物的位置

## 项目结构

```text
.
├── configs/
├── src/data_agent_baseline/
├── artifacts/
├── README.md
├── README.zh.md
└── pyproject.toml
```

主要目录说明：

- `src/data_agent_baseline/agents/`：模型调用、prompt 和 ReAct 主循环
- `src/data_agent_baseline/tools/`：文件系统、SQLite、Python 执行和答案提交工具
- `src/data_agent_baseline/run/`：单任务和批量运行逻辑
- `src/data_agent_baseline/benchmark/`：任务结构和数据集加载逻辑
- `configs/react_baseline.example.yaml`：示例配置文件

## 快速开始

1. 先安装 `uv`，参考官方文档：[uv installation](https://docs.astral.sh/uv/getting-started/installation/)
2. 安装依赖：

```bash
uv sync
```

3. 基于 `configs/react_baseline.example.yaml` 准备你自己的配置
4. 检查项目状态和数据路径：

```bash
uv run dabench status --config configs/react_baseline.example.yaml
```

5. 运行单题或批量 benchmark：

```bash
uv run dabench run-task task_1 --config configs/react_baseline.example.yaml
uv run dabench run-benchmark --config configs/react_baseline.example.yaml
```

## 配置

示例配置如下：

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

关键字段：

- `dataset.root_path`：本地数据集路径
- `agent.model`：模型名称
- `agent.api_base`：兼容 OpenAI 的接口地址
- `agent.api_key`：模型调用密钥
- `run.output_dir`：运行输出目录
- `run.run_id`：可选的运行名称
- `run.max_workers`：并行 worker 数

## CLI

```bash
uv run dabench <command> --config PATH
```

可用命令：

- `status`：查看项目路径和数据集状态
- `inspect-task`：查看单个任务和 `context/` 文件列表
- `run-task`：运行单个任务
- `run-benchmark`：批量运行多个任务

## 输出

运行结果会写到 `artifacts/runs/<run_id>/` 下。

常见产物包括：

- `summary.json`
- `<task_id>/trace.json`
- `<task_id>/prediction.csv`

## 赛事链接

- 官方网站：[dataagent.top](https://dataagent.top)
- Discord：[Join the community](https://discord.gg/vRr7uyK9)
- 官方 starter kit：[HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit)



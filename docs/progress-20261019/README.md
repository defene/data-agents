# KDD 项目进度归档（2026-10-19）

本次按用户要求保存项目进度，确认 GitHub 远端保存成功后清理本地 `data/`。原分支为 `codex/agent-v2-overhaul`，归档前 HEAD 为 `5d173dc45955529689890ad961d5ac5d7baf947c`。

## 当前代码状态

- 已提交历史包含数据结构预分析、反编造提示、answer/finish policy 拆分，以及多轮候选实验与回退。
- 本次保存原工作区的四个代码文件：`prompt.py`、`react.py`、`runtime.py`、新增的 `shape_hints.py`。它们为源数据中的独立姓名列提供输出结构提示，并在运行状态及用户提示中传递提示。
- 本次是工作进度快照，没有修改上述实现，也没有运行新的付费模型评测。当前提示策略的泛化效果尚待完整 benchmark 验证。
- 4 月 11 日的本地目录中，一个仅有 `task_67` 产物，另一个为空，不能据此报告完整评测成绩。

## 历史证据

`analysis/` 原样保存现有实验比较 JSON 和 Markdown 报告；`run-summaries/` 保存各运行根目录中已有的 `summary.json`。这些内容是历史结果，不代表本次代码快照的成绩，运行成功数也不等于答对数。

例如，4 月 9 日 answer-policy extraction 报告记录 `e10a7b2` 的三次成绩为 23、25、23（每轮 50 题），平均 23.67/50。之后还有候选实验与回退，不能将该成绩直接归于本次新增的 shape hints。

原始 traces、predictions、scratch 及全部 `artifacts/` 留在本地。与 KDD 无关的 CZU/SCU 绘图文件及本机 `.claude/` 设置留在原处，未纳入上传。

## 数据清理与恢复

- `data-manifest.json` 记录删除前 `data/` 下每个文件的相对路径、大小、修改时间和 SHA-256；它是校验清单，不包含原始数据。
- `dataset-README.md` 保留原数据包说明。
- 上传的是代码、进度报告和清单；本次不上传原始数据，也不宣称原始数据已备份。
- 仅在确认进度提交已到达 GitHub 后执行清理，并另存 `cleanup-receipt.json` 回执。
- 再次在本机跑 benchmark 前，需要重新取得 PHASE 1 Demo Data 并恢复 `data/public/input` 和 `data/public/output`，可用清单校验内容。现有配置路径仍指向该位置。

本次检查记录见 `validation.json`；归档文件的完整性清单见 `archive-manifest.json`。

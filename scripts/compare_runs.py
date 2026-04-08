from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as eval_module  # noqa: E402


@dataclass(slots=True)
class TaskComparison:
    task_id: str
    difficulty: str
    base_score: int
    candidate_score: int
    delta: int
    base_detail: str
    candidate_detail: str
    base_prediction_exists: bool
    candidate_prediction_exists: bool
    base_step_count: int | None
    candidate_step_count: int | None
    step_delta: int | None


def _prediction_exists(run_dir: Path, task_id: str) -> bool:
    return (run_dir / task_id / "prediction.csv").exists()


def _load_step_count(run_dir: Path, task_id: str) -> int | None:
    trace_path = run_dir / task_id / "trace.json"
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    return len(steps)


def _score_run_task(run_dir: Path, task_id: str) -> tuple[int, str]:
    gold_path = REPO_ROOT / "data" / "public" / "output" / task_id / "gold.csv"
    pred_path = run_dir / task_id / "prediction.csv"
    return eval_module.score_task(gold_path, pred_path)


def _score_by_ids(run_dir: Path, task_ids: list[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for task_id in task_ids:
        score, _detail = _score_run_task(run_dir, task_id)
        scores[task_id] = score
    return scores


def _difficulty_ids() -> dict[str, list[str]]:
    by_diff: dict[str, list[str]] = {}
    for task_id in eval_module.ALL_IDS:
        difficulty = eval_module.get_difficulty(task_id)
        by_diff.setdefault(difficulty, []).append(task_id)
    return by_diff


def _safe_mean(values: list[int]) -> float | None:
    if not values:
        return None
    return mean(values)


def compare_runs(base_run_dir: Path, candidate_run_dir: Path) -> dict[str, Any]:
    comparisons: list[TaskComparison] = []
    by_diff = _difficulty_ids()

    for task_id in eval_module.ALL_IDS:
        difficulty = eval_module.get_difficulty(task_id)
        base_score, base_detail = _score_run_task(base_run_dir, task_id)
        candidate_score, candidate_detail = _score_run_task(candidate_run_dir, task_id)
        base_steps = _load_step_count(base_run_dir, task_id)
        candidate_steps = _load_step_count(candidate_run_dir, task_id)
        step_delta = (
            None if base_steps is None or candidate_steps is None else candidate_steps - base_steps
        )
        comparisons.append(
            TaskComparison(
                task_id=task_id,
                difficulty=difficulty,
                base_score=base_score,
                candidate_score=candidate_score,
                delta=candidate_score - base_score,
                base_detail=base_detail.splitlines()[0],
                candidate_detail=candidate_detail.splitlines()[0],
                base_prediction_exists=_prediction_exists(base_run_dir, task_id),
                candidate_prediction_exists=_prediction_exists(candidate_run_dir, task_id),
                base_step_count=base_steps,
                candidate_step_count=candidate_steps,
                step_delta=step_delta,
            )
        )

    fixed = [item for item in comparisons if item.delta > 0]
    regressed = [item for item in comparisons if item.delta < 0]

    def summarize(ids: list[str]) -> dict[str, Any]:
        base_scores = _score_by_ids(base_run_dir, ids)
        cand_scores = _score_by_ids(candidate_run_dir, ids)
        base_correct = sum(base_scores.values())
        cand_correct = sum(cand_scores.values())
        return {
            "task_count": len(ids),
            "base_correct": base_correct,
            "candidate_correct": cand_correct,
            "delta": cand_correct - base_correct,
        }

    summary_by_diff = {
        diff: summarize(ids) for diff, ids in sorted(by_diff.items(), key=lambda item: item[0])
    }

    base_step_values = [item.base_step_count for item in comparisons if item.base_step_count is not None]
    candidate_step_values = [
        item.candidate_step_count for item in comparisons if item.candidate_step_count is not None
    ]
    fixed_step_values = [item.step_delta for item in fixed if item.step_delta is not None]
    regressed_step_values = [item.step_delta for item in regressed if item.step_delta is not None]

    return {
        "base_run_dir": str(base_run_dir.resolve()),
        "candidate_run_dir": str(candidate_run_dir.resolve()),
        "overall": {
            "task_count": len(comparisons),
            "base_correct": sum(item.base_score for item in comparisons),
            "candidate_correct": sum(item.candidate_score for item in comparisons),
            "delta": sum(item.delta for item in comparisons),
            "base_answered": sum(int(item.base_prediction_exists) for item in comparisons),
            "candidate_answered": sum(int(item.candidate_prediction_exists) for item in comparisons),
            "base_avg_steps": _safe_mean([value for value in base_step_values if value is not None]),
            "candidate_avg_steps": _safe_mean(
                [value for value in candidate_step_values if value is not None]
            ),
        },
        "by_difficulty": summary_by_diff,
        "fixed_tasks": [asdict(item) for item in fixed],
        "regressed_tasks": [asdict(item) for item in regressed],
        "step_deltas": {
            "fixed_avg_step_delta": _safe_mean([value for value in fixed_step_values if value is not None]),
            "regressed_avg_step_delta": _safe_mean(
                [value for value in regressed_step_values if value is not None]
            ),
        },
        "task_comparisons": [asdict(item) for item in comparisons],
    }


def _format_score(correct: int, total: int) -> str:
    rate = 0.0 if total == 0 else (correct / total * 100)
    return f"{correct}/{total} ({rate:.0f}%)"


def _print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    total = overall["task_count"]
    print("Run Comparison")
    print(f"Base:      {report['base_run_dir']}")
    print(f"Candidate: {report['candidate_run_dir']}")
    print()
    print(
        "Overall:   "
        f"{_format_score(overall['base_correct'], total)} -> "
        f"{_format_score(overall['candidate_correct'], total)} "
        f"(delta {overall['delta']:+d})"
    )
    print(
        "Answered:  "
        f"{overall['base_answered']}/{total} -> "
        f"{overall['candidate_answered']}/{total}"
    )
    print(
        "Avg steps: "
        f"{overall['base_avg_steps'] if overall['base_avg_steps'] is not None else 'n/a'} -> "
        f"{overall['candidate_avg_steps'] if overall['candidate_avg_steps'] is not None else 'n/a'}"
    )
    print()
    print("By difficulty:")
    for diff, summary in report["by_difficulty"].items():
        print(
            f"- {diff}: "
            f"{_format_score(summary['base_correct'], summary['task_count'])} -> "
            f"{_format_score(summary['candidate_correct'], summary['task_count'])} "
            f"(delta {summary['delta']:+d})"
        )

    print()
    fixed = report["fixed_tasks"]
    regressed = report["regressed_tasks"]
    print(f"Fixed tasks ({len(fixed)}):")
    if fixed:
        for item in fixed:
            print(
                f"- {item['task_id']} [{item['difficulty']}]: "
                f"{item['base_detail']} -> {item['candidate_detail']}"
            )
    else:
        print("- none")

    print()
    print(f"Regressed tasks ({len(regressed)}):")
    if regressed:
        for item in regressed:
            print(
                f"- {item['task_id']} [{item['difficulty']}]: "
                f"{item['base_detail']} -> {item['candidate_detail']}"
            )
    else:
        print("- none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark run directories.")
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = compare_runs(args.base_run_dir, args.candidate_run_dir)
    _print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"JSON report written to: {args.json_out.resolve()}")


if __name__ == "__main__":
    main()

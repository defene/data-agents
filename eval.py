"""
Evaluate predictions using official KDD Cup 2026 scoring rules:
- Column names are IGNORED
- Each column is an unordered value vector
- All gold columns must be fully covered by prediction columns
- Extra prediction columns are OK
- Score is binary: 1 (all gold columns matched) or 0
"""
import argparse
import csv
from pathlib import Path

EASY_IDS = [
    "task_11", "task_19", "task_22", "task_24", "task_25",
    "task_26", "task_27", "task_38", "task_64", "task_67",
    "task_74", "task_75", "task_80", "task_86", "task_89",
]

ALL_IDS = sorted(
    [p.name for p in Path("data/public/input").iterdir() if p.is_dir()],
    key=lambda x: int(x.split("_")[1]),
)


def read_csv_cols(path):
    """Read CSV and return a dict of {col_name: [values...]}."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = rows[0]
    data = rows[1:]
    cols = {}
    for ci, name in enumerate(header):
        cols[name] = sorted([r[ci].strip() for r in data])
    return cols


def col_vectors(csv_cols):
    """Extract sorted value vectors (ignoring column names)."""
    if csv_cols is None:
        return None
    return [tuple(v) for v in csv_cols.values()]


def score_task(gold_path, pred_path):
    """
    Official scoring: each gold column (as sorted value vector) must
    exist in the prediction columns. Column names ignored.
    Returns (score, detail_string).
    """
    gold_cols = read_csv_cols(gold_path)
    pred_cols = read_csv_cols(pred_path)

    if pred_cols is None:
        return 0, "No prediction file"

    gold_vectors = col_vectors(gold_cols)
    pred_vectors = col_vectors(pred_cols)

    pred_vector_set = set(pred_vectors)

    missing = []
    for gi, gv in enumerate(gold_vectors):
        if gv not in pred_vector_set:
            gold_col_name = list(gold_cols.keys())[gi]
            missing.append((gold_col_name, gv))

    if not missing:
        extra = len(pred_vectors) - len(gold_vectors)
        detail = "EXACT" if extra == 0 else f"OK (pred has {extra} extra col(s))"
        return 1, detail
    else:
        details = []
        for col_name, gv in missing:
            vals_preview = list(gv[:5])
            if len(gv) > 5:
                vals_preview.append("...")
            details.append(f"  gold col '{col_name}' = {vals_preview} not found in pred")
        return 0, f"{len(missing)}/{len(gold_vectors)} gold col(s) missing:\n" + "\n".join(details)


def evaluate(task_ids, label, run_dir: Path):
    gold_dir = Path("data/public/output")

    total = len(task_ids)
    correct = 0

    print(f"\n{'='*70}")
    print(f"  {label}: {total} tasks")
    print(f"{'='*70}")
    print(f"{'Task':<12} {'Score':<8} {'Detail'}")
    print("-" * 70)

    for tid in task_ids:
        gold_path = gold_dir / tid / "gold.csv"
        pred_path = run_dir / tid / "prediction.csv"
        sc, detail = score_task(gold_path, pred_path)
        correct += sc
        first_line = detail.split("\n")[0]
        print(f"{tid:<12} {sc:<8} {first_line}")
        if sc == 0 and "\n" in detail:
            for line in detail.split("\n")[1:]:
                print(f"{'':>20}{line}")

    print(f"\nScore: {correct}/{total} ({correct/total*100:.0f}%)")
    return correct, total


def get_difficulty(tid):
    import json
    task_json = Path(f"data/public/input/{tid}/task.json")
    return json.loads(task_json.read_text(encoding="utf-8"))["difficulty"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score predictions under a run directory vs public gold.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/run_v4_structured"),
        help="Run output root containing <task_id>/prediction.csv (default: artifacts/runs/run_v4_structured)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir

    print(f"Run directory: {run_dir.resolve()}")

    # Easy tasks
    easy_correct, easy_total = evaluate(EASY_IDS, "Easy Tasks", run_dir)

    # All tasks by difficulty
    by_diff = {}
    for tid in ALL_IDS:
        d = get_difficulty(tid)
        by_diff.setdefault(d, []).append(tid)

    print(f"\n{'='*70}")
    print("  Summary by Difficulty")
    print(f"{'='*70}")

    grand_correct = 0
    grand_total = 0
    for diff in ["easy", "medium", "hard", "extreme"]:
        ids = by_diff.get(diff, [])
        c, t = evaluate(ids, f"{diff.upper()} ({len(ids)} tasks)", run_dir)
        grand_correct += c
        grand_total += t

    print(f"\n{'='*70}")
    print(f"  OVERALL: {grand_correct}/{grand_total} ({grand_correct/grand_total*100:.0f}%)")
    print(f"{'='*70}")

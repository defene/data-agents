"""Data profiler that pre-computes a structural sketch of every context asset.

The profile is injected into the agent prompt at bootstrap so the model can
make grounded decisions without burning steps on exploratory tool calls.

Design notes:
- We never raise: profiling is best-effort and any failure degrades into
  a {"error": ...} entry instead of crashing bootstrap.
- We deliberately keep payloads compact (per-file caps) so the prompt stays
  under control even on Hard/Extreme tasks.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask


_CSV_PROFILE_ROW_LIMIT = 2000
_CSV_SAMPLE_VALUES = 8
_JSON_PREVIEW_CHARS = 600
_SQLITE_SAMPLE_ROWS = 5
_DOC_PREVIEW_CHARS = 400


def _is_numeric(value: str) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _profile_csv_column(values: list[str]) -> dict[str, Any]:
    non_null = [v for v in values if v not in (None, "")]
    null_count = len(values) - len(non_null)

    distinct: list[str] = []
    seen: set[str] = set()
    for v in non_null:
        if v not in seen:
            seen.add(v)
            distinct.append(v)
        if len(distinct) >= _CSV_SAMPLE_VALUES:
            break

    numeric_values: list[float] = []
    if non_null and all(_is_numeric(v) for v in non_null[:50]):
        for v in non_null:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                pass

    info: dict[str, Any] = {
        "null_count": null_count,
        "distinct_sample": distinct,
    }
    if numeric_values:
        info["dtype_hint"] = "numeric"
        info["min"] = min(numeric_values)
        info["max"] = max(numeric_values)
        info["has_zero"] = any(v == 0 for v in numeric_values)
    else:
        info["dtype_hint"] = "text"
    return info


def profile_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            rows = []
            for i, row in enumerate(reader):
                if i > _CSV_PROFILE_ROW_LIMIT:
                    break
                rows.append(row)
    except Exception as exc:
        return {"error": f"csv_read_failed: {exc}"}

    if not rows:
        return {"columns": [], "row_count_seen": 0}

    header = rows[0]
    data = rows[1:]
    columns: list[dict[str, Any]] = []
    for idx, name in enumerate(header):
        column_values = [r[idx] if idx < len(r) else "" for r in data]
        columns.append({"name": name, **_profile_csv_column(column_values)})

    return {
        "columns": columns,
        "row_count_seen": len(data),
        "row_count_capped": len(data) >= _CSV_PROFILE_ROW_LIMIT,
    }


def profile_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        payload = json.loads(text)
    except Exception as exc:
        return {"error": f"json_read_failed: {exc}"}

    summary: dict[str, Any] = {}
    if isinstance(payload, dict):
        summary["top_level_keys"] = list(payload.keys())[:20]
        if "records" in payload and isinstance(payload["records"], list):
            summary["records_count"] = len(payload["records"])
            if payload["records"]:
                first = payload["records"][0]
                if isinstance(first, dict):
                    summary["record_fields"] = list(first.keys())
                    summary["sample_records"] = payload["records"][:3]
        if "table" in payload:
            summary["table_name"] = payload.get("table")
    elif isinstance(payload, list):
        summary["list_length"] = len(payload)
        if payload and isinstance(payload[0], dict):
            summary["item_fields"] = list(payload[0].keys())
        summary["sample_items"] = payload[:3]

    preview = json.dumps(payload, ensure_ascii=False)[:_JSON_PREVIEW_CHARS]
    summary["preview"] = preview
    return summary


def profile_sqlite(path: Path) -> dict[str, Any]:
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            tables_info = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()

            tables: list[dict[str, Any]] = []
            for name, create_sql in tables_info:
                try:
                    cursor = conn.execute(f"SELECT * FROM \"{name}\" LIMIT {_SQLITE_SAMPLE_ROWS}")
                    cols = [d[0] for d in cursor.description or []]
                    sample = cursor.fetchall()
                    row_count = conn.execute(f"SELECT COUNT(*) FROM \"{name}\"").fetchone()[0]
                except Exception as exc:
                    tables.append({"name": name, "error": str(exc)})
                    continue
                tables.append({
                    "name": name,
                    "create_sql": create_sql,
                    "columns": cols,
                    "row_count": row_count,
                    "sample_rows": [list(r) for r in sample],
                })
    except Exception as exc:
        return {"error": f"sqlite_read_failed: {exc}"}

    return {"tables": tables}


def profile_doc(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"error": f"doc_read_failed: {exc}"}

    lines = text.splitlines()
    headings = [ln.strip() for ln in lines if ln.lstrip().startswith("#")][:30]
    return {
        "char_count": len(text),
        "line_count": len(lines),
        "headings": headings,
        "preview": text[:_DOC_PREVIEW_CHARS],
    }


def profile_context(task: PublicTask) -> dict[str, Any]:
    """Walk the task context dir and produce a per-file structural profile."""
    context_root = task.context_dir
    profile: dict[str, Any] = {
        "csv": {},
        "json": {},
        "sqlite": {},
        "doc": {},
        "other": [],
    }

    if not context_root.exists():
        return profile

    for child in sorted(context_root.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(context_root).as_posix()
        suffix = child.suffix.lower()
        try:
            if suffix == ".csv":
                profile["csv"][rel] = profile_csv(child)
            elif suffix == ".json":
                profile["json"][rel] = profile_json(child)
            elif suffix in (".db", ".sqlite", ".sqlite3"):
                profile["sqlite"][rel] = profile_sqlite(child)
            elif suffix in (".md", ".markdown", ".txt"):
                profile["doc"][rel] = profile_doc(child)
            else:
                profile["other"].append(rel)
        except Exception as exc:
            profile["other"].append({"path": rel, "error": str(exc)})

    return profile


def render_profile_for_prompt(profile: dict[str, Any]) -> str:
    """Render the profile as a compact, prompt-friendly text block."""
    lines: list[str] = ["Data Profile (auto-generated, ground truth from disk):"]

    for path, info in profile.get("csv", {}).items():
        lines.append(f"\n[CSV] {path}")
        if "error" in info:
            lines.append(f"  error: {info['error']}")
            continue
        lines.append(f"  rows_seen: {info.get('row_count_seen')}"
                     + (" (capped)" if info.get("row_count_capped") else ""))
        for col in info.get("columns", []):
            parts = [f"{col['name']} ({col['dtype_hint']})"]
            if col.get("null_count"):
                parts.append(f"nulls={col['null_count']}")
            if col.get("dtype_hint") == "numeric":
                parts.append(f"min={col.get('min')}")
                parts.append(f"max={col.get('max')}")
                if col.get("has_zero"):
                    parts.append("contains_zero")
            sample = col.get("distinct_sample") or []
            if sample:
                parts.append(f"sample={sample}")
            lines.append("  - " + " ".join(parts))

    for path, info in profile.get("json", {}).items():
        lines.append(f"\n[JSON] {path}")
        if "error" in info:
            lines.append(f"  error: {info['error']}")
            continue
        if "table_name" in info:
            lines.append(f"  table_name: {info['table_name']}")
        if "records_count" in info:
            lines.append(f"  records_count: {info['records_count']}")
        if "record_fields" in info:
            lines.append(f"  record_fields: {info['record_fields']}")
        if "sample_records" in info:
            lines.append(f"  sample_records: {json.dumps(info['sample_records'], ensure_ascii=False)}")
        if "list_length" in info:
            lines.append(f"  list_length: {info['list_length']}")
        if "item_fields" in info:
            lines.append(f"  item_fields: {info['item_fields']}")
        if "top_level_keys" in info and "records_count" not in info:
            lines.append(f"  top_level_keys: {info['top_level_keys']}")

    for path, info in profile.get("sqlite", {}).items():
        lines.append(f"\n[SQLITE] {path}")
        if "error" in info:
            lines.append(f"  error: {info['error']}")
            continue
        for tbl in info.get("tables", []):
            if "error" in tbl:
                lines.append(f"  table {tbl['name']}: error={tbl['error']}")
                continue
            lines.append(f"  table {tbl['name']} (rows={tbl['row_count']})")
            lines.append(f"    columns: {tbl['columns']}")
            if tbl.get("create_sql"):
                lines.append(f"    create_sql: {tbl['create_sql']}")
            if tbl.get("sample_rows"):
                lines.append(f"    sample_rows: {tbl['sample_rows'][:3]}")

    for path, info in profile.get("doc", {}).items():
        lines.append(f"\n[DOC] {path}")
        if "error" in info:
            lines.append(f"  error: {info['error']}")
            continue
        lines.append(f"  chars={info['char_count']} lines={info['line_count']}")
        if info.get("headings"):
            lines.append(f"  headings: {info['headings']}")
        if info.get("preview"):
            preview = info["preview"].replace("\n", " ⏎ ")
            lines.append(f"  preview: {preview}")

    return "\n".join(lines)

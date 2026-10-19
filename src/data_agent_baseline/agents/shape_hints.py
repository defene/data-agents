"""Detect likely answer-shape pitfalls and emit a short hint string.

The agent prompt tells the model to preserve source column granularity.
But models still concatenate first_name+last_name into "full_name". This
module looks at the profile + question and emits an explicit warning when
a known pitfall pattern is detected, so the warning lands in the user
message at every step.
"""

from __future__ import annotations

from typing import Any


_NAME_PAIR_HINTS = (
    ("first_name", "last_name"),
    ("firstname", "lastname"),
    ("fname", "lname"),
    ("given_name", "family_name"),
)


def _flatten_csv_columns(profile: dict[str, Any]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for path, info in profile.get("csv", {}).items():
        if "columns" not in info:
            continue
        out.append((path, [c["name"] for c in info["columns"]]))
    return out


def _flatten_sqlite_columns(profile: dict[str, Any]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for path, info in profile.get("sqlite", {}).items():
        for tbl in info.get("tables", []) or []:
            cols = tbl.get("columns") or []
            if cols:
                out.append((f"{path}::{tbl.get('name')}", list(cols)))
    return out


def _flatten_json_columns(profile: dict[str, Any]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for path, info in profile.get("json", {}).items():
        fields = info.get("record_fields") or info.get("item_fields")
        if fields:
            out.append((path, list(fields)))
    return out


def detect_shape_hints(question: str, profile: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if not profile:
        return hints

    q = (question or "").lower()
    all_columns: list[tuple[str, list[str]]] = []
    all_columns.extend(_flatten_csv_columns(profile))
    all_columns.extend(_flatten_sqlite_columns(profile))
    all_columns.extend(_flatten_json_columns(profile))

    asks_for_name = any(
        kw in q for kw in ("full name", "name of", "names of", "who ", "which person", "which member", "which student", "which driver", "which user")
    )
    if asks_for_name:
        for source, cols in all_columns:
            lowered = {c.lower(): c for c in cols}
            for first_kw, last_kw in _NAME_PAIR_HINTS:
                if first_kw in lowered and last_kw in lowered:
                    first_col = lowered[first_kw]
                    last_col = lowered[last_kw]
                    hints.append(
                        f"The source `{source}` has separate `{first_col}` and `{last_col}` columns. "
                        f"Return BOTH columns separately. Do not concatenate them into a single full_name column."
                    )
                    break

    return hints


def render_shape_hints(hints: list[str]) -> str:
    if not hints:
        return ""
    body = "\n".join(f"- {h}" for h in hints)
    return "Answer Shape Hints (do not ignore):\n" + body

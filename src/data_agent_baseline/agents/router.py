from __future__ import annotations

import re

from data_agent_baseline.agents.runtime import RouteMemory, SourceIndexMemory

DB_SUFFIXES = (".db", ".duckdb", ".sqlite")
DOC_SUFFIXES = (".md", ".rst", ".txt")
TABULAR_SUFFIXES = (".csv", ".json", ".parquet", ".tsv", ".xls", ".xlsx")


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _priority_key(path: str) -> tuple[int, str]:
    lowered = path.lower()
    if lowered.endswith((".db", ".sqlite", ".duckdb")):
        return (0, lowered)
    if lowered.endswith((".csv", ".tsv", ".json", ".parquet", ".xls", ".xlsx")):
        return (1, lowered)
    if lowered.startswith("doc/"):
        return (2, lowered)
    if lowered == "knowledge.md":
        return (3, lowered)
    return (4, lowered)


def build_source_index(context_listing: dict[str, object]) -> SourceIndexMemory:
    tabular_files: list[str] = []
    doc_files: list[str] = []
    db_files: list[str] = []
    other_files: list[str] = []

    raw_entries = context_listing.get("entries", [])
    if not isinstance(raw_entries, list):
        return SourceIndexMemory()

    for entry in raw_entries:
        if not isinstance(entry, dict) or entry.get("kind") != "file":
            continue

        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        path = raw_path.strip()
        lowered = path.lower()
        if lowered.endswith(DB_SUFFIXES):
            db_files.append(path)
        elif lowered.endswith(TABULAR_SUFFIXES):
            tabular_files.append(path)
        elif lowered.endswith(DOC_SUFFIXES):
            doc_files.append(path)
        else:
            other_files.append(path)

    return SourceIndexMemory(
        tabular_files=sorted(_unique_preserve_order(tabular_files), key=_priority_key),
        doc_files=sorted(_unique_preserve_order(doc_files), key=_priority_key),
        db_files=sorted(_unique_preserve_order(db_files), key=_priority_key),
        other_files=sorted(_unique_preserve_order(other_files), key=_priority_key),
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _infer_task_type(question: str) -> str:
    lowered = _normalize_question(question)
    ratio_percentage_terms = (
        "percentage",
        "percent",
        "ratio",
        "proportion",
        "share",
        "fraction",
        "how many times",
        "times was",
        "times were",
        "times is",
        "times are",
    )
    average_terms = ("average", "avg", "mean")
    count_distinct_terms = ("distinct", "unique", "different ")
    count_terms = ("how many", "number of", "count")
    aggregate_terms = ("total", "sum", "minimum", "maximum", "max", "min")
    compare_terms = (
        " more than ",
        " less than ",
        " higher than ",
        " greater than ",
        " compared to ",
        " lower than ",
    )
    table_terms = ("list ", "show ", "which ", "what are ", "give all ", "provide all ", "return all ")

    if _contains_any(lowered, ratio_percentage_terms):
        return "ratio_percentage"
    if _contains_any(lowered, average_terms):
        return "average"
    if _contains_any(lowered, count_distinct_terms):
        return "count_distinct"
    if _contains_any(lowered, count_terms):
        return "count"
    if _contains_any(lowered, aggregate_terms):
        return "aggregate"
    if _contains_any(lowered, compare_terms):
        return "entity_compare"
    if _contains_any(lowered, table_terms):
        return "multi_row_extract"
    return "lookup"


def _contextual_doc_files(source_index: SourceIndexMemory) -> list[str]:
    return [path for path in source_index.doc_files if path.lower() != "knowledge.md"]


def _knowledge_doc_files(source_index: SourceIndexMemory) -> list[str]:
    return [path for path in source_index.doc_files if path.lower() == "knowledge.md"]


def _infer_solver_mode(source_index: SourceIndexMemory, task_type: str) -> str:
    contextual_docs = _contextual_doc_files(source_index)
    has_contextual_docs = bool(contextual_docs)
    has_db = bool(source_index.db_files)
    has_tabular = bool(source_index.tabular_files)
    doc_link_task_types = {"lookup", "multi_row_extract", "entity_compare"}

    if has_db:
        if has_contextual_docs and task_type in doc_link_task_types:
            return "doc_link"
        return "sql_first"
    if has_tabular:
        if has_contextual_docs and (
            task_type in doc_link_task_types
            or (task_type == "ratio_percentage" and len(source_index.tabular_files) <= 1)
        ):
            return "doc_link"
        return "tabular_first"
    if source_index.doc_files:
        return "doc_first"
    return "react_fallback"


def _infer_output_hint(question: str, task_type: str) -> str:
    lowered = _normalize_question(question)
    if task_type == "multi_row_extract":
        return "multi_row_table"
    if task_type in {
        "count",
        "average",
        "aggregate",
        "ratio_percentage",
        "count_distinct",
        "entity_compare",
    }:
        return "single_value_or_single_row"
    if " and " in lowered or "," in lowered or " include " in lowered:
        return "single_row_with_multiple_columns"
    return "single_row_lookup"


def _pick_primary_sources(source_index: SourceIndexMemory, solver_mode: str, task_type: str) -> list[str]:
    contextual_docs = _contextual_doc_files(source_index)
    knowledge_docs = _knowledge_doc_files(source_index)
    ordered_sources: list[str] = []

    if solver_mode == "sql_first":
        ordered_sources.extend(source_index.db_files)
        ordered_sources.extend(source_index.tabular_files)
        ordered_sources.extend(contextual_docs)
        ordered_sources.extend(knowledge_docs)
    elif solver_mode == "tabular_first":
        ordered_sources.extend(source_index.tabular_files)
        ordered_sources.extend(source_index.db_files)
        ordered_sources.extend(contextual_docs)
        ordered_sources.extend(knowledge_docs)
    elif solver_mode == "doc_link":
        if task_type == "ratio_percentage" and contextual_docs and not source_index.db_files:
            ordered_sources.extend(contextual_docs)
            ordered_sources.extend(source_index.db_files)
            ordered_sources.extend(source_index.tabular_files)
        else:
            ordered_sources.extend(source_index.db_files)
            ordered_sources.extend(source_index.tabular_files)
            ordered_sources.extend(contextual_docs)
        ordered_sources.extend(knowledge_docs)
    elif solver_mode == "doc_first":
        ordered_sources.extend(contextual_docs)
        ordered_sources.extend(knowledge_docs)
        ordered_sources.extend(source_index.db_files)
        ordered_sources.extend(source_index.tabular_files)
    else:
        ordered_sources.extend(source_index.db_files)
        ordered_sources.extend(source_index.tabular_files)
        ordered_sources.extend(contextual_docs)
        ordered_sources.extend(knowledge_docs)

    ordered_sources.extend(source_index.other_files)
    return _unique_preserve_order(ordered_sources)[:3]


def _build_route_rationale(task_type: str, solver_mode: str) -> str:
    if solver_mode == "doc_link":
        if task_type == "entity_compare":
            return (
                "Resolve the compared entities in structured data first, then use the linked "
                "document references only to extract the final comparable values."
            )
        if task_type == "ratio_percentage":
            return (
                "Use the main entity source to form the filtered set, then use the linked "
                "mapping or document evidence needed to compute the numerator, denominator, and final ratio."
            )
        if task_type == "lookup":
            return (
                "Use structured data to find the target record or ID first, then read only the "
                "specific document snippets needed to resolve the final label or attribute."
            )
        return (
            "Start from structured data, then use documents only for the narrow ID-to-value or "
            "definition lookup that structured files cannot answer alone."
        )
    if solver_mode == "sql_first":
        return "Prefer a precise SQL query over repeated previews or document browsing."
    if solver_mode == "tabular_first":
        return "Answer from the structured files directly unless a document is clearly required."
    if solver_mode == "doc_first":
        return "Use the documents as the primary source and search by exact terms or IDs."
    return "Use the fallback reasoning loop and choose the shortest verifiable path."


def infer_task_route(question: str, source_index: SourceIndexMemory) -> RouteMemory:
    task_type = _infer_task_type(question)
    solver_mode = _infer_solver_mode(source_index, task_type)
    primary_sources = _pick_primary_sources(source_index, solver_mode, task_type)
    supporting_sources = [
        path
        for path in (
            source_index.db_files
            + source_index.tabular_files
            + source_index.doc_files
            + source_index.other_files
        )
        if path not in primary_sources
    ]

    return RouteMemory(
        task_type=task_type,
        solver_mode=solver_mode,
        question_intent=task_type,
        primary_sources=primary_sources,
        supporting_sources=supporting_sources[:4],
        rationale=_build_route_rationale(task_type, solver_mode),
        output_hint=_infer_output_hint(question, task_type),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.doc_retrieval import (
    list_doc_chunks as build_doc_chunk_listing,
    read_doc_chunk as read_doc_chunk_content,
    search_doc_chunks,
)
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None
    memory_patch: dict[str, Any] | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _list_doc_chunks(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    chunk_chars = int(action_input.get("chunk_chars", 1400))
    chunk_overlap = int(action_input.get("chunk_overlap", 200))
    preview_chars = int(action_input.get("preview_chars", 240))
    return ToolExecutionResult(
        ok=True,
        content=build_doc_chunk_listing(
            path,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            preview_chars=preview_chars,
        ),
    )


def _search_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    query = str(action_input["query"])
    top_k = int(action_input.get("top_k", 5))
    chunk_chars = int(action_input.get("chunk_chars", 1400))
    chunk_overlap = int(action_input.get("chunk_overlap", 200))
    preview_chars = int(action_input.get("preview_chars", 240))
    return ToolExecutionResult(
        ok=True,
        content=search_doc_chunks(
            path,
            query,
            top_k=top_k,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            preview_chars=preview_chars,
        ),
    )


def _read_doc_chunk(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    chunk_id = str(action_input["chunk_id"])
    chunk_chars = int(action_input.get("chunk_chars", 1400))
    chunk_overlap = int(action_input.get("chunk_overlap", 200))
    return ToolExecutionResult(
        ok=True,
        content=read_doc_chunk_content(
            path,
            chunk_id,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    if task.scratch_dir is None:
        raise RuntimeError("Task scratch_dir is not configured for execute_python.")
    content = execute_python_code(
        context_root=task.context_dir,
        scratch_root=task.scratch_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings.")
    return [item.strip() for item in value if item.strip()]


def _normalize_output_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("mappings must be a non-empty list.")

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"mappings[{index}] must be an object.")

        question_target = item.get("question_target", "")
        source_kind = item.get("source_kind", "")
        granularity = item.get("granularity", "")
        format_rule = item.get("format_rule", "")

        if not isinstance(question_target, str) or not question_target.strip():
            raise ValueError(f"mappings[{index}].question_target must be a non-empty string.")
        if not isinstance(source_kind, str) or not source_kind.strip():
            raise ValueError(f"mappings[{index}].source_kind must be a non-empty string.")
        if not isinstance(granularity, str):
            raise ValueError(f"mappings[{index}].granularity must be a string.")
        if not isinstance(format_rule, str):
            raise ValueError(f"mappings[{index}].format_rule must be a string.")

        output_columns = _normalize_string_list(
            item.get("output_columns"),
            field_name=f"mappings[{index}].output_columns",
        )
        if not output_columns:
            raise ValueError(f"mappings[{index}].output_columns must contain at least one column.")

        normalized_items.append(
            {
                "question_target": question_target.strip(),
                "source_kind": source_kind.strip(),
                "source_refs": _normalize_string_list(
                    item.get("source_refs"),
                    field_name=f"mappings[{index}].source_refs",
                ),
                "output_columns": output_columns,
                "granularity": granularity.strip(),
                "format_rule": format_rule.strip(),
            }
        )

    return normalized_items


def _record_memory(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    section = action_input.get("section", "")
    fact = action_input.get("fact", "")
    if section not in {"schema", "rules", "entities", "evidence"}:
        return ToolExecutionResult(
            ok=False,
            content={"error": "section must be one of: schema, rules, entities, evidence."},
        )
    if not isinstance(fact, str) or not fact.strip():
        return ToolExecutionResult(ok=False, content={"error": "fact must be a non-empty string."})
    normalized_fact = fact.strip()
    return ToolExecutionResult(
        ok=True,
        content={"status": "recorded", "section": section, "fact": normalized_fact},
        memory_patch={"append": {"section": section, "fact": normalized_fact}},
    )


def _set_output_mapping(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    mappings = _normalize_output_mappings(action_input.get("mappings"))
    return ToolExecutionResult(
        ok=True,
        content={"status": "output_mapping_set", "mapping_count": len(mappings), "mappings": mappings},
        memory_patch={"set_output_mapping": {"mappings": mappings}},
    )


def _set_answer_shape(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        return ToolExecutionResult(
            ok=False,
            content={"error": "columns must be a non-empty list of strings."},
        )

    expected_rows = action_input.get("expected_rows")
    if expected_rows is not None:
        expected_rows = int(expected_rows)
        if expected_rows < 0:
            raise ValueError("expected_rows must be >= 0.")

    sort_requirement = action_input.get("sort_requirement", "")
    if not isinstance(sort_requirement, str):
        raise ValueError("sort_requirement must be a string.")

    value_format_rules = _normalize_string_list(
        action_input.get("value_format_rules"),
        field_name="value_format_rules",
    )

    payload = {
        "columns": [item.strip() for item in columns],
        "expected_rows": expected_rows,
        "sort_requirement": sort_requirement.strip(),
        "value_format_rules": value_format_rules,
    }
    return ToolExecutionResult(
        ok=True,
        content={"status": "answer_shape_set", **payload},
        memory_patch={"set_answer_shape": payload},
    )


def _set_plan(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    done = _normalize_string_list(action_input.get("done"), field_name="done")
    todo = _normalize_string_list(action_input.get("todo"), field_name="todo")
    next_action = action_input.get("next_action", "")
    if not isinstance(next_action, str):
        raise ValueError("next_action must be a string.")

    payload = {
        "done": done,
        "todo": todo,
        "next_action": next_action.strip(),
    }
    return ToolExecutionResult(
        ok=True,
        content={"status": "plan_set", **payload},
        memory_patch={"set_plan": payload},
    )


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code inside an isolated per-task scratch directory. "
                "The Python globals `context_root` and `scratch_root` are available as Path "
                "objects. Read original task files from `context_root / <relative_path>` and "
                "write any temporary outputs only under `scratch_root`. The tool returns the "
                "code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": (
                    "import pandas as pd\n"
                    "df = pd.read_csv(context_root / 'csv/data.csv')\n"
                    "df.head().to_csv(scratch_root / 'preview.csv', index=False)\n"
                    "print(df.head())"
                ),
            },
        ),
        "record_memory": ToolSpec(
            name="record_memory",
            description=(
                "Record a reusable fact into structured working memory. "
                "Use section='schema' for joins, columns, or files; 'rules' for definitions or thresholds; "
                "'entities' for target IDs, names, or date windows; and 'evidence' for verified query results."
            ),
            input_schema={
                "section": "entities",
                "fact": "target race_id=18",
            },
        ),
        "set_output_mapping": ToolSpec(
            name="set_output_mapping",
            description=(
                "Map the question target to its source fields or derived expression before finalizing the answer shape. "
                "Use this to declare what each output column represents and whether the result is detail rows or an aggregate."
            ),
            input_schema={
                "mappings": [
                    {
                        "question_target": "driver surname",
                        "source_kind": "raw_column",
                        "source_refs": ["results.surname"],
                        "output_columns": ["surname"],
                        "granularity": "detail_rows",
                        "format_rule": "preserve source formatting",
                    }
                ]
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "list_doc_chunks": ToolSpec(
            name="list_doc_chunks",
            description=(
                "Split a document into numbered chunks and list chunk previews. "
                "Use this before reading a long document in detail."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "chunk_chars": 1400,
                "chunk_overlap": 200,
                "preview_chars": 240,
            },
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc_chunk": ToolSpec(
            name="read_doc_chunk",
            description="Read one specific chunk from a text-like document inside context.",
            input_schema={
                "path": "relative/path/to/file.md",
                "chunk_id": "chunk_0003",
                "chunk_chars": 1400,
                "chunk_overlap": 200,
            },
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read the beginning of a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
        "search_doc": ToolSpec(
            name="search_doc",
            description=(
                "Keyword-search a long document and return the most relevant chunks. "
                "Use this to locate definitions, thresholds, entities, or evidence before reading chunks."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "query": "status definition threshold business rule",
                "top_k": 5,
                "chunk_chars": 1400,
                "chunk_overlap": 200,
                "preview_chars": 240,
            },
        ),
        "set_answer_shape": ToolSpec(
            name="set_answer_shape",
            description=(
                "Define the expected final answer table before you submit. "
                "Use this once your output mapping is clear and you know target columns, row count, sorting, or value-format constraints."
            ),
            input_schema={
                "columns": ["name", "percentage"],
                "expected_rows": 1,
                "sort_requirement": "single summary row",
                "value_format_rules": ["preserve source formatting"],
            },
        ),
        "set_plan": ToolSpec(
            name="set_plan",
            description=(
                "Track what is done and what remains. "
                "Update this after major discoveries so you stop re-searching and move to the next concrete action."
            ),
            input_schema={
                "done": ["identified target entity"],
                "todo": ["run final aggregation", "submit answer"],
                "next_action": "query the results table for the final metric",
            },
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "record_memory": _record_memory,
        "set_output_mapping": _set_output_mapping,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "list_doc_chunks": _list_doc_chunks,
        "read_csv": _read_csv,
        "read_doc_chunk": _read_doc_chunk,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "search_doc": _search_doc,
        "set_answer_shape": _set_answer_shape,
        "set_plan": _set_plan,
    }
    return ToolRegistry(specs=specs, handlers=handlers)

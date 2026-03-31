from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            stdout_encoding = getattr(original_stdout, "encoding", None) or "utf-8"
            stderr_encoding = getattr(original_stderr, "encoding", None) or "utf-8"

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=stdout_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=stderr_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_bootstrap_script(script_path: Path, code: str) -> None:
    encoded_user_code = repr(code)
    script_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "import traceback",
                "",
                "context_root = Path(os.environ['DABENCH_CONTEXT_ROOT'])",
                "scratch_root = Path(os.environ['DABENCH_SCRATCH_ROOT'])",
                "os.chdir(scratch_root)",
                f"user_code = {encoded_user_code}",
                "namespace = {",
                "    '__builtins__': __builtins__,",
                "    '__name__': '__main__',",
                "    'Path': Path,",
                "    'context_root': context_root,",
                "    'scratch_root': scratch_root,",
                "}",
                "try:",
                "    exec(compile(user_code, str(context_root / '<user_code>'), 'exec'), namespace, namespace)",
                "except BaseException as exc:  # noqa: BLE001",
                "    print(traceback.format_exc(), file=os.sys.stderr, end='')",
                "    raise",
            ]
        ),
        encoding="utf-8",
    )


def execute_python_code(
    context_root: Path,
    scratch_root: Path,
    code: str,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    resolved_context_root = context_root.resolve()
    resolved_scratch_root = scratch_root.resolve()
    resolved_scratch_root.mkdir(parents=True, exist_ok=True)
    exec_dir = resolved_scratch_root / f"exec-{uuid.uuid4().hex[:8]}"
    exec_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = exec_dir / "stdout.txt"
    stderr_path = exec_dir / "stderr.txt"
    script_path = exec_dir / "run_user_code.py"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    _write_bootstrap_script(script_path, code)

    try:
        env = os.environ.copy()
        env["DABENCH_CONTEXT_ROOT"] = resolved_context_root.as_posix()
        env["DABENCH_SCRATCH_ROOT"] = resolved_scratch_root.as_posix()

        with stdout_path.open("w", encoding="utf-8", errors="replace", newline="") as stdout_handle:
            with stderr_path.open("w", encoding="utf-8", errors="replace", newline="") as stderr_handle:
                try:
                    completed = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=resolved_scratch_root,
                        env=env,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        timeout=timeout_seconds,
                        text=True,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    completed = None

        if completed is None:
            return {
                "success": False,
                "context_root": resolved_context_root.as_posix(),
                "scratch_root": resolved_scratch_root.as_posix(),
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
            }

        result: dict[str, Any] = {
            "success": completed.returncode == 0,
            "context_root": resolved_context_root.as_posix(),
            "scratch_root": resolved_scratch_root.as_posix(),
            "output": _read_captured_stream(stdout_path),
            "stderr": _read_captured_stream(stderr_path),
        }
        if completed.returncode != 0:
            result["error"] = f"Python execution failed with exit code {completed.returncode}."
        return result
    finally:
        shutil.rmtree(exec_dir, ignore_errors=True)

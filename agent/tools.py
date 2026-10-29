import pathlib
import os
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent.runtime import ensure_generation_time

PROJECT_ROOT = pathlib.Path.cwd() / "generated_project"
_PROJECT_ROOT = ContextVar("thought2code_project_root", default=None)
_PROJECT_LIMITS = ContextVar("thought2code_project_limits", default=None)


def current_project_root() -> pathlib.Path:
    """Return the request-scoped root, or the historical CLI root."""
    return _PROJECT_ROOT.get() or PROJECT_ROOT


@contextmanager
def use_project_root(
    path: pathlib.Path,
    *,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
    max_files: int | None = None,
):
    """Temporarily direct file tools to an isolated project directory."""
    token = _PROJECT_ROOT.set(pathlib.Path(path).resolve())
    limits_token = _PROJECT_LIMITS.set(
        (max_file_bytes, max_total_bytes, max_files)
        if any(limit is not None for limit in (max_file_bytes, max_total_bytes, max_files))
        else None
    )
    try:
        yield
    finally:
        _PROJECT_LIMITS.reset(limits_token)
        _PROJECT_ROOT.reset(token)


def safe_path_for_project(path: str) -> pathlib.Path:
    project_root = current_project_root().resolve()
    p = (project_root / path).resolve()
    if project_root not in p.parents and project_root != p:
        raise ValueError("Attempt to write outside project root")
    return p


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the specified path within the project root."""
    ensure_generation_time()
    p = safe_path_for_project(path)
    limits = _PROJECT_LIMITS.get()
    if limits is not None:
        max_file_bytes, max_total_bytes, max_files = limits
        content_bytes = len(content.encode("utf-8"))
        if max_file_bytes is not None and content_bytes > max_file_bytes:
            raise ValueError("Generated file exceeds the configured size limit")
        root = current_project_root()
        existing_bytes = 0
        file_count = 0
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_path = pathlib.Path(directory)
            directory_names[:] = [
                name for name in directory_names if not (directory_path / name).is_symlink()
            ]
            for name in file_names:
                candidate = directory_path / name
                try:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    file_count += 1
                    if candidate != p:
                        existing_bytes += candidate.stat().st_size
                except OSError:
                    raise ValueError("Could not safely inspect generated output") from None
        if max_files is not None and file_count + (0 if p.is_file() else 1) > max_files:
            raise ValueError("Generated project exceeds the configured file limit")
        if max_total_bytes is not None and existing_bytes + content_bytes > max_total_bytes:
            raise ValueError("Generated project exceeds the configured size limit")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE:{p}"


@tool
def read_file(path: str) -> str:
    """Reads content from a file at the specified path within the project root."""
    ensure_generation_time()
    p = safe_path_for_project(path)
    if not p.exists():
        return ""
    limits = _PROJECT_LIMITS.get()
    if limits is not None:
        max_file_bytes = limits[0]
        if max_file_bytes is not None and p.stat().st_size > max_file_bytes:
            raise ValueError("Generated file exceeds the configured size limit")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read((max_file_bytes + 1) if limits and max_file_bytes else -1)
    if limits and max_file_bytes is not None and len(content.encode("utf-8")) > max_file_bytes:
        raise ValueError("Generated file exceeds the configured size limit")
    return content


@tool
def get_current_directory() -> str:
    """Returns the current working directory."""
    return str(current_project_root())


class ListGeneratedFilesInput(BaseModel):
    directory: str = Field(default=".")


@tool(args_schema=ListGeneratedFilesInput)
def list_generated_files(directory: str = ".") -> str:
    """Lists all files in the specified directory within the project root."""
    ensure_generation_time()
    p = safe_path_for_project(directory)
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    project_root = current_project_root()
    files = [str(f.relative_to(project_root)) for f in p.glob("**/*") if f.is_file()]
    return "\n".join(files) if files else "No files found."

def init_project_root():
    project_root = current_project_root()
    project_root.mkdir(parents=True, exist_ok=True)
    return str(project_root)

import pathlib
import logging
import os
import re
import stat
import tempfile
import traceback
import unicodedata
import zipfile
from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import RateLimitError
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.background import BackgroundTask

from agent.graph import agent
from agent.tools import use_project_root


PROJECTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "generated_project" / "jobs"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SENSITIVE_ARCHIVE_NAME_PATTERN = re.compile(
    r"(^|[._-])(credential|credentials|secret|secrets|token|tokens|password|passwords|private[_-]?key|api[_-]?key)([._-]|$)"
)
SENSITIVE_CONTENT_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgsk_[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
)
MAX_VIEWABLE_FILE_BYTES = 1_048_576
MAX_JOB_FILE_COUNT = 1_000
MAX_DOWNLOAD_BYTES = 50 * 1_048_576
ARCHIVE_EXCLUDED_PARTS = {
    ".cache",
    ".conda",
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "__pypackages__",
    "cache",
    "env",
    "node_modules",
    "venv",
}
ARCHIVE_EXCLUDED_NAMES = {
    "auth.json",
    "authorized_keys",
    "credentials",
    "credentials.json",
    "credentials.toml",
    "credentials.yaml",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
    "password.txt",
    "passwords.txt",
    "private_key.txt",
    "private-key.txt",
    "secret.txt",
    "secrets",
    "secrets.json",
    "secrets.yaml",
    "service-account.json",
    "service_account.json",
    "token.json",
    "tokens.json",
}
TEXT_FILE_LANGUAGES = {
    ".c": "c",
    ".cpp": "cpp",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".sh": "shell",
    ".sql": "sql",
    ".svg": "xml",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
logger = logging.getLogger(__name__)
RATE_LIMIT_DETAIL = "The AI provider's usage limit has been reached. Please wait and try again later."
SAFE_EXCEPTION_TYPES = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
LOCAL_DIAGNOSTICS_PRODUCTION_MARKERS = (
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "DYNO",
    "FLY_APP_NAME",
    "K_SERVICE",
    "RENDER",
    "VERCEL",
    "WEBSITE_SITE_NAME",
)


def _failed_graph_node(error: Exception) -> str:
    for note in getattr(error, "__notes__", ()):
        prefix = "LangGraph node: "
        if not isinstance(note, str) or not note.startswith(prefix):
            continue
        node_name = note[len(prefix):]
        if node_name in {"planner", "architect", "coder"}:
            return node_name
    return "unknown"


def _safe_exception_type(error: Exception) -> str:
    """Return only a known exception class name suitable for diagnostics."""
    return (
        type(error).__name__
        if type(error) in SAFE_EXCEPTION_TYPES
        else "OtherError"
    )


def _local_diagnostics_enabled(request: Request) -> bool:
    """Enable stack-location diagnostics only for an explicitly local loopback run."""
    if (
        os.environ.get("THOUGHT2CODE_ENV") != "local"
        or os.environ.get("THOUGHT2CODE_LOCAL_DIAGNOSTICS") != "1"
        or any(os.environ.get(name) for name in LOCAL_DIAGNOSTICS_PRODUCTION_MARKERS)
    ):
        return False

    client = request.client
    server = request.scope.get("server")
    if client is None or server is None:
        return False
    try:
        return (
            ip_address(client.host).is_loopback
            and ip_address(server[0]).is_loopback
        )
    except ValueError:
        return False


def _diagnostic_class_name(error: BaseException) -> str:
    exception_type = type(error)
    module = getattr(exception_type, "__module__", "")
    name = getattr(exception_type, "__qualname__", "")
    safe_module = re.sub(r"[^A-Za-z0-9_.]", "_", module)[:120] or "unknown"
    safe_name = re.sub(r"[^A-Za-z0-9_.<>-]", "_", name)[:120] or "unknown"
    return f"{safe_module}.{safe_name}"


def _diagnostic_exception_chain(error: BaseException) -> str:
    classes: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(classes) < 8 and id(current) not in seen:
        seen.add(id(current))
        classes.append(_diagnostic_class_name(current))
        current = current.__cause__ or current.__context__
    return ">".join(classes)


def _diagnostic_stack_locations(error: BaseException) -> str:
    locations = []
    for frame in traceback.extract_tb(error.__traceback__)[-20:]:
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", pathlib.Path(frame.filename).name)[:120]
        function = re.sub(r"[^A-Za-z0-9_.<>-]", "_", frame.name)[:120]
        locations.append(f"{filename}:{function}:{frame.lineno}")
    return ">".join(locations) or "unavailable"

app = FastAPI(title="Thought2Code.Ai Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=4000),
    ]


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    project: dict
    generated_files: list[str]


class GeneratedFileInfo(BaseModel):
    path: str
    size_bytes: int
    language: str
    viewable: bool


class JobFilesResponse(BaseModel):
    job_id: str
    files: list[GeneratedFileInfo]


class GeneratedFileResponse(BaseModel):
    job_id: str
    path: str
    language: str
    content: str


def _job_output_dir(job_id: str) -> pathlib.Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid job identifier")

    jobs_root = PROJECTS_ROOT.resolve()
    candidate = PROJECTS_ROOT / job_id
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Completed job not found") from None

    if resolved.parent != jobs_root or not resolved.is_dir():
        raise HTTPException(status_code=404, detail="Completed job not found")

    completed_root = PROJECTS_ROOT / ".completed"
    try:
        resolved_completed_root = completed_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Completed job not found") from None
    if resolved_completed_root.parent != jobs_root:
        raise HTTPException(status_code=404, detail="Completed job not found")
    marker = resolved_completed_root / f"{job_id}.done"
    if marker.is_symlink() or not marker.is_file():
        raise HTTPException(status_code=404, detail="Completed job not found")
    return resolved


def _safe_job_file(job_root: pathlib.Path, requested_path: str) -> pathlib.Path:
    if (
        not requested_path
        or "\x00" in requested_path
        or "\\" in requested_path
        or re.match(r"^[a-zA-Z]:", requested_path)
    ):
        raise HTTPException(status_code=400, detail="Invalid generated file path")

    relative = PurePosixPath(requested_path)
    if relative.is_absolute() or any(part in {"..", "."} for part in relative.parts):
        raise HTTPException(status_code=400, detail="Invalid generated file path")

    candidate = job_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise HTTPException(status_code=400, detail="Invalid generated file path")

    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Generated file not found") from None

    if job_root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid generated file path")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Generated file not found")
    return resolved


def _completed_job_files(job_root: pathlib.Path) -> list[GeneratedFileInfo]:
    result: list[GeneratedFileInfo] = []
    for directory, directory_names, file_names in os.walk(job_root, followlinks=False):
        directory_path = pathlib.Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (directory_path / name).is_symlink()
        ]
        for file_name in file_names:
            file_path = directory_path / file_name
            if file_path.is_symlink():
                continue
            try:
                resolved = file_path.resolve(strict=True)
                metadata = resolved.stat()
            except (FileNotFoundError, OSError, RuntimeError):
                continue
            if job_root not in resolved.parents or not stat.S_ISREG(metadata.st_mode):
                continue
            language = TEXT_FILE_LANGUAGES.get(resolved.suffix.lower(), "binary")
            result.append(
                GeneratedFileInfo(
                    path=resolved.relative_to(job_root).as_posix(),
                    size_bytes=metadata.st_size,
                    language=language,
                    viewable=(
                        language != "binary"
                        and metadata.st_size <= MAX_VIEWABLE_FILE_BYTES
                        and not metadata.st_mode & 0o111
                    ),
                )
            )
            if len(result) > MAX_JOB_FILE_COUNT:
                raise HTTPException(
                    status_code=413,
                    detail="This project contains too many files to browse safely.",
                )
    return sorted(result, key=lambda file: file.path)


def _archive_path_is_allowed(file_info: GeneratedFileInfo) -> bool:
    if (
        "\x00" in file_info.path
        or "\\" in file_info.path
        or ":" in file_info.path
        or re.match(r"^[a-zA-Z]:", file_info.path)
    ):
        return False
    relative = PurePosixPath(file_info.path)
    if (
        not file_info.viewable
        or relative.is_absolute()
        or relative.as_posix() != file_info.path
        or any(part in {".", ".."} or part.startswith(".") for part in relative.parts)
        or any(
            part.lower() in ARCHIVE_EXCLUDED_PARTS
            or part.lower().endswith(("_cache", "-cache"))
            for part in relative.parts
        )
    ):
        return False

    filename = relative.name.lower()
    return (
        filename not in ARCHIVE_EXCLUDED_NAMES
        and not SENSITIVE_ARCHIVE_NAME_PATTERN.search(filename)
        and not filename.startswith(".env")
        and not filename.endswith(".zip")
    )


def _content_looks_sensitive(content: bytes) -> bool:
    return any(pattern.search(content) for pattern in SENSITIVE_CONTENT_PATTERNS)


def _secure_archive_open_supported() -> bool:
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
    )


def _open_job_file_no_follow(job_root: pathlib.Path, requested_path: str):
    file_path = _safe_job_file(job_root, requested_path)
    relative_parts = file_path.relative_to(job_root).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open(job_root, directory_flags)
    try:
        for part in relative_parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(
            relative_parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current_fd,
        )
    finally:
        os.close(current_fd)
    return os.fdopen(file_fd, "rb")


def _archive_chunks(archive_file):
    try:
        while chunk := archive_file.read(64 * 1024):
            yield chunk
    finally:
        archive_file.close()


def _build_job_archive(job_root: pathlib.Path):
    if not _secure_archive_open_supported():
        raise HTTPException(
            status_code=501,
            detail="Safe project downloads are not supported on this platform.",
        )
    archive_file = tempfile.SpooledTemporaryFile(max_size=8 * 1_048_576, mode="w+b")
    total_bytes = 0
    archive_names: set[str] = set()
    try:
        with zipfile.ZipFile(archive_file, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_info in _completed_job_files(job_root):
                if not _archive_path_is_allowed(file_info):
                    continue
                try:
                    with _open_job_file_no_follow(job_root, file_info.path) as generated_file:
                        metadata = os.fstat(generated_file.fileno())
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_mode & 0o111
                            or metadata.st_size > MAX_VIEWABLE_FILE_BYTES
                        ):
                            continue
                        content = generated_file.read(MAX_VIEWABLE_FILE_BYTES + 1)
                except (HTTPException, OSError):
                    # A file that changed after listing is excluded from the archive.
                    continue

                if len(content) > MAX_VIEWABLE_FILE_BYTES:
                    continue
                try:
                    content.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if _content_looks_sensitive(content):
                    continue

                archive_name = unicodedata.normalize("NFC", file_info.path).casefold()
                if archive_name in archive_names:
                    raise HTTPException(
                        status_code=500,
                        detail="This project contains duplicate file paths.",
                    )

                total_bytes += len(content)
                if total_bytes > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="This project is too large to download safely.",
                    )
                archive_names.add(archive_name)
                archive.writestr(file_info.path, content)
        archive_file.seek(0)
        return archive_file
    except Exception:
        archive_file.close()
        raise


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs/{job_id}/files", response_model=JobFilesResponse)
def list_job_files(job_id: str) -> JobFilesResponse:
    job_root = _job_output_dir(job_id)
    return JobFilesResponse(job_id=job_id, files=_completed_job_files(job_root))


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str) -> StreamingResponse:
    job_root = _job_output_dir(job_id)
    archive_file = _build_job_archive(job_root)
    return StreamingResponse(
        _archive_chunks(archive_file),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="thought2code-ai-{job_id}.zip"'
        },
        background=BackgroundTask(archive_file.close),
    )


@app.get("/jobs/{job_id}/files/{file_path:path}", response_model=GeneratedFileResponse)
def get_job_file(job_id: str, file_path: str) -> GeneratedFileResponse:
    job_root = _job_output_dir(job_id)
    file_path_on_disk = _safe_job_file(job_root, file_path)
    language = TEXT_FILE_LANGUAGES.get(file_path_on_disk.suffix.lower())
    if language is None:
        raise HTTPException(status_code=400, detail="This file type is not available for viewing")

    try:
        with file_path_on_disk.open("rb") as generated_file:
            metadata = pathlib.Path(file_path_on_disk).stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise HTTPException(status_code=400, detail="This file is not available for viewing")
            if metadata.st_size > MAX_VIEWABLE_FILE_BYTES:
                raise HTTPException(status_code=413, detail="Generated file exceeds the viewing limit")
            if metadata.st_mode & 0o111:
                raise HTTPException(status_code=400, detail="Executable files are not available for viewing")
            content = generated_file.read(MAX_VIEWABLE_FILE_BYTES + 1)
    except OSError:
        raise HTTPException(status_code=404, detail="Generated file not found") from None

    if len(content) > MAX_VIEWABLE_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Generated file exceeds the viewing limit")
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="This file is not valid UTF-8 text") from None

    return GeneratedFileResponse(
        job_id=job_id,
        path=file_path_on_disk.relative_to(job_root).as_posix(),
        language=language,
        content=text_content,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    """Run generation synchronously in an isolated, server-owned directory."""
    job_id = uuid4().hex
    output_dir = (PROJECTS_ROOT / job_id).resolve()
    if PROJECTS_ROOT.resolve() not in output_dir.parents:
        raise HTTPException(status_code=500, detail="Could not initialize generation job")

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            with use_project_root(output_dir):
                result = agent.invoke(
                    {"user_prompt": request.project_description},
                    {"recursion_limit": 100},
                )
        except RateLimitError:
            raise HTTPException(status_code=429, detail=RATE_LIMIT_DETAIL) from None

        if not isinstance(result, Mapping):
            raise RuntimeError("Graph returned an unexpected state type")

        plan = result.get("plan")
        if plan is None:
            raise RuntimeError("Graph completed without a project plan")

        files = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and output_dir in path.resolve().parents
        )
        response = GenerateResponse(
            job_id=job_id,
            status="completed",
            project={
                "name": plan.name,
                "description": plan.description,
                "techstack": plan.techstack,
                "features": plan.features,
            },
            generated_files=files,
        )
        completed_root_path = PROJECTS_ROOT / ".completed"
        completed_root_path.mkdir(parents=True, exist_ok=True)
        completed_root = completed_root_path.resolve()
        if completed_root.parent != PROJECTS_ROOT.resolve():
            raise RuntimeError("Completed-job marker directory is outside the jobs root")
        (completed_root / f"{job_id}.done").touch(exist_ok=False)
        return response
    except HTTPException:
        raise
    except Exception as error:
        # Log only allowlisted metadata; exception messages and state can contain secrets.
        logger.error(
            "Generation failed job_id=%s node=%s exception_type=%s",
            job_id,
            _failed_graph_node(error),
            _safe_exception_type(error),
        )
        if _local_diagnostics_enabled(http_request):
            logger.error(
                "Local generation diagnostic job_id=%s node=%s exception=%s frames=%s chain=%s",
                job_id,
                _failed_graph_node(error),
                _diagnostic_class_name(error),
                _diagnostic_stack_locations(error),
                _diagnostic_exception_chain(error),
            )
        raise HTTPException(
            status_code=500,
            detail="Generation failed. Check the backend configuration and try again.",
        ) from None

import pathlib
import logging
import os
import json
import math
import re
import stat
import tempfile
import threading
import traceback
import unicodedata
import zipfile
from datetime import datetime, timezone
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from ipaddress import ip_address
from pathlib import PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from groq import APIStatusError, RateLimitError
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.background import BackgroundTask

from agent.graph import agent
from agent.tools import safe_path_for_project, use_project_root
from agent.repair import repair_generated_files
from api.output_validation import (
    MAX_VALIDATION_FILE_BYTES,
    ValidationReport,
    _resolve_local_reference,
    validate_generated_output,
)
from agent.runtime import (
    ORPHAN_CLEANUP_MAX_ENTRIES,
    ORPHAN_CLEANUP_MAX_JOBS,
    ORPHAN_CLEANUP_MAX_TREE_ENTRIES,
    ORPHAN_JOB_RETENTION_SECONDS,
    MAX_GRAPH_RECURSION_STEPS,
    MAX_CODER_AGENT_RECURSION_STEPS,
    GENERATION_TIMEOUT_SECONDS,
    MAX_REQUEST_BODY_BYTES,
    MAX_CONCURRENT_GENERATIONS,
    MAX_GENERATED_FILE_BYTES,
    MAX_GENERATED_PROJECT_BYTES,
    MAX_GENERATED_PROJECT_FILES,
    MAX_PROMPT_CHARACTERS,
    TRANSIENT_PROVIDER_ERRORS,
    ContextBudgetExceeded,
    CoderAgentStepLimitExceeded,
    CoderReActRecursionLimitExceeded,
    GraphRecursionLimitExceeded,
    GenerationDeadlineExceeded,
    StructuredOutputIncomplete,
    ensure_generation_time,
    emit_local_diagnostic,
    graph_execution_trace,
    generation_deadline,
    local_diagnostic_scope,
)
from api.job_cleanup import active_job, cleanup_orphan_jobs


PROJECTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "generated_project" / "jobs"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SENSITIVE_ARCHIVE_NAME_PATTERN = re.compile(
    r"(^|[._-])(credential|credentials|secret|secrets|token|tokens|password|passwords|private[_-]?key|api[_-]?key)([._-]|$)"
)
SENSITIVE_CONTENT_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgsk_[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)
MAX_VIEWABLE_FILE_BYTES = 1_048_576
MAX_PREVIEW_ASSET_BYTES = 2 * 1_048_576
MAX_JOB_FILE_COUNT = 1_000
MAX_DOWNLOAD_BYTES = 50 * 1_048_576
MAX_REPAIR_FINDINGS = 12
MAX_REPAIR_FILES = 8
MAX_REPAIR_CONTEXT_BYTES = 256 * 1_024
MAX_REPAIR_OUTPUT_BYTES = 256 * 1_024
REPAIRABLE_FINDINGS = {
    "required_reference_missing",
    "unsafe_local_reference",
    "html_unmatched_closing_tag",
    "html_mismatched_tag_nesting",
    "html_unclosed_tag",
    "html_parse_error",
    "css_unbalanced_delimiter",
    "css_unclosed_construct",
    "javascript_syntax_invalid",
}
REPAIR_DESCRIPTIONS = {
    "required_reference_missing": "A required local file reference or static module import is missing.",
    "unsafe_local_reference": "A local reference uses an unsafe path; change it to a safe path within the project.",
    "html_unmatched_closing_tag": "HTML contains an unmatched closing tag.",
    "html_mismatched_tag_nesting": "HTML tags have mismatched nesting.",
    "html_unclosed_tag": "HTML contains an unclosed required element.",
    "html_parse_error": "HTML could not be parsed safely.",
    "css_unbalanced_delimiter": "CSS contains an unmatched delimiter.",
    "css_unclosed_construct": "CSS contains an unclosed string, comment, or delimiter.",
    "javascript_syntax_invalid": "JavaScript syntax validation failed.",
}
IMPLEMENTATION_FILE_SUFFIXES = {
    ".astro",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".htm",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".mjs",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sass",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}
ARCHIVE_EXCLUDED_PARTS = {
    ".aws",
    ".azure",
    ".cache",
    ".conda",
    ".gcloud",
    ".git",
    ".gnupg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".ssh",
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
    "credentials.ini",
    "credentials.toml",
    "credentials.yaml",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    "id_rsa",
    "known_hosts",
    "password.txt",
    "passwords.txt",
    "private_key.txt",
    "private-key.txt",
    "secret.txt",
    "secrets",
    "secrets.json",
    "secrets.ini",
    "secrets.yaml",
    "service-account.json",
    "service-account-key.json",
    "firebase-adminsdk.json",
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
PREVIEW_ASSET_MIME_TYPES = {
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".otf": "font/otf",
    ".png": "image/png",
    ".ttf": "font/ttf",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}
logger = logging.getLogger(__name__)
_GENERATION_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_GENERATIONS)
RATE_LIMIT_DETAIL = "The AI provider has temporarily limited this request. Please wait and try again."
SAFE_EXCEPTION_TYPES = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    StructuredOutputIncomplete,
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
    for current in _walk_exception_chain(error):
        for note in getattr(current, "__notes__", ()):
            prefix = "LangGraph node: "
            if not isinstance(note, str) or not note.startswith(prefix):
                continue
            node_name = note[len(prefix):]
            if node_name in {"planner", "architect", "coder"}:
                return node_name
    return "unknown"


def _walk_exception_chain(error: BaseException):
    """Visit a bounded chain/group without reading or formatting messages."""
    pending = [error]
    seen: set[int] = set()
    while pending and len(seen) < 16:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        linked = [current.__cause__, current.__context__]
        children = getattr(current, "exceptions", ())
        if isinstance(children, tuple):
            linked.extend(children[:16])
        pending.extend(
            candidate
            for candidate in linked
            if isinstance(candidate, BaseException) and id(candidate) not in seen
        )


def _chained_exception(error: BaseException, error_types: tuple[type, ...]):
    return next(
        (current for current in _walk_exception_chain(error) if isinstance(current, error_types)),
        None,
    )


def _safe_exception_type(error: Exception) -> str:
    """Return only a known exception class name suitable for diagnostics."""
    return (
        type(error).__name__
        if type(error) in SAFE_EXCEPTION_TYPES
        else "OtherError"
    )


_SAFE_PROVIDER_ERROR_TYPES = {
    "api_error",
    "authentication_error",
    "insufficient_quota",
    "invalid_request_error",
    "permission_error",
    "rate_limit_error",
    "server_error",
}


def _bounded_seconds(value: float | None) -> int | None:
    if value is None:
        return None
    if not math.isfinite(value) or value >= 86_400:
        return 86_400
    if value <= 0:
        return 0
    return int(round(value))


def _parse_duration_header(value: object) -> int | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    raw = value.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return _bounded_seconds(float(raw))

    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)(ms|s|m|h)", raw))
    if not matches or "".join(match.group(0) for match in matches) != raw:
        return None
    return _bounded_seconds(
        sum(float(match.group(1)) * units[match.group(2)] for match in matches)
    )


def _retry_after_seconds(headers) -> int | None:
    value = headers.get("retry-after") if headers is not None else None
    if not isinstance(value, str) or len(value) > 128:
        return None
    delay = _parse_duration_header(value)
    if delay is not None:
        return delay
    try:
        reset_at = parsedate_to_datetime(value)
        if reset_at.tzinfo is None:
            reset_at = reset_at.replace(tzinfo=timezone.utc)
        return _bounded_seconds((reset_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _provider_headers(error: APIStatusError) -> dict[str, int | None]:
    headers = getattr(getattr(error, "response", None), "headers", None)

    def integer_header(name: str) -> int | None:
        value = headers.get(name) if headers is not None else None
        if not isinstance(value, str) or len(value) > 12 or not value.isdigit():
            return None
        return min(int(value), 2_147_483_647)

    request_reset = _parse_duration_header(
        headers.get("x-ratelimit-reset-requests") if headers is not None else None
    )
    token_reset = _parse_duration_header(
        headers.get("x-ratelimit-reset-tokens") if headers is not None else None
    )
    retry_after = _retry_after_seconds(headers)
    return {
        "retry_after_seconds": retry_after,
        "reset_requests_seconds": request_reset,
        "reset_tokens_seconds": token_reset,
        "remaining_requests": integer_header("x-ratelimit-remaining-requests"),
        "remaining_tokens": integer_header("x-ratelimit-remaining-tokens"),
    }


def _provider_error_type(error: APIStatusError) -> str:
    body = getattr(error, "body", None)
    provider_error = body.get("error") if isinstance(body, dict) else None
    candidate = provider_error.get("type") if isinstance(provider_error, dict) else None
    if isinstance(candidate, str) and candidate in _SAFE_PROVIDER_ERROR_TYPES:
        return candidate
    class_name = type(error).__name__
    return class_name if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", class_name) else "ProviderError"


def _raise_provider_http_error(
    error: APIStatusError, job_id: str, *, stage_override: str | None = None
) -> None:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or not 400 <= status <= 599:
        status = 502
    if status == 429 or isinstance(error, RateLimitError):
        category = "rate_limit"
        message = RATE_LIMIT_DETAIL
    elif status in {401, 403}:
        category = "authentication"
        message = "The AI provider rejected backend authentication. Check the server configuration."
    elif status == 400:
        category = "invalid_request"
        message = "The AI provider rejected this generation request. Review it and try again."
    elif status >= 500:
        category = "provider_unavailable"
        message = "The AI provider is temporarily unavailable. Please try again shortly."
    else:
        category = "provider_error"
        message = f"The AI provider could not complete the request (HTTP {status}). Please try again."

    stage = stage_override
    if stage is None:
        stage = {
            "planner": "Planner",
            "architect": "Architect",
            "coder": "Coder",
        }.get(_failed_graph_node(error), "unknown")
    metadata = _provider_headers(error)
    error_type = _provider_error_type(error)
    logger.error(
        "AI provider request failed job_id=%s stage=%s http_status=%s category=%s "
        "error_type=%s retry_after_seconds=%s reset_requests_seconds=%s "
        "reset_tokens_seconds=%s remaining_requests=%s remaining_tokens=%s",
        job_id,
        stage,
        status,
        category,
        error_type,
        metadata["retry_after_seconds"],
        metadata["reset_requests_seconds"],
        metadata["reset_tokens_seconds"],
        metadata["remaining_requests"],
        metadata["remaining_tokens"],
    )
    detail = {"message": message, "error_type": error_type, "category": category}
    if metadata["retry_after_seconds"] is not None:
        detail["retry_after_seconds"] = metadata["retry_after_seconds"]
    if status == 429:
        reset_timing = {
            key: metadata[key]
            for key in ("reset_requests_seconds", "reset_tokens_seconds")
            if metadata[key] is not None
        }
        if reset_timing:
            detail["rate_limit_reset_seconds"] = reset_timing
    raise HTTPException(status_code=status, detail=detail) from None


def _log_transient_provider_failure(
    error: Exception, job_id: str, *, stage_override: str | None = None
) -> None:
    stage = stage_override or {
        "planner": "Planner",
        "architect": "Architect",
        "coder": "Coder",
    }.get(_failed_graph_node(error), "unknown")
    error_type = type(error).__name__
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error_type):
        error_type = "ProviderError"
    category = "provider_timeout" if error_type == "APITimeoutError" else "provider_connection"
    logger.error(
        "AI provider request failed job_id=%s stage=%s http_status=unavailable "
        "category=%s error_type=%s retry_after_seconds=unavailable "
        "reset_requests_seconds=unavailable reset_tokens_seconds=unavailable "
        "remaining_requests=unavailable remaining_tokens=unavailable",
        job_id,
        stage,
        category,
        error_type,
    )


def _raise_workflow_limit_error(
    job_id: str,
    *,
    stage: str,
    category: str,
    error_type: str,
    configured_limit: int,
    configured_limit_unit: str,
    observed_count: int,
    count_kind: str,
) -> None:
    """Return a safe, actionable response for a known bounded graph limit."""
    safe_stages = {"Planner", "Architect", "Coder"}
    if stage not in safe_stages:
        stage = "unknown"
    safe_error_types = {
        "graph_recursion_limit",
        "coder_step_limit",
        "coder_react_recursion_limit",
    }
    if error_type not in safe_error_types:
        error_type = "graph_recursion_limit"
    safe_categories = {"workflow_limit"}
    if category not in safe_categories:
        category = "workflow_limit"
    if not isinstance(configured_limit, int) or not 1 <= configured_limit <= 1_000:
        configured_limit = 0
    if not isinstance(observed_count, int) or not 0 <= observed_count <= 1_000:
        observed_count = 0
    if count_kind not in {"node_steps", "model_iterations"}:
        count_kind = "node_steps"
    if configured_limit_unit not in {"graph_supersteps", "model_iterations"}:
        configured_limit_unit = "graph_supersteps"
    logger.error(
        "Generation workflow limit reached job_id=%s stage=%s category=%s error_type=%s "
        "configured_limit=%s configured_limit_unit=%s observed_%s=%s",
        job_id,
        stage,
        category,
        error_type,
        configured_limit,
        configured_limit_unit,
        count_kind,
        observed_count,
    )
    raise HTTPException(
        status_code=422,
        detail={
            "message": (
                "Generation reached a bounded workflow step limit. "
                "Try a smaller or simpler project description."
            ),
            "category": category,
            "error_type": error_type,
            "stage": stage,
            "configured_limit": configured_limit,
            "configured_limit_unit": configured_limit_unit,
            f"observed_{count_kind}": observed_count,
        },
    ) from None


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
    return ">".join(
        _diagnostic_class_name(current)
        for index, current in enumerate(_walk_exception_chain(error))
        if index < 8
    )


def _diagnostic_stack_locations(error: BaseException) -> str:
    locations = []
    for frame in traceback.extract_tb(error.__traceback__)[-20:]:
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", pathlib.Path(frame.filename).name)[:120]
        function = re.sub(r"[^A-Za-z0-9_.<>-]", "_", frame.name)[:120]
        locations.append(f"{filename}:{function}:{frame.lineno}")
    return ">".join(locations) or "unavailable"


class RequestBodyLimitMiddleware:
    """Count ASGI request chunks without copying or buffering the body."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.response_body = b'{"detail":"Request body exceeds the configured size limit."}'

    async def _send_too_large(self, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(self.response_body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": self.response_body,
                "more_body": False,
            }
        )

    def _declared_body_is_too_large(self, headers) -> bool:
        for name, value in headers:
            if name.lower() != b"content-length":
                continue
            value = value.strip()
            if not value or not value.isdigit():
                continue
            normalized = value.lstrip(b"0") or b"0"
            maximum = str(self.max_body_bytes).encode("ascii")
            if len(normalized) > len(maximum):
                return True
            if int(normalized) > self.max_body_bytes:
                return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self._declared_body_is_too_large(scope.get("headers", ())):
            await self._send_too_large(send)
            return

        received_bytes = 0
        too_large = False

        async def limited_receive():
            nonlocal received_bytes, too_large
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if received_bytes + len(body) > self.max_body_bytes:
                    too_large = True
                    # Stop the downstream parser immediately; do not pass or
                    # buffer the over-limit chunk.
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
                received_bytes += len(body)
            return message

        async def limited_send(message):
            if not too_large:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not too_large:
                raise

        if too_large:
            await self._send_too_large(send)


app = FastAPI(title="Thought2Code.Ai Local API", version="0.1.0")
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
)
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
        StringConstraints(
            strip_whitespace=True,
            min_length=10,
            max_length=MAX_PROMPT_CHARACTERS,
        ),
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


def _validated_output_path(output_dir: pathlib.Path, value: object) -> pathlib.Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or re.match(r"^[a-zA-Z]:", value)
    ):
        raise ValueError("Generation produced an invalid output path")

    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Generation produced an invalid output path")

    candidate = output_dir
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Generation produced an unsafe output path")

    try:
        resolved = safe_path_for_project(value)
    except (OSError, ValueError, RuntimeError):
        raise ValueError("Generation produced an unsafe output path") from None

    if output_dir not in resolved.parents:
        raise ValueError("Generation produced an unsafe output path")
    return resolved


def _non_whitespace_file(path: pathlib.Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as generated_file:
            while chunk := generated_file.read(64 * 1024):
                if chunk.strip():
                    return True
    except (OSError, UnicodeError):
        return False
    return False


def _validate_generation_completion(
    result: Mapping, output_dir: pathlib.Path
) -> tuple[object, list[str]]:
    plan = result.get("plan")
    if plan is None:
        raise ValueError("Graph completed without a project plan")

    task_plan = result.get("task_plan")
    steps = getattr(task_plan, "implementation_steps", None)
    if not isinstance(steps, list) or not steps:
        raise ValueError("Generation completed without implementation steps")

    plan_files = getattr(plan, "files", None)
    if not isinstance(plan_files, list) or not plan_files:
        raise ValueError("Generation plan contains no required output files")

    with use_project_root(output_dir):
        planned_paths = [
            _validated_output_path(output_dir, getattr(file, "path", None))
            for file in plan_files
        ]
        task_paths = []
        for step in steps:
            task_description = getattr(step, "task_description", None)
            if not isinstance(task_description, str) or not task_description.strip():
                raise ValueError("Generation contains an invalid implementation step")
            task_paths.append(
                _validated_output_path(output_dir, getattr(step, "filepath", None))
            )

        current_coder_state = result.get("coder_state")
        completed_index = getattr(current_coder_state, "current_step_idx", None)
        if (
            result.get("status") != "DONE"
            or completed_index != len(steps)
            or getattr(getattr(current_coder_state, "task_plan", None), "implementation_steps", None)
            != steps
        ):
            raise ValueError("Generation graph did not finish all implementation steps")

        if len({path.relative_to(output_dir) for path in planned_paths}) != len(
            planned_paths
        ):
            raise ValueError("Generation plan contains duplicate output paths")

        required_paths = planned_paths + [
            path for path in task_paths if path not in planned_paths
        ]
        nonempty_paths = set()
        for required_path in required_paths:
            if not required_path.is_file() or not _non_whitespace_file(required_path):
                raise ValueError("A required planned output file is missing or empty")
            nonempty_paths.add(required_path)

        if not any(
            path.suffix.lower() in IMPLEMENTATION_FILE_SUFFIXES
            for path in nonempty_paths
        ):
            raise ValueError("Generation produced no usable implementation file")

    files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and output_dir in path.resolve().parents
    )
    if not files:
        raise ValueError("Generation produced no usable output files")
    total_bytes = 0
    if len(files) > MAX_GENERATED_PROJECT_FILES:
        raise ValueError("Generation exceeded the project file limit")
    for relative_path in files:
        generated_path = output_dir / relative_path
        metadata = generated_path.lstat()
        if metadata.st_size > MAX_GENERATED_FILE_BYTES:
            raise ValueError("Generation produced an oversized file")
        total_bytes += metadata.st_size
        if total_bytes > MAX_GENERATED_PROJECT_BYTES:
            raise ValueError("Generation exceeded the project size limit")
    return plan, files


def _build_repair_context(
    report: ValidationReport, output_dir: pathlib.Path
) -> tuple[dict, set[str]] | None:
    if not report.errors or any(
        finding.code not in REPAIRABLE_FINDINGS for finding in report.errors
    ):
        return None
    if len(report.errors) > MAX_REPAIR_FINDINGS:
        raise ContextBudgetExceeded(
            "Validation feedback exceeds the configured repair context limit. "
            "Please simplify the project request."
        )

    allowed_paths: set[str] = set()
    file_contents: dict[str, str | None] = {}

    def include_file(relative_path: str, *, may_be_missing: bool) -> bool:
        try:
            path = _validated_output_path(output_dir, relative_path)
            canonical = path.relative_to(output_dir).as_posix()
            if canonical in file_contents:
                return True
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                if not may_be_missing:
                    return False
                file_contents[canonical] = None
                allowed_paths.add(canonical)
                return len(file_contents) <= MAX_REPAIR_FILES
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > MAX_VALIDATION_FILE_BYTES
            ):
                return False
            with path.open("rb") as source_file:
                content = source_file.read(MAX_VALIDATION_FILE_BYTES + 1)
            if len(content) > MAX_VALIDATION_FILE_BYTES:
                return False
            text = content.decode("utf-8")
            file_contents[canonical] = text
            allowed_paths.add(canonical)
            return len(file_contents) <= MAX_REPAIR_FILES
        except (OSError, UnicodeError, ValueError):
            return False

    repair_findings = []
    with use_project_root(output_dir):
        for finding in report.errors:
            if not finding.path or not include_file(finding.path, may_be_missing=False):
                return None
            item = {
                "code": finding.code,
                "description": REPAIR_DESCRIPTIONS[finding.code],
                "file": finding.path,
            }
            if finding.reference is not None:
                item["reference"] = finding.reference
                if finding.code == "required_reference_missing":
                    source_path = _validated_output_path(output_dir, finding.path)
                    target, problem = _resolve_local_reference(
                        output_dir, source_path, finding.reference
                    )
                    if problem or target is None:
                        return None
                    target_path = target.relative_to(output_dir).as_posix()
                    if not include_file(target_path, may_be_missing=True):
                        return None
            repair_findings.append(item)

    repair_context = {
        "findings": repair_findings,
        "files": [
            {"path": path, "content": content}
            for path, content in sorted(file_contents.items())
        ],
    }
    if len(json.dumps(repair_context, ensure_ascii=False).encode("utf-8")) > MAX_REPAIR_CONTEXT_BYTES:
        raise ContextBudgetExceeded(
            "Repair input exceeds the configured context limit. "
            "Please simplify the project request."
        )
    return repair_context, allowed_paths


def _apply_repair_response(
    response: object,
    allowed_paths: set[str],
    output_dir: pathlib.Path,
) -> None:
    response_files = getattr(response, "files", None)
    if (
        not isinstance(response_files, list)
        or not response_files
        or len(response_files) > MAX_REPAIR_FILES
        or len(response_files) > len(allowed_paths)
    ):
        raise ValueError("Repair response exceeded its file limit")

    staged: list[tuple[pathlib.Path, str]] = []
    returned_paths: set[str] = set()
    output_bytes = 0
    with use_project_root(output_dir):
        for item in response_files:
            relative_path = getattr(item, "path", None)
            content = getattr(item, "content", None)
            if not isinstance(relative_path, str) or not isinstance(content, str):
                raise ValueError("Repair response contained an invalid file")
            if relative_path not in allowed_paths or relative_path in returned_paths:
                raise ValueError("Repair response contained an unauthorized path")
            path = _validated_output_path(output_dir, relative_path)
            encoded_content = content.encode("utf-8")
            if len(encoded_content) > MAX_VALIDATION_FILE_BYTES:
                raise ValueError("Repair response exceeded its per-file limit")
            output_bytes += len(encoded_content)
            if output_bytes > MAX_REPAIR_OUTPUT_BYTES:
                raise ValueError("Repair response exceeded its output limit")
            returned_paths.add(relative_path)
            staged.append((path, content))

        for path, content in staged:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def _job_output_dir(job_id: str) -> pathlib.Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid job identifier")

    jobs_root = PROJECTS_ROOT.resolve()
    candidate = jobs_root / job_id
    try:
        if _secure_archive_open_supported():
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            root_fd = os.open(jobs_root, directory_flags)
            job_fd = completed_root_fd = marker_fd = None
            try:
                # Open each component relative to a pinned parent. O_NOFOLLOW
                # rejects job/.completed/marker symlinks at the point of use.
                job_fd = os.open(job_id, directory_flags, dir_fd=root_fd)
                if not stat.S_ISDIR(os.fstat(job_fd).st_mode):
                    raise OSError("Invalid job directory")
                completed_root_fd = os.open(
                    ".completed", directory_flags, dir_fd=root_fd
                )
                if not stat.S_ISDIR(os.fstat(completed_root_fd).st_mode):
                    raise OSError("Invalid completion marker directory")
                marker_fd = os.open(
                    f"{job_id}.done",
                    os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=completed_root_fd,
                )
                if not stat.S_ISREG(os.fstat(marker_fd).st_mode):
                    raise OSError("Invalid completion marker")
            finally:
                for descriptor in (marker_fd, completed_root_fd, job_fd, root_fd):
                    if descriptor is not None:
                        os.close(descriptor)
        else:
            # Portable fallback: reject symlinks by inspecting directory entries
            # without following them. This cannot close a concurrent rename race.
            job_metadata = candidate.lstat()
            completed_root = jobs_root / ".completed"
            completed_metadata = completed_root.lstat()
            marker_metadata = (completed_root / f"{job_id}.done").lstat()
            if (
                not stat.S_ISDIR(job_metadata.st_mode)
                or not stat.S_ISDIR(completed_metadata.st_mode)
                or not stat.S_ISREG(marker_metadata.st_mode)
                or candidate.resolve(strict=True).parent != jobs_root
                or completed_root.resolve(strict=True).parent != jobs_root
            ):
                raise OSError("Invalid completed job paths")
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Completed job not found") from None

    return candidate


def _create_completion_marker(job_id: str) -> None:
    """Create a marker only through a real, descriptor-pinned marker directory."""
    if not _secure_archive_open_supported():
        raise OSError("Secure completion marker creation is unavailable")

    jobs_root = PROJECTS_ROOT.resolve()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(jobs_root, directory_flags)
    completed_root_fd = marker_fd = None
    try:
        try:
            os.mkdir(".completed", mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        completed_root_fd = os.open(".completed", directory_flags, dir_fd=root_fd)
        if not stat.S_ISDIR(os.fstat(completed_root_fd).st_mode):
            raise OSError("Invalid completion marker directory")
        marker_fd = os.open(
            f"{job_id}.done",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=completed_root_fd,
        )
        if not stat.S_ISREG(os.fstat(marker_fd).st_mode):
            raise OSError("Invalid completion marker")
    finally:
        for descriptor in (marker_fd, completed_root_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)


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
    configured_root = PROJECTS_ROOT.resolve()
    if job_root.parent != configured_root or not JOB_ID_PATTERN.fullmatch(job_root.name):
        raise HTTPException(status_code=404, detail="Completed job not found")

    if _secure_archive_open_supported():
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(configured_root, directory_flags)
            try:
                job_fd = os.open(job_root.name, directory_flags, dir_fd=root_fd)
            finally:
                os.close(root_fd)
        except OSError:
            raise HTTPException(status_code=404, detail="Completed job not found") from None

        def visit_directory(directory_fd: int, prefix: str = "") -> None:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        try:
                            child_fd = os.open(
                                entry.name, directory_flags, dir_fd=directory_fd
                            )
                        except OSError:
                            continue
                        try:
                            if stat.S_ISDIR(os.fstat(child_fd).st_mode):
                                visit_directory(child_fd, relative_path)
                        finally:
                            os.close(child_fd)
                    elif stat.S_ISREG(metadata.st_mode):
                        language = TEXT_FILE_LANGUAGES.get(
                            pathlib.PurePosixPath(entry.name).suffix.lower(), "binary"
                        )
                        result.append(
                            GeneratedFileInfo(
                                path=relative_path,
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

        try:
            visit_directory(job_fd)
        finally:
            os.close(job_fd)
    else:
        try:
            if not stat.S_ISDIR(job_root.lstat().st_mode):
                raise HTTPException(status_code=404, detail="Completed job not found")
        except OSError:
            raise HTTPException(status_code=404, detail="Completed job not found") from None
        # Portable fallback; descriptor-relative no-follow traversal is unavailable.
        for directory, directory_names, file_names in os.walk(job_root, followlinks=False):
            directory_path = pathlib.Path(directory)
            directory_names[:] = [
                name for name in directory_names if not (directory_path / name).is_symlink()
            ]
            for file_name in file_names:
                file_path = directory_path / file_name
                try:
                    metadata = file_path.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                language = TEXT_FILE_LANGUAGES.get(file_path.suffix.lower(), "binary")
                result.append(
                    GeneratedFileInfo(
                        path=file_path.relative_to(job_root).as_posix(),
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
    try:
        relative_parts = file_path.relative_to(job_root).parts
    except ValueError:
        raise HTTPException(status_code=404, detail="Generated file not found") from None
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    configured_root = PROJECTS_ROOT.resolve()
    if job_root.parent != configured_root or not JOB_ID_PATTERN.fullmatch(job_root.name):
        raise HTTPException(status_code=404, detail="Generated file not found")

    # Anchor traversal at the configured jobs root, then open the job directory
    # and every child relative to pinned directory descriptors.
    root_fd = os.open(configured_root, directory_flags)
    try:
        current_fd = os.open(job_root.name, directory_flags, dir_fd=root_fd)
    finally:
        os.close(root_fd)

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

    if not _secure_archive_open_supported():
        raise HTTPException(
            status_code=501,
            detail="Safe project file viewing is not supported on this platform.",
        )

    try:
        with _open_job_file_no_follow(job_root, file_path) as generated_file:
            metadata = os.fstat(generated_file.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise HTTPException(status_code=400, detail="This file is not available for viewing")
            if metadata.st_size > MAX_VIEWABLE_FILE_BYTES:
                raise HTTPException(status_code=413, detail="Generated file exceeds the viewing limit")
            if metadata.st_mode & 0o111:
                raise HTTPException(status_code=400, detail="Executable files are not available for viewing")
            content = generated_file.read(MAX_VIEWABLE_FILE_BYTES + 1)
    except HTTPException:
        raise
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


@app.get("/jobs/{job_id}/preview-assets/{file_path:path}")
def get_job_preview_asset(job_id: str, file_path: str) -> FastAPIResponse:
    """Serve only small, allowlisted passive assets for the sandboxed preview."""
    job_root = _job_output_dir(job_id)
    validated_path = _safe_job_file(job_root, file_path)
    media_type = PREVIEW_ASSET_MIME_TYPES.get(validated_path.suffix.lower())
    if media_type is None:
        raise HTTPException(
            status_code=400, detail="This asset type is not available for preview"
        )
    if not _secure_archive_open_supported():
        raise HTTPException(
            status_code=501,
            detail="Safe project asset viewing is not supported on this platform.",
        )

    try:
        with _open_job_file_no_follow(job_root, file_path) as asset_file:
            metadata = os.fstat(asset_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111:
                raise HTTPException(
                    status_code=400, detail="This asset is not available for preview"
                )
            if metadata.st_size > MAX_PREVIEW_ASSET_BYTES:
                raise HTTPException(
                    status_code=413, detail="Preview asset exceeds the size limit"
                )
            content = asset_file.read(MAX_PREVIEW_ASSET_BYTES + 1)
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(status_code=404, detail="Preview asset not found") from None

    if len(content) > MAX_PREVIEW_ASSET_BYTES:
        raise HTTPException(status_code=413, detail="Preview asset exceeds the size limit")
    return FastAPIResponse(
        content=content,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _generate_job_for_id(
    request: GenerateRequest, http_request: Request, job_id: str
) -> GenerateResponse:
    """Run generation synchronously in an isolated, server-owned directory."""
    output_dir = (PROJECTS_ROOT / job_id).resolve()
    if PROJECTS_ROOT.resolve() not in output_dir.parents:
        raise HTTPException(status_code=500, detail="Could not initialize generation job")

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        with use_project_root(
            output_dir,
            max_file_bytes=MAX_GENERATED_FILE_BYTES,
            max_total_bytes=MAX_GENERATED_PROJECT_BYTES,
            max_files=MAX_GENERATED_PROJECT_FILES,
        ):
            with local_diagnostic_scope(
                job_id, _local_diagnostics_enabled(http_request)
            ):
                with graph_execution_trace() as trace:
                    try:
                        result = agent.invoke(
                            {"user_prompt": request.project_description},
                            {"recursion_limit": MAX_GRAPH_RECURSION_STEPS},
                        )
                    except GraphRecursionError:
                        emit_local_diagnostic(
                            "workflow_terminated",
                            node=trace.last_node,
                            outcome="failure",
                            termination_reason="outer_graph_limit",
                            configured_limit=MAX_GRAPH_RECURSION_STEPS,
                            configured_limit_unit="graph_supersteps",
                            observed_count=trace.node_steps,
                            count_kind="node_steps",
                        )
                        raise GraphRecursionLimitExceeded(
                            MAX_GRAPH_RECURSION_STEPS, trace
                        ) from None
                emit_local_diagnostic(
                    "workflow_terminated",
                    node="coder",
                    task_index=getattr(
                        result.get("coder_state"), "current_step_idx", None
                    ),
                    outcome=(
                        "success" if result.get("status") == "DONE" else "failure"
                    ),
                    termination_reason=(
                        "generation_complete"
                        if result.get("status") == "DONE"
                        else "unexpected_failure"
                    ),
                )

        if not isinstance(result, Mapping):
            raise RuntimeError("Graph returned an unexpected state type")

        ensure_generation_time()
        plan, files = _validate_generation_completion(result, output_dir)
        ensure_generation_time()
        validation = validate_generated_output(output_dir, files)
        ensure_generation_time()
        if validation.errors:
            repair_request = _build_repair_context(validation, output_dir)
            if repair_request is None:
                logger.error(
                    "Generated output validation failed job_id=%s codes=%s",
                    job_id,
                    ",".join(validation.error_codes),
                )
                raise RuntimeError("Generated output failed validation")

            repair_context, allowed_paths = repair_request
            ensure_generation_time()
            try:
                repair_response = repair_generated_files(repair_context)
            except APIStatusError as error:
                _raise_provider_http_error(error, job_id, stage_override="repair")
            except TRANSIENT_PROVIDER_ERRORS as error:
                _log_transient_provider_failure(
                    error, job_id, stage_override="repair"
                )
                raise
            except Exception as error:
                provider_error = _chained_exception(error, (APIStatusError,))
                if provider_error is not None:
                    _raise_provider_http_error(
                        provider_error, job_id, stage_override="repair"
                    )
                transient_error = _chained_exception(
                    error, TRANSIENT_PROVIDER_ERRORS
                )
                if transient_error is not None:
                    _log_transient_provider_failure(
                        transient_error, job_id, stage_override="repair"
                    )
                    raise transient_error
                raise
            ensure_generation_time()
            _apply_repair_response(repair_response, allowed_paths, output_dir)

            # Re-run the original completion gate and validator exactly once.
            plan, files = _validate_generation_completion(result, output_dir)
            validation = validate_generated_output(output_dir, files)
            if validation.errors:
                logger.error(
                    "Generated output validation remains blocking job_id=%s codes=%s",
                    job_id,
                    ",".join(validation.error_codes),
                )
                raise RuntimeError("Generated output failed validation")
        if validation.warnings:
            logger.warning(
                "Generated output validation warnings job_id=%s codes=%s",
                job_id,
                ",".join(validation.warning_codes),
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
        ensure_generation_time()
        ensure_generation_time()
        _create_completion_marker(job_id)
        return response
    except CoderAgentStepLimitExceeded as error:
        _raise_workflow_limit_error(
            job_id,
            stage="Coder",
            category="workflow_limit",
            error_type="coder_step_limit",
            configured_limit=error.configured_limit,
            configured_limit_unit="model_iterations",
            observed_count=error.observed_model_iterations,
            count_kind="model_iterations",
        )
    except CoderReActRecursionLimitExceeded as error:
        _raise_workflow_limit_error(
            job_id,
            stage="Coder",
            category="workflow_limit",
            error_type="coder_react_recursion_limit",
            configured_limit=error.configured_limit,
            configured_limit_unit="graph_supersteps",
            observed_count=error.observed_model_iterations,
            count_kind="model_iterations",
        )
    except GraphRecursionLimitExceeded as error:
        stage = {
            "planner": "Planner",
            "architect": "Architect",
            "coder": "Coder",
        }.get(error.last_node, "unknown")
        _raise_workflow_limit_error(
            job_id,
            stage=stage,
            category="workflow_limit",
            error_type="graph_recursion_limit",
            configured_limit=error.configured_limit,
            configured_limit_unit="graph_supersteps",
            observed_count=error.observed_node_steps,
            count_kind="node_steps",
        )
    except HTTPException:
        raise
    except APIStatusError as error:
        _raise_provider_http_error(error, job_id)
    except TRANSIENT_PROVIDER_ERRORS as error:
        _log_transient_provider_failure(error, job_id)
        raise
    except (GenerationDeadlineExceeded, ContextBudgetExceeded):
        raise
    except Exception as error:
        provider_error = _chained_exception(error, (APIStatusError,))
        if provider_error is not None:
            stage = {
                "planner": "Planner",
                "architect": "Architect",
                "coder": "Coder",
            }.get(_failed_graph_node(error), "unknown")
            _raise_provider_http_error(
                provider_error, job_id, stage_override=stage
            )
        transient_error = _chained_exception(error, TRANSIENT_PROVIDER_ERRORS)
        if transient_error is not None:
            stage = {
                "planner": "Planner",
                "architect": "Architect",
                "coder": "Coder",
            }.get(_failed_graph_node(error), "unknown")
            _log_transient_provider_failure(
                transient_error, job_id, stage_override=stage
            )
            raise transient_error
        # Log only allowlisted metadata; exception messages and state can contain secrets.
        logger.error(
            "Generation failed job_id=%s node=%s exception_type=%s "
            "exception_class=%s chained_exception_classes=%s",
            job_id,
            _failed_graph_node(error),
            _safe_exception_type(error),
            _diagnostic_class_name(error),
            _diagnostic_exception_chain(error),
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


def _generate_job(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    job_id = uuid4().hex
    try:
        with active_job(PROJECTS_ROOT, job_id):
            return _generate_job_for_id(request, http_request, job_id)
    except HTTPException:
        raise
    except (GenerationDeadlineExceeded, ContextBudgetExceeded, *TRANSIENT_PROVIDER_ERRORS):
        raise
    except Exception as error:
        logger.error(
            "Generation job setup failed job_id=%s exception_type=%s",
            job_id,
            _safe_exception_type(error),
        )
        raise HTTPException(
            status_code=500,
            detail="Generation failed. Check the backend configuration and try again.",
        ) from None


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    """Bound local synchronous generation without claiming hard cancellation."""
    if not _GENERATION_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The generation service is busy. Please try again shortly.",
        )
    try:
        cleanup = cleanup_orphan_jobs(
            PROJECTS_ROOT,
            retention_seconds=ORPHAN_JOB_RETENTION_SECONDS,
            max_jobs=ORPHAN_CLEANUP_MAX_JOBS,
            max_entries=ORPHAN_CLEANUP_MAX_ENTRIES,
            max_tree_entries=ORPHAN_CLEANUP_MAX_TREE_ENTRIES,
        )
        if cleanup.removed_jobs or cleanup.errors or cleanup.skipped_oversized:
            logger.info(
                "Orphan job cleanup scanned=%s removed=%s oversized_skipped=%s errors=%s",
                cleanup.scanned_entries,
                cleanup.removed_jobs,
                cleanup.skipped_oversized,
                cleanup.errors,
            )
        with generation_deadline(GENERATION_TIMEOUT_SECONDS):
            try:
                return _generate_job(request, http_request)
            except ContextBudgetExceeded:
                raise HTTPException(
                    status_code=422,
                    detail="The generation context exceeds a safe limit. Please simplify the request.",
                ) from None
            except GenerationDeadlineExceeded:
                raise HTTPException(
                    status_code=504,
                    detail="Generation timed out. Please try again with a smaller request.",
                ) from None
            except TRANSIENT_PROVIDER_ERRORS:
                raise HTTPException(
                    status_code=503,
                    detail="The AI service is temporarily unavailable. Please try again shortly.",
                ) from None
    finally:
        _GENERATION_SLOTS.release()

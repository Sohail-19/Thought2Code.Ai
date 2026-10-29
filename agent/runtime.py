"""Shared bounded runtime settings for generation and model calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import re
import time

from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, InternalServerError

_ = load_dotenv()


def _bounded_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def _bounded_float_setting(
    name: str, default: float, minimum: float, maximum: float
) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


MAX_PROMPT_CHARACTERS = _bounded_setting(
    "THOUGHT2CODE_MAX_PROMPT_CHARACTERS", 4_000, 10, 4_000
)
MAX_REQUEST_BODY_BYTES = _bounded_setting(
    "THOUGHT2CODE_MAX_REQUEST_BODY_BYTES", 64 * 1_024, 4 * 1_024, 1_048_576
)
GENERATION_TIMEOUT_SECONDS = _bounded_float_setting(
    "THOUGHT2CODE_GENERATION_TIMEOUT_SECONDS", 300.0, 10.0, 900.0
)
MODEL_REQUEST_TIMEOUT_SECONDS = _bounded_float_setting(
    "THOUGHT2CODE_MODEL_TIMEOUT_SECONDS", 30.0, 1.0, 120.0
)
MAX_TRANSIENT_RETRIES = _bounded_setting(
    "THOUGHT2CODE_TRANSIENT_RETRIES", 1, 0, 2
)
TRANSIENT_RETRY_BACKOFF_SECONDS = _bounded_float_setting(
    "THOUGHT2CODE_TRANSIENT_BACKOFF_SECONDS", 0.25, 0.0, 1.0
)
MAX_IMPLEMENTATION_STEPS = _bounded_setting(
    "THOUGHT2CODE_MAX_IMPLEMENTATION_STEPS", 40, 1, 100
)
# Planner + Architect + one Coder node per implementation task + one final
# Coder node that observes the completed plan and routes to END.
MAX_GRAPH_RECURSION_STEPS = MAX_IMPLEMENTATION_STEPS + 3
MAX_CODER_AGENT_STEPS = _bounded_setting(
    "THOUGHT2CODE_MAX_CODER_AGENT_STEPS", 8, 2, 16
)
# ReAct executes a pre-model hook and agent node per model call, plus one tools
# node after each non-final model call, and an initial input superstep.
MAX_CODER_AGENT_RECURSION_STEPS = 3 * MAX_CODER_AGENT_STEPS
MAX_PLANNER_CONTEXT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_PLANNER_CONTEXT_TOKENS", 6_000, 1_000, 12_000
)
MAX_ARCHITECT_CONTEXT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_ARCHITECT_CONTEXT_TOKENS", 12_000, 2_000, 24_000
)
MAX_PLANNER_OUTPUT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_PLANNER_MAX_OUTPUT_TOKENS", 4_096, 512, 8_192
)
MAX_ARCHITECT_OUTPUT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_ARCHITECT_MAX_OUTPUT_TOKENS", 8_192, 1_024, 16_384
)
MAX_CODER_CONTEXT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_CODER_CONTEXT_TOKENS", 16_000, 2_000, 32_000
)
MAX_REPAIR_CONTEXT_TOKENS = _bounded_setting(
    "THOUGHT2CODE_REPAIR_CONTEXT_TOKENS", 16_000, 2_000, 32_000
)
MAX_GENERATED_FILE_BYTES = _bounded_setting(
    "THOUGHT2CODE_MAX_GENERATED_FILE_BYTES", 1_048_576, 64 * 1_024, 1_048_576
)
MAX_GENERATED_PROJECT_BYTES = _bounded_setting(
    "THOUGHT2CODE_MAX_GENERATED_PROJECT_BYTES",
    50 * 1_048_576,
    1_048_576,
    50 * 1_048_576,
)
MAX_GENERATED_PROJECT_FILES = _bounded_setting(
    "THOUGHT2CODE_MAX_GENERATED_FILES", 200, 1, 1_000
)
MAX_CONCURRENT_GENERATIONS = _bounded_setting(
    "THOUGHT2CODE_MAX_CONCURRENT_GENERATIONS", 1, 1, 4
)
ORPHAN_JOB_RETENTION_SECONDS = _bounded_setting(
    "THOUGHT2CODE_ORPHAN_JOB_RETENTION_SECONDS",
    7 * 24 * 60 * 60,
    60 * 60,
    365 * 24 * 60 * 60,
)
ORPHAN_CLEANUP_MAX_JOBS = _bounded_setting(
    "THOUGHT2CODE_ORPHAN_CLEANUP_MAX_JOBS", 10, 1, 100
)
ORPHAN_CLEANUP_MAX_ENTRIES = _bounded_setting(
    "THOUGHT2CODE_ORPHAN_CLEANUP_MAX_ENTRIES", 100, 10, 1_000
)
ORPHAN_CLEANUP_MAX_TREE_ENTRIES = _bounded_setting(
    "THOUGHT2CODE_ORPHAN_CLEANUP_MAX_TREE_ENTRIES", 1_000, 10, 10_000
)

TRANSIENT_PROVIDER_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class GenerationDeadlineExceeded(TimeoutError):
    """Raised when the soft end-to-end generation deadline has elapsed."""


class ContextBudgetExceeded(ValueError):
    """Raised when a model input exceeds its configured context budget."""


class StructuredOutputIncomplete(RuntimeError):
    """A Planner or Architect structured response was truncated or invalid."""


class CoderAgentStepLimitExceeded(RuntimeError):
    """The per-task Coder reached its configured model-iteration limit."""

    def __init__(self, configured_limit: int, observed_model_iterations: int):
        super().__init__("Coder reached its configured per-task step limit")
        self.configured_limit = configured_limit
        self.observed_model_iterations = observed_model_iterations


class CoderReActRecursionLimitExceeded(RuntimeError):
    """The ReAct subgraph exhausted its graph-superstep safety cap."""

    def __init__(self, configured_limit: int, observed_model_iterations: int):
        super().__init__("Coder ReAct graph reached its recursion limit")
        self.configured_limit = configured_limit
        self.observed_model_iterations = observed_model_iterations


@dataclass
class GraphExecutionTrace:
    node_steps: int = 0
    last_node: str = "unknown"


class GraphRecursionLimitExceeded(RuntimeError):
    """Outer workflow recursion cap, with safe counts gathered locally."""

    def __init__(self, configured_limit: int, trace: GraphExecutionTrace):
        super().__init__("The outer workflow reached its configured recursion limit")
        self.configured_limit = configured_limit
        self.observed_node_steps = trace.node_steps
        self.last_node = trace.last_node


_GRAPH_EXECUTION_TRACE: ContextVar[GraphExecutionTrace | None] = ContextVar(
    "thought2code_graph_execution_trace", default=None
)
_LOCAL_DIAGNOSTIC_CONTEXT: ContextVar[tuple[str, bool] | None] = ContextVar(
    "thought2code_local_diagnostic_context", default=None
)
_DIAGNOSTIC_LOGGER = logging.getLogger("thought2code.diagnostics")
_DIAGNOSTIC_EVENTS = {
    "node_entered",
    "node_exited",
    "node_failed",
    "coder_task_started",
    "coder_iteration",
    "coder_task_completed",
    "tool_started",
    "tool_succeeded",
    "tool_failed",
    "workflow_terminated",
}
_DIAGNOSTIC_NODES = {"planner", "architect", "coder"}
_DIAGNOSTIC_TOOLS = {
    "read_file",
    "write_file",
    "list_generated_files",
    "get_current_directory",
}


@contextmanager
def local_diagnostic_scope(job_id: str, enabled: bool):
    """Enable allowlisted event logging only for a request approved as local."""
    safe_job_id = job_id if re.fullmatch(r"[0-9a-f]{32}", job_id) else "unknown"
    token = _LOCAL_DIAGNOSTIC_CONTEXT.set((safe_job_id, bool(enabled)))
    try:
        yield
    finally:
        _LOCAL_DIAGNOSTIC_CONTEXT.reset(token)


def local_diagnostics_active() -> bool:
    context = _LOCAL_DIAGNOSTIC_CONTEXT.get()
    return context is not None and context[1]


def emit_local_diagnostic(
    event: str,
    *,
    node: str | None = None,
    task_index: int | None = None,
    iteration: int | None = None,
    tool_name: str | None = None,
    outcome: str | None = None,
    termination_reason: str | None = None,
    configured_limit: int | None = None,
    configured_limit_unit: str | None = None,
    observed_count: int | None = None,
    count_kind: str | None = None,
) -> None:
    """Emit only fixed event names and bounded non-content metadata."""
    context = _LOCAL_DIAGNOSTIC_CONTEXT.get()
    if context is None or not context[1] or event not in _DIAGNOSTIC_EVENTS:
        return
    if node not in _DIAGNOSTIC_NODES:
        node = "unknown"
    if tool_name not in _DIAGNOSTIC_TOOLS:
        tool_name = "unknown"
    if outcome not in {"success", "failure", "entered", "exited"}:
        outcome = "unavailable"
    allowed_reasons = {
        "plan_created",
        "tasks_created",
        "task_advanced",
        "all_tasks_complete",
        "per_task_limit",
        "model_iteration_limit",
        "coder_react_recursion_limit",
        "outer_graph_limit",
        "validation_failed",
        "validation_passed",
        "generation_complete",
        "unexpected_failure",
    }
    if termination_reason not in allowed_reasons:
        termination_reason = "unavailable"
    if not isinstance(task_index, int) or not 0 <= task_index <= 1_000:
        task_index = -1
    if not isinstance(iteration, int) or not 0 <= iteration <= 1_000:
        iteration = -1
    if not isinstance(configured_limit, int) or not 0 <= configured_limit <= 1_000:
        configured_limit = -1
    if not isinstance(observed_count, int) or not 0 <= observed_count <= 1_000:
        observed_count = -1
    if configured_limit_unit not in {"graph_supersteps", "model_iterations"}:
        configured_limit_unit = "unavailable"
    if count_kind not in {"node_steps", "model_iterations"}:
        count_kind = "unavailable"
    _DIAGNOSTIC_LOGGER.info(
        "Generation diagnostic job_id=%s event=%s node=%s task_index=%s "
        "iteration=%s tool_name=%s outcome=%s termination_reason=%s "
        "configured_limit=%s configured_limit_unit=%s observed_count=%s count_kind=%s",
        context[0],
        event,
        node,
        task_index,
        iteration,
        tool_name,
        outcome,
        termination_reason,
        configured_limit,
        configured_limit_unit,
        observed_count,
        count_kind,
    )


@contextmanager
def graph_execution_trace():
    trace = GraphExecutionTrace()
    token = _GRAPH_EXECUTION_TRACE.set(trace)
    try:
        yield trace
    finally:
        _GRAPH_EXECUTION_TRACE.reset(token)


def record_graph_node_execution(node_name: str) -> None:
    trace = _GRAPH_EXECUTION_TRACE.get()
    if trace is not None:
        trace.node_steps += 1
        trace.last_node = node_name


def estimate_context_tokens(value: str) -> int:
    """Conservative deterministic token-equivalent estimate without a tokenizer."""
    byte_count = len(value.encode("utf-8"))
    return (byte_count + 2) // 3


def enforce_context_budget(stage: str, value: str, maximum_tokens: int) -> int:
    estimate = estimate_context_tokens(value)
    if estimate > maximum_tokens:
        raise ContextBudgetExceeded(
            f"The {stage} input exceeds its configured context limit. "
            "Please simplify the project request or generated files."
        )
    return estimate


_GENERATION_DEADLINE: ContextVar[float | None] = ContextVar(
    "thought2code_generation_deadline", default=None
)
_TRANSIENT_RETRIES_REMAINING: ContextVar[int | None] = ContextVar(
    "thought2code_transient_retries_remaining", default=None
)


@contextmanager
def generation_deadline(seconds: float):
    now = time.monotonic()
    requested_deadline = now + seconds
    current_deadline = _GENERATION_DEADLINE.get()
    if current_deadline is not None:
        requested_deadline = min(requested_deadline, current_deadline)
    token = _GENERATION_DEADLINE.set(requested_deadline)
    retry_token = _TRANSIENT_RETRIES_REMAINING.set(MAX_TRANSIENT_RETRIES)
    try:
        yield
    finally:
        _TRANSIENT_RETRIES_REMAINING.reset(retry_token)
        _GENERATION_DEADLINE.reset(token)


def remaining_generation_seconds() -> float | None:
    deadline = _GENERATION_DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()


def ensure_generation_time() -> None:
    remaining = remaining_generation_seconds()
    if remaining is not None and remaining <= 0:
        raise GenerationDeadlineExceeded("Generation deadline exceeded")


def bounded_model(model, model_updates: dict | None = None):
    ensure_generation_time()
    remaining = remaining_generation_seconds()
    timeout = MODEL_REQUEST_TIMEOUT_SECONDS
    if remaining is not None:
        timeout = min(timeout, remaining)
    updates = {"request_timeout": timeout, "max_retries": 0}
    if model_updates:
        updates.update(model_updates)
    return model.model_copy(update=updates)


def invoke_with_transient_retries(model, operation, *, model_updates=None):
    """Retry only explicitly transient Groq transport/server exceptions."""
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        ensure_generation_time()
        try:
            return operation(bounded_model(model, model_updates))
        except TRANSIENT_PROVIDER_ERRORS:
            retries_remaining = _TRANSIENT_RETRIES_REMAINING.get()
            if attempt >= MAX_TRANSIENT_RETRIES or retries_remaining == 0:
                ensure_generation_time()
                raise
            if retries_remaining is not None:
                _TRANSIENT_RETRIES_REMAINING.set(retries_remaining - 1)
            delay = TRANSIENT_RETRY_BACKOFF_SECONDS * (attempt + 1)
            remaining = remaining_generation_seconds()
            if remaining is not None and remaining <= delay:
                ensure_generation_time()
                raise
            if delay:
                time.sleep(delay)
    raise AssertionError("Transient retry loop exited unexpectedly")

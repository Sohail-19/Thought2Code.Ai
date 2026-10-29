
from functools import wraps
import json

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from agent.prompts import *
from agent.states import *
from agent.tools import (
    write_file,
    read_file,
    get_current_directory,
    list_generated_files,
)
from agent.runtime import (
    MAX_ARCHITECT_CONTEXT_TOKENS,
    MAX_ARCHITECT_OUTPUT_TOKENS,
    MAX_CODER_AGENT_STEPS,
    MAX_CODER_AGENT_RECURSION_STEPS,
    MAX_CODER_CONTEXT_TOKENS,
    MAX_GRAPH_RECURSION_STEPS,
    MAX_IMPLEMENTATION_STEPS,
    MAX_PLANNER_CONTEXT_TOKENS,
    MAX_PLANNER_OUTPUT_TOKENS,
    CoderAgentStepLimitExceeded,
    CoderReActRecursionLimitExceeded,
    StructuredOutputIncomplete,
    enforce_context_budget,
    ensure_generation_time,
    emit_local_diagnostic,
    invoke_with_transient_retries,
    record_graph_node_execution,
)

_ = load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


def _invoke_structured_output(model, schema, prompt: str, stage: str):
    """Parse an include-raw model result and fail closed on truncation/errors."""
    try:
        response = model.with_structured_output(schema, include_raw=True).invoke(prompt)
    except (OutputParserException, ValidationError):
        raise StructuredOutputIncomplete(
            f"{stage} structured output was incomplete"
        ) from None
    if not isinstance(response, dict):
        raise StructuredOutputIncomplete(f"{stage} structured output was incomplete")

    raw = response.get("raw")
    metadata = getattr(raw, "response_metadata", {}) if raw is not None else {}
    finish_reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    if finish_reason == "length":
        raise StructuredOutputIncomplete(f"{stage} output token limit was reached")
    if finish_reason not in {"stop", "tool_calls"}:
        raise StructuredOutputIncomplete(
            f"{stage} response did not confirm structured output completion"
        )

    parsed = response.get("parsed")
    if response.get("parsing_error") is not None or not isinstance(parsed, schema):
        # Do not chain parser exceptions: their messages can contain response text.
        raise StructuredOutputIncomplete(f"{stage} structured output was incomplete") from None
    return parsed


def _with_node_diagnostics(node_name: str, node):
    """Annotate node failures so the API traceback identifies the failing stage."""
    @wraps(node)
    def run(state):
        record_graph_node_execution(node_name)
        coder_state = state.get("coder_state") if isinstance(state, dict) else None
        task_index = getattr(coder_state, "current_step_idx", None)
        task_index = task_index + 1 if isinstance(task_index, int) else None
        emit_local_diagnostic(
            "node_entered", node=node_name, task_index=task_index, outcome="entered"
        )
        try:
            ensure_generation_time()
            result = node(state)
            reason = {
                "planner": "plan_created",
                "architect": "tasks_created",
                "coder": (
                    "all_tasks_complete"
                    if result.get("status") == "DONE"
                    else "task_advanced"
                ),
            }.get(node_name)
            emit_local_diagnostic(
                "node_exited",
                node=node_name,
                task_index=task_index,
                outcome="exited",
                termination_reason=reason,
            )
            return result
        except Exception as error:
            emit_local_diagnostic(
                "node_failed",
                node=node_name,
                task_index=task_index,
                outcome="failure",
                termination_reason="unexpected_failure",
            )
            error.add_note(f"LangGraph node: {node_name}")
            raise
    return run


class _CoderToolDiagnosticHandler(BaseCallbackHandler):
    """Record tool names/outcomes without callback input or output contents."""

    def __init__(self, task_index: int):
        self.task_index = task_index
        self._names_by_run: dict[object, str] = {}
        self.model_iterations = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.model_iterations += 1
        emit_local_diagnostic(
            "coder_iteration",
            node="coder",
            task_index=self.task_index,
            iteration=self.model_iterations,
            outcome="entered",
        )

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name") if isinstance(serialized, dict) else None
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._names_by_run[run_id] = name
        emit_local_diagnostic(
            "tool_started",
            node="coder",
            task_index=self.task_index,
            tool_name=name,
            outcome="entered",
        )

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        name = self._names_by_run.pop(run_id, None)
        emit_local_diagnostic(
            "tool_succeeded",
            node="coder",
            task_index=self.task_index,
            tool_name=name,
            outcome="success",
        )

    def on_tool_error(self, error, **kwargs):
        run_id = kwargs.get("run_id")
        name = self._names_by_run.pop(run_id, None)
        emit_local_diagnostic(
            "tool_failed",
            node="coder",
            task_index=self.task_index,
            tool_name=name,
            outcome="failure",
            termination_reason="unexpected_failure",
        )

def planner_agent(state: dict) -> dict:
    """Converts the user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    prompt = planner_prompt(user_prompt)
    enforce_context_budget("Planner", prompt, MAX_PLANNER_CONTEXT_TOKENS)

    resp = invoke_with_transient_retries(
        llm,
        lambda model: _invoke_structured_output(
            model, Plan, prompt, "Planner"
        ),
        model_updates={"max_tokens": MAX_PLANNER_OUTPUT_TOKENS},
    )

    if resp is None:
        raise StructuredOutputIncomplete("Planner structured output was incomplete")
    if not resp.files:
        raise StructuredOutputIncomplete("Planner returned no planned files")
    if len(resp.files) > MAX_IMPLEMENTATION_STEPS:
        raise ValueError("Planner returned too many planned files.")
    enforce_context_budget(
        "project plan", resp.model_dump_json(), MAX_ARCHITECT_CONTEXT_TOKENS
    )

    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates a TaskPlan from the project Plan."""
    plan: Plan = state["plan"]

    prompt = architect_prompt(plan=plan.model_dump_json())
    enforce_context_budget("Architect", prompt, MAX_ARCHITECT_CONTEXT_TOKENS)
    resp = invoke_with_transient_retries(
        llm,
        lambda model: _invoke_structured_output(
            model, TaskPlan, prompt, "Architect"
        ),
        model_updates={"max_tokens": MAX_ARCHITECT_OUTPUT_TOKENS},
    )

    if resp is None:
        raise StructuredOutputIncomplete("Architect structured output was incomplete")

    if not resp.implementation_steps:
        raise StructuredOutputIncomplete("Architect returned no implementation steps")
    if len(resp.implementation_steps) > MAX_IMPLEMENTATION_STEPS:
        raise ValueError("Architect returned too many implementation steps.")
    enforce_context_budget(
        "implementation plan", resp.model_dump_json(), MAX_ARCHITECT_CONTEXT_TOKENS
    )

    resp.plan = plan

    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    """Implements one task using the tool-using Coder agent."""

    coder_state: CoderState = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        emit_local_diagnostic(
            "workflow_terminated",
            node="coder",
            task_index=len(steps),
            outcome="success",
            termination_reason="all_tasks_complete",
        )
        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    current_task = steps[coder_state.current_step_idx]
    task_index = coder_state.current_step_idx + 1
    emit_local_diagnostic(
        "coder_task_started",
        node="coder",
        task_index=task_index,
        outcome="entered",
    )

    system_prompt = coder_system_prompt()

    coder_tools = [
        read_file,
        write_file,
        list_generated_files,
        get_current_directory,
    ]

    def run_coder_task(model):
        ensure_generation_time()
        existing_content = read_file.invoke({"path": current_task.filepath})
        user_prompt = (
            f"Task: {current_task.task_description}\n"
            f"File: {current_task.filepath}\n"
            f"Existing content:\n{existing_content}\n"
            "Use write_file(path, content) to save your changes."
        )
        enforce_context_budget(
            "Coder", system_prompt + "\n" + user_prompt, MAX_CODER_CONTEXT_TOKENS
        )
        diagnostic_handler = _CoderToolDiagnosticHandler(task_index)

        def bound_accumulated_messages(agent_state):
            if diagnostic_handler.model_iterations >= MAX_CODER_AGENT_STEPS:
                emit_local_diagnostic(
                    "workflow_terminated",
                    node="coder",
                    task_index=task_index,
                    iteration=diagnostic_handler.model_iterations,
                    outcome="failure",
                    termination_reason="model_iteration_limit",
                    configured_limit=MAX_CODER_AGENT_STEPS,
                    configured_limit_unit="model_iterations",
                    observed_count=diagnostic_handler.model_iterations,
                    count_kind="model_iterations",
                )
                raise CoderAgentStepLimitExceeded(
                    MAX_CODER_AGENT_STEPS, diagnostic_handler.model_iterations
                )
            messages = agent_state.get("messages", [])
            serialized = json.dumps(
                [
                    message.model_dump()
                    if hasattr(message, "model_dump")
                    else message
                    for message in messages
                ],
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            enforce_context_budget(
                "Coder conversation", serialized, MAX_CODER_CONTEXT_TOKENS
            )
            return {"llm_input_messages": messages}

        react_agent = create_react_agent(
            model, coder_tools, pre_model_hook=bound_accumulated_messages
        )
        try:
            coder_config = {
                "recursion_limit": MAX_CODER_AGENT_RECURSION_STEPS,
                "callbacks": [diagnostic_handler],
            }
            coder_result = react_agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                },
                coder_config,
            )
        except CoderAgentStepLimitExceeded:
            raise
        except GraphRecursionError:
            emit_local_diagnostic(
                "workflow_terminated",
                node="coder",
                task_index=task_index,
                iteration=diagnostic_handler.model_iterations,
                outcome="failure",
                termination_reason="coder_react_recursion_limit",
                configured_limit=MAX_CODER_AGENT_RECURSION_STEPS,
                configured_limit_unit="graph_supersteps",
                observed_count=diagnostic_handler.model_iterations,
                count_kind="model_iterations",
            )
            raise CoderReActRecursionLimitExceeded(
                MAX_CODER_AGENT_RECURSION_STEPS,
                diagnostic_handler.model_iterations,
            ) from None
        ensure_generation_time()
        return coder_result

    coder_result = invoke_with_transient_retries(llm, run_coder_task)

    messages = coder_result.get("messages", [])

    if not messages:
        raise RuntimeError(
            f"Coder returned no messages for {current_task.filepath}"
        )

    written_content = read_file.invoke(
        {"path": current_task.filepath}
    )

    if not written_content.strip():
        emit_local_diagnostic(
            "coder_task_completed",
            node="coder",
            task_index=task_index,
            outcome="failure",
            termination_reason="unexpected_failure",
        )
        raise RuntimeError(
            f"Coder did not create or populate {current_task.filepath}"
        )

    coder_state.current_step_idx += 1
    emit_local_diagnostic(
        "coder_task_completed",
        node="coder",
        task_index=task_index,
        outcome="success",
        termination_reason="task_advanced",
    )

    return {"coder_state": coder_state}


graph = StateGraph(AgentState)

graph.add_node("planner", _with_node_diagnostics("planner", planner_agent))
graph.add_node("architect", _with_node_diagnostics("architect", architect_agent))
graph.add_node("coder", _with_node_diagnostics("coder", coder_agent))

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    lambda state: (
        "END" if state.get("status") == "DONE" else "coder"
    ),
    {
        "END": END,
        "coder": "coder",
    },
)

graph.set_entry_point("planner")

agent = graph.compile()


if __name__ == "__main__":
    result = agent.invoke(
        {
            "user_prompt": (
                "Build a colourful modern todo app in HTML, CSS and JavaScript"
            )
        },
        {"recursion_limit": MAX_GRAPH_RECURSION_STEPS},
    )

    print("Thought2Code.Ai generation finished.")
    print("Final State:", result)


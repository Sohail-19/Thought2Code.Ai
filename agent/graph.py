
from functools import wraps

from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import *
from agent.states import *
from agent.tools import (
    write_file,
    read_file,
    get_current_directory,
    list_generated_files,
)

_ = load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


def _with_node_diagnostics(node_name: str, node):
    """Annotate node failures so the API traceback identifies the failing stage."""
    @wraps(node)
    def run(state):
        try:
            return node(state)
        except Exception as error:
            error.add_note(f"LangGraph node: {node_name}")
            raise

    return run


def planner_agent(state: dict) -> dict:
    """Converts the user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]

    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )

    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates a TaskPlan from the project Plan."""
    plan: Plan = state["plan"]

    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )

    if resp is None:
        raise ValueError("Architect did not return a valid response.")

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
        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    current_task = steps[coder_state.current_step_idx]

    existing_content = read_file.invoke(
        {"path": current_task.filepath}
    )

    system_prompt = coder_system_prompt()

    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [
        read_file,
        write_file,
        list_generated_files,
        get_current_directory,
    ]

    react_agent = create_react_agent(llm, coder_tools)

    coder_result = react_agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        }
    )

    messages = coder_result.get("messages", [])

    if not messages:
        raise RuntimeError(
            f"Coder returned no messages for {current_task.filepath}"
        )

    written_content = read_file.invoke(
        {"path": current_task.filepath}
    )

    if not written_content.strip():
        raise RuntimeError(
            f"Coder did not create or populate {current_task.filepath}"
        )

    coder_state.current_step_idx += 1

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
        {"recursion_limit": 100},
    )

    print("Thought2Code.Ai generation finished.")
    print("Final State:", result)


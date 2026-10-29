import pytest
from types import SimpleNamespace
from langchain_core.exceptions import OutputParserException
from langgraph.errors import GraphRecursionError

import agent.graph as graph_module
from agent.runtime import (
    MAX_CODER_AGENT_RECURSION_STEPS,
    MAX_GRAPH_RECURSION_STEPS,
    MAX_IMPLEMENTATION_STEPS,
    CoderAgentStepLimitExceeded,
    CoderReActRecursionLimitExceeded,
    ContextBudgetExceeded,
    StructuredOutputIncomplete,
)
from agent.states import CoderState, ImplementationTask, Plan, TaskPlan


class StubStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _prompt):
        if self.schema is Plan:
            parsed = Plan(
                name="Graph state regression",
                description="A project used to verify graph state retention",
                techstack="Python",
                features=["State survives each workflow node"],
                files=[{"path": "index.html", "purpose": "Application entry point"}],
            )
        elif self.schema is TaskPlan:
            parsed = TaskPlan(
                implementation_steps=[
                    ImplementationTask(
                        filepath="index.html",
                        task_description="Create the application entry point",
                    )
                ]
            )
        else:
            raise AssertionError(f"Unexpected structured output schema: {self.schema}")
        return {
            "raw": SimpleNamespace(response_metadata={"finish_reason": "tool_calls"}),
            "parsed": parsed,
            "parsing_error": None,
        }


class StubModel:
    def __init__(self):
        self.updates = []

    def model_copy(self, *, update):
        assert update["max_retries"] == 0
        assert update["request_timeout"] > 0
        self.updates.append(update)
        return self

    def with_structured_output(self, schema, *, include_raw=False):
        assert include_raw is True
        return StubStructuredModel(schema)


def test_compiled_graph_preserves_plan_through_coder(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "llm", StubModel())

    class StubCoder:
        def invoke(self, _state, _config):
            graph_module.write_file.invoke(
                {"path": "index.html", "content": "<main>Generated</main>"}
            )
            return {"messages": ["file written"]}

    def make_stub_coder(_llm, _tools, *, pre_model_hook):
        class HookedStubCoder(StubCoder):
            def invoke(self, state, config):
                pre_model_hook({"messages": state["messages"]})
                return super().invoke(state, config)

        return HookedStubCoder()

    monkeypatch.setattr(graph_module, "create_react_agent", make_stub_coder)

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        result = graph_module.agent.invoke(
            {"user_prompt": "A safe local regression prompt"},
            {"recursion_limit": MAX_GRAPH_RECURSION_STEPS},
        )

    assert isinstance(result["plan"], Plan)
    assert result["plan"].name == "Graph state regression"
    assert isinstance(result["task_plan"], TaskPlan)
    assert result["status"] == "DONE"
    assert MAX_GRAPH_RECURSION_STEPS == MAX_IMPLEMENTATION_STEPS + 3
    assert {update["max_tokens"] for update in graph_module.llm.updates if "max_tokens" in update} >= {
        graph_module.MAX_PLANNER_OUTPUT_TOKENS,
        graph_module.MAX_ARCHITECT_OUTPUT_TOKENS,
    }


def test_simple_three_file_style_plan_finishes_all_coder_tasks_offline(
    monkeypatch, tmp_path
):
    class ThreeStepStructuredModel(StubModel):
        def with_structured_output(self, schema, *, include_raw=False):
            assert include_raw is True
            if schema is not TaskPlan:
                return super().with_structured_output(schema, include_raw=include_raw)

            class Structured:
                def invoke(self, _prompt):
                    return {
                        "raw": SimpleNamespace(
                            response_metadata={"finish_reason": "tool_calls"}
                        ),
                        "parsed": TaskPlan(
                            implementation_steps=[
                                ImplementationTask(
                                    filepath=path,
                                    task_description=f"Create {path}",
                                )
                                for path in ("index.html", "styles.css", "app.js")
                            ]
                        ),
                        "parsing_error": None,
                    }

            return Structured()

    def make_coder(_model, _tools, *, pre_model_hook):
        class CompletingCoder:
            def invoke(self, state, _config):
                pre_model_hook({"messages": state["messages"]})
                task = state["messages"][-1]["content"]
                filepath = next(
                    line.removeprefix("File: ")
                    for line in task.splitlines()
                    if line.startswith("File: ")
                )
                graph_module.write_file.invoke(
                    {"path": filepath, "content": f"content for {filepath}"}
                )
                return {"messages": state["messages"] + ["task complete"]}

        return CompletingCoder()

    monkeypatch.setattr(graph_module, "llm", ThreeStepStructuredModel())
    monkeypatch.setattr(graph_module, "create_react_agent", make_coder)

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        result = graph_module.agent.invoke(
            {"user_prompt": "Create a simple local webpage"},
            {"recursion_limit": graph_module.MAX_GRAPH_RECURSION_STEPS},
        )

    assert result["status"] == "DONE"
    assert result["coder_state"].current_step_idx == 3
    assert all((tmp_path / path).is_file() for path in ("index.html", "styles.css", "app.js"))


def test_coder_model_tool_list_excludes_shell_and_arbitrary_command_execution(
    monkeypatch, tmp_path
):
    exposed_tool_names = []

    def make_stub_coder(_model, tools, *, pre_model_hook):
        exposed_tool_names.extend(tool.name for tool in tools)

        class BoundStubCoder:
            def invoke(self, state, _config):
                pre_model_hook({"messages": state["messages"]})
                graph_module.write_file.invoke(
                    {"path": "app.py", "content": "print('safe')"}
                )
                return {"messages": ["file written"]}

        return BoundStubCoder()

    monkeypatch.setattr(graph_module, "llm", StubModel())
    monkeypatch.setattr(graph_module, "create_react_agent", make_stub_coder)
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath="app.py", task_description="Create app")
        ]
    )
    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        graph_module.coder_agent({"task_plan": task_plan})

    assert exposed_tool_names == [
        "read_file",
        "write_file",
        "list_generated_files",
        "get_current_directory",
    ]
    assert not any(
        any(term in name.lower() for term in ("shell", "command", "exec", "run_cmd"))
        for name in exposed_tool_names
    )


def test_coder_model_iteration_budget_is_independent_of_react_graph_steps(
    monkeypatch, tmp_path
):
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath="app.py", task_description="Create app")
        ]
    )

    def make_step_capped_coder(_model, _tools, *, pre_model_hook):
        class StepCappedCoder:
            def invoke(self, state, config):
                callback = config["callbacks"][0]
                for _ in range(graph_module.MAX_CODER_AGENT_STEPS):
                    pre_model_hook({"messages": state["messages"]})
                    callback.on_chat_model_start({}, [])
                # Refuse the next model invocation before it reaches the model.
                pre_model_hook({"messages": state["messages"]})
                raise AssertionError("the model iteration budget should have stopped first")

        return StepCappedCoder()

    monkeypatch.setattr(graph_module, "llm", StubModel())
    monkeypatch.setattr(
        graph_module, "create_react_agent", make_step_capped_coder
    )

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        with pytest.raises(CoderAgentStepLimitExceeded) as raised:
            graph_module.coder_agent({"task_plan": task_plan})

    assert raised.value.configured_limit == graph_module.MAX_CODER_AGENT_STEPS
    assert raised.value.observed_model_iterations == graph_module.MAX_CODER_AGENT_STEPS
    assert MAX_CODER_AGENT_RECURSION_STEPS == 3 * graph_module.MAX_CODER_AGENT_STEPS


def test_unexpected_react_graph_limit_has_separate_safe_classification(
    monkeypatch, tmp_path
):
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath="app.py", task_description="Create app")
        ]
    )

    class RecursionLimitedCoder:
        def invoke(self, _state, _config):
            raise GraphRecursionError("private inner graph state")

    monkeypatch.setattr(graph_module, "llm", StubModel())
    monkeypatch.setattr(
        graph_module,
        "create_react_agent",
        lambda *_args, **_kwargs: RecursionLimitedCoder(),
    )

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        with pytest.raises(CoderReActRecursionLimitExceeded) as raised:
            graph_module.coder_agent({"task_plan": task_plan})

    assert "private inner graph state" not in str(raised.value)
    assert raised.value.configured_limit == MAX_CODER_AGENT_RECURSION_STEPS


def test_compiled_graph_rejects_empty_implementation_steps(monkeypatch):
    class EmptyTaskPlanModel(StubModel):
        def with_structured_output(self, schema, *, include_raw=False):
            assert include_raw is True
            if schema is TaskPlan:
                class EmptyTaskPlan:
                    def invoke(self, _prompt):
                        return {
                            "raw": SimpleNamespace(response_metadata={"finish_reason": "tool_calls"}),
                            "parsed": TaskPlan(implementation_steps=[]),
                            "parsing_error": None,
                        }

                return EmptyTaskPlan()
            return super().with_structured_output(
                schema, include_raw=include_raw
            )

    monkeypatch.setattr(graph_module, "llm", EmptyTaskPlanModel())

    with pytest.raises(StructuredOutputIncomplete, match="implementation steps"):
        graph_module.agent.invoke(
            {"user_prompt": "A safe local regression prompt"},
            {"recursion_limit": 100},
        )


def test_compiled_graph_failure_identifies_the_failing_node(monkeypatch):
    class FailingStructuredModel:
        def invoke(self, _prompt):
            raise RuntimeError("structured model failure")

    class FailingModel:
        def model_copy(self, *, update):
            return self

        def with_structured_output(self, _schema, *, include_raw=False):
            assert include_raw is True
            return FailingStructuredModel()

    monkeypatch.setattr(graph_module, "llm", FailingModel())

    with pytest.raises(RuntimeError, match="structured model failure") as raised:
        graph_module.agent.invoke(
            {"user_prompt": "A safe local regression prompt"},
            {"recursion_limit": 100},
        )

    assert "LangGraph node: planner" in raised.value.__notes__


def test_planner_output_cap_is_applied_to_model_copy(monkeypatch):
    model = StubModel()
    monkeypatch.setattr(graph_module, "llm", model)
    monkeypatch.setattr(graph_module, "MAX_PLANNER_OUTPUT_TOKENS", 1_234)

    graph_module.planner_agent({"user_prompt": "A safe local application idea"})

    assert model.updates[-1]["max_tokens"] == 1_234


def test_architect_output_cap_is_applied_to_model_copy(monkeypatch):
    model = StubModel()
    plan = Plan(
        name="Example",
        description="An example project",
        techstack="Python",
        features=["A feature"],
        files=[{"path": "app.py", "purpose": "Application"}],
    )
    monkeypatch.setattr(graph_module, "llm", model)
    monkeypatch.setattr(graph_module, "MAX_ARCHITECT_OUTPUT_TOKENS", 5_678)

    graph_module.architect_agent({"plan": plan})

    assert model.updates[-1]["max_tokens"] == 5_678


def test_planner_rejects_provider_response_truncated_at_output_cap(monkeypatch):
    class TruncatedModel(StubModel):
        def with_structured_output(self, schema, *, include_raw=False):
            assert include_raw is True

            class Structured:
                def invoke(self, _prompt):
                    return {
                        "raw": SimpleNamespace(
                            response_metadata={"finish_reason": "length"}
                        ),
                        "parsed": Plan(
                            name="Partial",
                            description="Partial plan",
                            techstack="Python",
                            features=[],
                            files=[],
                        ),
                        "parsing_error": None,
                    }

            return Structured()

    monkeypatch.setattr(graph_module, "llm", TruncatedModel())

    with pytest.raises(StructuredOutputIncomplete, match="token limit"):
        graph_module.planner_agent({"user_prompt": "A safe local application idea"})


def test_architect_rejects_malformed_structured_response_without_parser_details(
    monkeypatch,
):
    class MalformedModel(StubModel):
        def with_structured_output(self, schema, *, include_raw=False):
            assert include_raw is True

            class Structured:
                def invoke(self, _prompt):
                    return {
                        "raw": SimpleNamespace(
                            response_metadata={"finish_reason": "tool_calls"}
                        ),
                        "parsed": None,
                        "parsing_error": OutputParserException(
                            "private truncated response fragment"
                        ),
                    }

            return Structured()

    plan = Plan(
        name="Example",
        description="An example project",
        techstack="Python",
        features=["A feature"],
        files=[{"path": "app.py", "purpose": "Application"}],
    )
    monkeypatch.setattr(graph_module, "llm", MalformedModel())

    with pytest.raises(StructuredOutputIncomplete) as raised:
        graph_module.architect_agent({"plan": plan})

    assert "private truncated response fragment" not in str(raised.value)


def test_planner_context_budget_fails_before_model_call(monkeypatch):
    class NeverCalledModel:
        def model_copy(self, **_kwargs):
            pytest.fail("oversized Planner input must fail before model invocation")

    monkeypatch.setattr(graph_module, "llm", NeverCalledModel())
    monkeypatch.setattr(graph_module, "MAX_PLANNER_CONTEXT_TOKENS", 1)

    with pytest.raises(ContextBudgetExceeded, match="Planner input"):
        graph_module.planner_agent({"user_prompt": "A sufficiently long prompt"})


def test_architect_context_budget_fails_before_model_call(monkeypatch):
    class NeverCalledModel:
        def model_copy(self, **_kwargs):
            pytest.fail("oversized Architect input must fail before model invocation")

    plan = Plan(
        name="Example",
        description="An example project",
        techstack="Python",
        features=["A feature"],
        files=[{"path": "app.py", "purpose": "Application"}],
    )
    monkeypatch.setattr(graph_module, "llm", NeverCalledModel())
    monkeypatch.setattr(graph_module, "MAX_ARCHITECT_CONTEXT_TOKENS", 1)

    with pytest.raises(ContextBudgetExceeded, match="Architect input"):
        graph_module.architect_agent({"plan": plan})


def test_oversized_architect_implementation_plan_fails_after_model_response(
    monkeypatch,
):
    plan = Plan(
        name="Example",
        description="An example project",
        techstack="Python",
        features=["A feature"],
        files=[{"path": "app.py", "purpose": "Application"}],
    )

    class Structured:
        def invoke(self, _prompt):
                return {
                    "raw": SimpleNamespace(
                        response_metadata={"finish_reason": "tool_calls"}
                    ),
                    "parsed": TaskPlan(
                        implementation_steps=[
                            ImplementationTask(
                                filepath="app.py", task_description="x" * 5_000
                            )
                        ]
                    ),
                    "parsing_error": None,
                }

    class Model(StubModel):
        def with_structured_output(self, _schema, *, include_raw=False):
            assert include_raw is True
            return Structured()

    monkeypatch.setattr(graph_module, "llm", Model())
    monkeypatch.setattr(graph_module, "MAX_ARCHITECT_CONTEXT_TOKENS", 1_000)

    with pytest.raises(ContextBudgetExceeded, match="implementation plan"):
        graph_module.architect_agent({"plan": plan})


def test_coder_context_budget_checks_accumulated_react_messages(
    monkeypatch, tmp_path
):
    class FakeCoder:
        def invoke(self, _state, _config):
            return {"messages": ["not reached"]}

    def make_coder(_llm, _tools, *, pre_model_hook):
        class HookedCoder(FakeCoder):
            def invoke(self, state, config):
                pre_model_hook(
                    {"messages": [{"role": "user", "content": "x" * 5_000}]}
                )
                return super().invoke(state, config)

        return HookedCoder()

    monkeypatch.setattr(graph_module, "llm", StubModel())
    monkeypatch.setattr(graph_module, "create_react_agent", make_coder)
    monkeypatch.setattr(graph_module, "MAX_CODER_CONTEXT_TOKENS", 1_000)
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath="app.js", task_description="Create app")
        ]
    )

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        with pytest.raises(ContextBudgetExceeded, match="Coder conversation"):
            graph_module.coder_agent({"task_plan": task_plan})




def test_coder_does_not_truncate_oversized_existing_file_context(
    monkeypatch, tmp_path
):
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath="app.js", task_description="Update app")
        ]
    )
    (tmp_path / "app.js").write_text("x" * 5_000)
    monkeypatch.setattr(graph_module, "llm", StubModel())
    monkeypatch.setattr(graph_module, "MAX_CODER_CONTEXT_TOKENS", 1_000)
    monkeypatch.setattr(
        graph_module,
        "create_react_agent",
        lambda *_args, **_kwargs: pytest.fail("oversized context must not call model"),
    )

    from agent.tools import use_project_root

    with use_project_root(tmp_path):
        with pytest.raises(ContextBudgetExceeded, match="Coder input"):
            graph_module.coder_agent({"task_plan": task_plan})

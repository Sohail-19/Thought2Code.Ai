import asyncio
import io
import logging
import shutil
import subprocess
import zipfile
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from groq import APIStatusError, RateLimitError
from langgraph.errors import GraphRecursionError
from httpx import Request, Response
from starlette.requests import Request as StarletteRequest

import api.main as api_main
from agent.repair import RepairFile, RepairResponse
from agent.runtime import (
    MAX_CODER_AGENT_STEPS,
    MAX_CODER_AGENT_RECURSION_STEPS,
    MAX_GRAPH_RECURSION_STEPS,
    ContextBudgetExceeded,
    GenerationDeadlineExceeded,
    record_graph_node_execution,
)
from agent.tools import current_project_root, use_project_root, write_file
from agent.states import CoderState, File, ImplementationTask, Plan, TaskPlan


client = TestClient(api_main.app)


def make_http_request(client_host="127.0.0.1", server_host="127.0.0.1"):
    return StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "headers": [],
            "client": (client_host, 12345),
            "server": (server_host, 8000),
        }
    )


def make_completed_job(monkeypatch, tmp_path, files):
    jobs_root = tmp_path / "jobs"
    job_id = uuid4().hex
    job_root = jobs_root / job_id
    job_root.mkdir(parents=True)
    for path, content in files.items():
        output = job_root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    marker_root = jobs_root / ".completed"
    marker_root.mkdir(exist_ok=True)
    (marker_root / f"{job_id}.done").touch()
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    return job_id, job_root


def generation_state(plan_paths, task_paths=None, *, status="DONE", completed_index=None):
    task_paths = task_paths if task_paths is not None else plan_paths
    plan = Plan(
        name="Example",
        description="Example project",
        techstack="HTML, CSS, JavaScript",
        features=["A sample feature"],
        files=[File(path=path, purpose="Required generated output") for path in plan_paths],
    )
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(
                filepath=path,
                task_description="Implement the requested application file",
            )
            for path in task_paths
        ]
    )
    index = len(task_plan.implementation_steps) if completed_index is None else completed_index
    return {
        "plan": plan,
        "task_plan": task_plan,
        "coder_state": CoderState(task_plan=task_plan, current_step_idx=index),
        "status": status,
    }


def assert_failed_job_is_not_accessible(response, jobs_root, *, expected_status=500):
    assert response.status_code == expected_status
    if expected_status == 500:
        assert response.json() == {
            "detail": "Generation failed. Check the backend configuration and try again."
        }
    job_dirs = [
        path
        for path in jobs_root.iterdir()
        if path.is_dir() and path.name not in {".completed", ".active"}
    ]
    assert len(job_dirs) == 1
    job_id = job_dirs[0].name
    assert not (jobs_root / ".completed" / f"{job_id}.done").exists()
    assert client.get(f"/jobs/{job_id}/files").status_code == 404
    assert client.get(f"/jobs/{job_id}/download").status_code == 404


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert api_main.app.title == "Thought2Code.Ai Local API"


def _run_request_body_middleware(middleware, chunks, headers=()):
    received = 0
    sent = []

    async def receive():
        nonlocal received
        if received < len(chunks):
            message = chunks[received]
            received += 1
            return message
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/generate",
        "raw_path": b"/generate",
        "query_string": b"",
        "headers": list(headers),
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 12345),
    }
    asyncio.run(middleware(scope, receive, send))
    return sent


def _draining_asgi_app(calls):
    async def app(_scope, receive, send):
        calls.append(True)
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    return app


def test_request_body_limit_rejects_oversized_declared_length_before_app():
    calls = []
    middleware = api_main.RequestBodyLimitMiddleware(_draining_asgi_app(calls), 5)
    response = _run_request_body_middleware(
        middleware,
        [{"type": "http.request", "body": b"ignored", "more_body": False}],
        [(b"content-length", b"6")],
    )

    assert calls == []
    assert response[0]["status"] == 413
    assert b"".join(message.get("body", b"") for message in response) == (
        b'{"detail":"Request body exceeds the configured size limit."}'
    )


@pytest.mark.parametrize("headers", [(), ((b"content-length", b"1"),)])
def test_request_body_limit_rejects_oversized_streams_without_trusting_length(headers):
    calls = []
    middleware = api_main.RequestBodyLimitMiddleware(_draining_asgi_app(calls), 5)
    response = _run_request_body_middleware(
        middleware,
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ],
        headers,
    )

    assert calls == [True]
    assert response[0]["status"] == 413
    assert b"".join(message.get("body", b"") for message in response) == (
        b'{"detail":"Request body exceeds the configured size limit."}'
    )


def test_request_body_limit_accepts_exact_boundary_body():
    calls = []
    middleware = api_main.RequestBodyLimitMiddleware(_draining_asgi_app(calls), 5)
    response = _run_request_body_middleware(
        middleware,
        [{"type": "http.request", "body": b"12345", "more_body": False}],
        [(b"content-length", b"5")],
    )

    assert calls == [True]
    assert response[0]["status"] == 200
    assert response[1]["body"] == b"12345"


def test_normal_generation_request_passes_body_limit_with_mocked_graph(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.py", "content": "print('ready')"})
        return generation_state(["app.py"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    response = client.post(
        "/generate", json={"project_description": "Build a safe sample application"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_generate_rejects_invalid_description():
    response = client.post("/generate", json={"project_description": "short"})

    assert response.status_code == 422


def test_generate_rejects_oversized_description_without_invoking_graph(monkeypatch):
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("graph must not be invoked"),
    )

    response = client.post(
        "/generate",
        json={"project_description": "x" * (api_main.MAX_PROMPT_CHARACTERS + 1)},
    )

    assert response.status_code == 422


def test_generation_rejects_excess_concurrency_without_queueing(monkeypatch):
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("graph must not be invoked"),
    )
    api_main._GENERATION_SLOTS.acquire()
    try:
        response = client.post(
            "/generate", json={"project_description": "Build a queued test app"}
        )
    finally:
        api_main._GENERATION_SLOTS.release()

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]


def test_generation_timeout_returns_safe_error_without_completion_marker(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def timeout(_state, _config):
        write_file.invoke({"path": "partial.txt", "content": "partial"})
        raise GenerationDeadlineExceeded("sensitive prompt-like diagnostic")

    monkeypatch.setattr(api_main.agent, "invoke", timeout)

    response = client.post(
        "/generate", json={"project_description": "Build a timed test application"}
    )

    assert response.status_code == 504
    assert "sensitive" not in response.text
    job_id = next(path.name for path in jobs_root.iterdir() if path.is_dir())
    assert not (jobs_root / ".completed" / f"{job_id}.done").exists()
    assert client.get(f"/jobs/{job_id}/files").status_code == 404


def test_context_budget_failure_returns_safe_422_without_completion_marker(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fail_context(_state, _config):
        write_file.invoke({"path": "partial.txt", "content": "partial"})
        raise ContextBudgetExceeded("private context contents")

    monkeypatch.setattr(api_main.agent, "invoke", fail_context)
    response = client.post(
        "/generate", json={"project_description": "Build a context test app"}
    )

    assert response.status_code == 422
    assert "private context contents" not in response.text
    job_id = next(
        path.name
        for path in jobs_root.iterdir()
        if path.is_dir() and path.name not in {".completed", ".active"}
    )
    assert not (jobs_root / ".completed" / f"{job_id}.done").exists()
    assert client.get(f"/jobs/{job_id}/files").status_code == 404


def test_transient_provider_failure_returns_safe_503_without_retrying_api_job(
    monkeypatch, tmp_path
):
    class TemporaryProviderFailure(Exception):
        pass

    monkeypatch.setattr(api_main, "TRANSIENT_PROVIDER_ERRORS", (TemporaryProviderFailure,))
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    calls = []

    def fail(_state, _config):
        calls.append(True)
        raise TemporaryProviderFailure("private provider response and prompt")

    monkeypatch.setattr(api_main.agent, "invoke", fail)

    response = client.post(
        "/generate", json={"project_description": "Build a transient test app"}
    )

    assert response.status_code == 503
    assert "private provider" not in response.text
    assert len(calls) == 1
    job_id = next(path.name for path in (tmp_path / "jobs").iterdir() if path.is_dir())
    assert not ((tmp_path / "jobs" / ".completed") / f"{job_id}.done").exists()


def test_generated_file_resource_limit_fails_job_without_completion(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    monkeypatch.setattr(api_main, "MAX_GENERATED_FILE_BYTES", 5)

    def oversized(_state, _config):
        write_file.invoke({"path": "large.txt", "content": "0123456789"})
        return generation_state(["large.txt"])

    monkeypatch.setattr(api_main.agent, "invoke", oversized)

    response = client.post(
        "/generate", json={"project_description": "Build a bounded test app"}
    )

    assert response.status_code == 500
    job_id = next(path.name for path in jobs_root.iterdir() if path.is_dir())
    assert not (jobs_root / ".completed" / f"{job_id}.done").exists()


def test_generate_returns_plan_and_job_scoped_files(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def fake_invoke(state, config):
        assert state["user_prompt"] == "Build a small example application"
        assert config == {"recursion_limit": MAX_GRAPH_RECURSION_STEPS}
        write_file.invoke({"path": "index.html", "content": "<main>Example</main>"})
        write_file.invoke({"path": "app.js", "content": "document.body.dataset.ready = 'yes';"})
        return generation_state(["index.html", "app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)

    response = client.post(
        "/generate",
        json={"project_description": "Build a small example application"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["project"]["name"] == "Example"
    assert body["generated_files"] == ["app.js", "index.html"]
    assert (tmp_path / "jobs" / body["job_id"] / "index.html").read_text() == "<main>Example</main>"
    assert (tmp_path / "jobs" / body["job_id"] / "app.js").read_text() == "document.body.dataset.ready = 'yes';"
    assert (tmp_path / "jobs" / ".completed" / f"{body['job_id']}.done").exists()


def test_mock_model_generates_small_project_through_validation_and_completion(
    monkeypatch, tmp_path, caplog
):
    from types import SimpleNamespace
    import agent.graph as graph_module
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api_main, "_local_diagnostics_enabled", lambda _request: True)
    caplog.set_level(logging.INFO, logger="thought2code.diagnostics")

    plan = Plan(
        name="Small page",
        description="A simple page with one button",
        techstack="HTML, CSS, JavaScript",
        features=["A heading, paragraph, and button"],
        files=[
            File(path="index.html", purpose="Page structure"),
            File(path="style.css", purpose="Page styling"),
            File(path="app.js", purpose="Button behavior"),
        ],
    )
    task_plan = TaskPlan(
        implementation_steps=[
            ImplementationTask(filepath=path, task_description=f"Create {path}")
            for path in ("index.html", "style.css", "app.js")
        ]
    )

    class Structured:
        def __init__(self, parsed):
            self.parsed = parsed

        def invoke(self, _prompt):
            return {
                "raw": SimpleNamespace(
                    response_metadata={"finish_reason": "tool_calls"}
                ),
                "parsed": self.parsed,
                "parsing_error": None,
            }

    class MockModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "thought2code-offline-test"

        def model_copy(self, *, update):
            assert update["request_timeout"] > 0
            assert update["max_retries"] == 0
            return self

        def bind_tools(self, _tools, **_kwargs):
            return self

        def with_structured_output(self, schema, *, include_raw=False):
            assert include_raw is True
            return Structured(plan if schema is Plan else task_plan)

        def _generate(self, messages, stop=None, run_manager=None, **_kwargs):
            tool_messages = [
                message for message in messages if isinstance(message, ToolMessage)
            ]
            if tool_messages and tool_messages[-1].name == "write_file":
                response = AIMessage(content="Task complete")
            else:
                task_text = next(
                    message.content
                    for message in reversed(messages)
                    if isinstance(message, HumanMessage)
                )
                path = next(
                    line.removeprefix("File: ")
                    for line in task_text.splitlines()
                    if line.startswith("File: ")
                )
                if not tool_messages:
                    tool_name = "list_generated_files"
                    tool_args = {"directory": "."}
                else:
                    tool_name = "write_file"
                    tool_args = {"path": path, "content": contents[path]}
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"{tool_name}-{path}",
                            "type": "tool_call",
                        }
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=response)])

    contents = {
        "index.html": (
            '<!doctype html><html><head><link rel="stylesheet" href="style.css">'
            '</head><body><h1>Hello</h1><p>Welcome.</p><button id="action">Go</button>'
            '<script src="app.js"></script></body></html>'
        ),
        "style.css": "body { color: navy; }",
        "app.js": "document.querySelector('#action').addEventListener('click', () => {});",
    }

    monkeypatch.setattr(graph_module, "llm", MockModel())

    response = client.post(
        "/generate",
        json={"project_description": "Create a simple local webpage with a button"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    job_id = body["job_id"]
    job_dir = tmp_path / "jobs" / job_id
    assert body["status"] == "completed"
    assert body["project"]["name"] == "Small page"
    assert set(body["generated_files"]) == set(contents)
    assert all((job_dir / name).read_text() == content for name, content in contents.items())
    assert (tmp_path / "jobs" / ".completed" / f"{job_id}.done").is_file()
    assert "event=node_entered node=planner" in caplog.text
    assert "event=node_exited node=architect" in caplog.text
    assert "event=coder_task_started node=coder task_index=1" in caplog.text
    assert "event=coder_iteration node=coder task_index=3 iteration=3" in caplog.text
    assert "event=tool_succeeded node=coder task_index=1" in caplog.text
    assert "tool_name=list_generated_files outcome=success" in caplog.text
    assert "tool_name=write_file outcome=success" in caplog.text
    assert "event=workflow_terminated" in caplog.text
    assert "termination_reason=all_tasks_complete" in caplog.text
    assert "sensitive tool arguments must not be logged" not in caplog.text
    assert contents["app.js"] not in caplog.text


@pytest.mark.parametrize(
    ("state_factory", "written_files"),
    [
        (lambda: generation_state(["index.html"], task_paths=[]), {}),
        (lambda: generation_state(["index.html"], status="RUNNING"), {"index.html": "<main>ok</main>"}),
        (lambda: generation_state(["index.html"], completed_index=0), {"index.html": "<main>ok</main>"}),
        (lambda: generation_state(["index.html", "app.js"]), {"index.html": "<main>ok</main>"}),
        (lambda: generation_state(["index.html"]), {"index.html": " \n\t "}),
        (lambda: generation_state(["../outside.html"]), {}),
    ],
    ids=[
        "empty-steps",
        "graph-not-done",
        "coder-step-unfinished",
        "missing-planned-file",
        "whitespace-only-file",
        "path-traversal",
    ],
)
def test_incomplete_generation_is_not_marked_or_accessible(
    monkeypatch, tmp_path, state_factory, written_files
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        for path, content in written_files.items():
            write_file.invoke({"path": path, "content": content})
        return state_factory()

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)


def test_incomplete_structured_model_output_is_not_marked_or_accessible(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def incomplete_invoke(_state, _config):
        raise api_main.StructuredOutputIncomplete(
            "Architect response did not confirm structured output completion"
        )

    monkeypatch.setattr(api_main.agent, "invoke", incomplete_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert "exception_type=StructuredOutputIncomplete" in caplog.text


def test_generation_requires_a_usable_implementation_file(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "README.md", "content": "# Documentation"})
        return generation_state(["README.md"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)


def test_api_request_runs_bounded_cleanup_before_generation(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    stale_id = uuid4().hex
    stale_job = jobs_root / stale_id
    stale_job.mkdir(parents=True)
    (stale_job / "partial.txt").write_text("partial")
    import os
    import time

    os.utime(stale_job, (time.time() - 600, time.time() - 600))
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    monkeypatch.setattr(api_main, "ORPHAN_JOB_RETENTION_SECONDS", 60)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.py", "content": "print('ready')"})
        return generation_state(["app.py"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    response = client.post(
        "/generate", json={"project_description": "Build a cleanup test app"}
    )

    assert response.status_code == 200
    assert not stale_job.exists()
    assert (jobs_root / response.json()["job_id"] / "app.py").exists()


def test_generation_rejects_planned_symlink_escape(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    outside_file = tmp_path / "outside.html"
    outside_file.write_text("<main>outside the job</main>")
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        job_root = current_project_root()
        (job_root / "index.html").symlink_to(outside_file)
        return generation_state(["index.html"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)


def test_blocking_output_validation_failure_does_not_mark_job_complete(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke(
            {
                "path": "index.html",
                "content": "<html><body><script src='missing.js'></script></body></html>",
            }
        )
        return generation_state(["index.html"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: RepairResponse(
            files=[
                RepairFile(
                    path="index.html",
                    content="<html><body><script src='missing.js'></script></body></html>",
                )
            ]
        ),
    )
    caplog.set_level(logging.ERROR, logger="api.main")
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert "required_reference_missing" in caplog.text
    assert "missing.js" not in caplog.text


def test_missing_static_javascript_import_never_gets_completion_marker(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke(
            {"path": "app.js", "content": 'import helper from "./missing.js";'}
        )
        return generation_state(["app.js"])

    def ineffective_repair(context):
        assert context["findings"][0]["code"] == "required_reference_missing"
        return RepairResponse(
            files=[
                RepairFile(
                    path="app.js",
                    content='import helper from "./missing.js";',
                )
            ]
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", ineffective_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)


def test_generation_repairs_a_blocking_finding_once_and_completes(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    repair_calls = []

    def fake_invoke(_state, _config):
        write_file.invoke(
            {
                "path": "index.html",
                "content": "<html><body><script src='typo.js'></script></body></html>",
            }
        )
        write_file.invoke({"path": "app.js", "content": "document.body.dataset.ready = 'yes';"})
        return generation_state(["index.html", "app.js"])

    def fake_repair(context):
        repair_calls.append(context)
        assert [finding["code"] for finding in context["findings"]] == [
            "required_reference_missing"
        ]
        assert {file["path"] for file in context["files"]} == {
            "index.html",
            "typo.js",
        }
        assert "typo.js" in next(
            file["content"] for file in context["files"] if file["path"] == "index.html"
        )
        return RepairResponse(
            files=[
                RepairFile(
                    path="typo.js",
                    content="document.body.dataset.repaired = 'yes';",
                )
            ]
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", fake_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert response.status_code == 200
    assert len(repair_calls) == 1
    job_id = response.json()["job_id"]
    assert (jobs_root / job_id / "typo.js").read_text() == "document.body.dataset.repaired = 'yes';"
    assert (jobs_root / ".completed" / f"{job_id}.done").exists()


def test_failed_repair_leaves_job_unmarked(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    repair_calls = []

    def failing_repair(_context):
        repair_calls.append(True)
        raise RuntimeError("private repair provider detail")

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", failing_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert len(repair_calls) == 1
    assert "private repair provider detail" not in response.text


def test_remaining_blocking_error_does_not_trigger_a_second_repair(
    monkeypatch, tmp_path
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    repair_calls = []

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    def ineffective_repair(_context):
        repair_calls.append(True)
        return RepairResponse(
            files=[RepairFile(path="app.js", content="const value = ;")]
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", ineffective_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert len(repair_calls) == 1


def test_repair_rejects_unsafe_model_paths(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    outside_file = tmp_path / "outside.txt"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: RepairResponse(
            files=[RepairFile(path="../outside.txt", content="must not be written")]
        ),
    )

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert not outside_file.exists()


def test_repair_rechecks_symlinks_before_writing(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    outside_file = tmp_path / "outside.js"
    outside_file.write_text("outside job contents")
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    def insert_symlink(_context):
        job_dirs = [path for path in jobs_root.iterdir() if path.is_dir()]
        generated_file = job_dirs[0] / "app.js"
        generated_file.unlink()
        generated_file.symlink_to(outside_file)
        return RepairResponse(
            files=[RepairFile(path="app.js", content="must not overwrite outside file")]
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", insert_symlink)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert outside_file.read_text() == "outside job contents"


def test_repair_skips_oversized_input(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    repair_calls = []
    source = "//" + (" padding" * (api_main.MAX_REPAIR_CONTEXT_BYTES // 8)) + "\nconst value = ;"

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": source})
        return generation_state(["app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: RepairResponse(
            files=[RepairFile(path="app.js", content=source)]
        ),
    )
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda context: repair_calls.append(context),
    )

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert response.status_code == 422
    job_dirs = [
        path
        for path in jobs_root.iterdir()
        if path.is_dir() and path.name not in {".completed", ".active"}
    ]
    assert len(job_dirs) == 1
    job_id = job_dirs[0].name
    assert not (jobs_root / ".completed" / f"{job_id}.done").exists()
    assert client.get(f"/jobs/{job_id}/files").status_code == 404
    assert repair_calls == []


def test_repair_rejects_oversized_output(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    repair_calls = []

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    def oversized_repair(_context):
        repair_calls.append(True)
        return RepairResponse(
            files=[
                RepairFile(
                    path="app.js",
                    content="x" * (api_main.MAX_REPAIR_OUTPUT_BYTES + 1),
                )
            ]
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", oversized_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert len(repair_calls) == 1


def test_rate_limit_during_repair_returns_safe_429_without_retry(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    repair_calls = []
    request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    provider_response = Response(
        status_code=429,
        headers={"retry-after": "5", "x-ratelimit-reset-tokens": "9s"},
        request=request,
    )

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const value = ;"})
        return generation_state(["app.js"])

    def rate_limited_repair(_context):
        repair_calls.append(True)
        raise RateLimitError(
            "private quota details",
            response=provider_response,
            body={"error": {"message": "private quota details"}},
        )

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(api_main, "repair_generated_files", rate_limited_repair)

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["message"] == api_main.RATE_LIMIT_DETAIL
    assert detail["error_type"] == "RateLimitError"
    assert detail["category"] == "rate_limit"
    assert detail["retry_after_seconds"] == 5
    assert detail["rate_limit_reset_seconds"] == {"reset_tokens_seconds": 9}
    assert "private quota details" not in response.text
    assert "private quota details" not in caplog.text
    assert "stage=repair" in caplog.text
    assert "http_status=429" in caplog.text
    assert len(repair_calls) == 1
    job_dirs = [
        path
        for path in jobs_root.iterdir()
        if path.is_dir() and path.name not in {".completed", ".active"}
    ]
    assert not (jobs_root / ".completed" / f"{job_dirs[0].name}.done").exists()


def test_optional_asset_validation_warning_does_not_block_completion(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def fake_invoke(_state, _config):
        write_file.invoke(
            {"path": "index.html", "content": "<html><body><img src='missing.png'></body></html>"}
        )
        write_file.invoke({"path": "app.js", "content": "document.body.dataset.ready = 'yes';"})
        return generation_state(["index.html", "app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: pytest.fail("warnings must not trigger repair"),
    )
    caplog.set_level(logging.WARNING, logger="api.main")
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert (jobs_root / ".completed" / f"{job_id}.done").exists()
    assert "optional_asset_missing" in caplog.text
    assert "missing.png" not in caplog.text


def test_invalid_javascript_fails_safely_before_completion(monkeypatch, tmp_path, caplog):
    jobs_root = tmp_path / "jobs"
    source = "const privateMarkerForTest = ;"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    monkeypatch.setattr("api.output_validation.shutil.which", lambda _name: "/mock/node")
    monkeypatch.setattr(
        "api.output_validation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": source})
        return generation_state(["app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: RepairResponse(
            files=[RepairFile(path="app.js", content=source)]
        ),
    )
    caplog.set_level(logging.ERROR, logger="api.main")
    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert "javascript_syntax_invalid" in caplog.text
    assert source not in caplog.text + response.text


@pytest.mark.parametrize(
    ("node_path", "parser_failure"),
    [
        (None, None),
        ("/mock/node", subprocess.TimeoutExpired("node", 5)),
        ("/mock/node", OSError("private parser detail")),
        ("/mock/node", subprocess.CompletedProcess(["node"], -9)),
    ],
    ids=["node-missing", "timeout", "cannot-start", "parser-failed"],
)
def test_unperformed_javascript_syntax_check_blocks_completion(
    monkeypatch, tmp_path, node_path, parser_failure
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    monkeypatch.setattr("api.output_validation.shutil.which", lambda _name: node_path)

    def parser_call(*_args, **_kwargs):
        if parser_failure is not None:
            raise parser_failure if isinstance(parser_failure, BaseException) else AssertionError()
        return parser_failure

    monkeypatch.setattr("api.output_validation.subprocess.run", parser_call)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.js", "content": "const working = true;"})
        return generation_state(["app.js"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: pytest.fail("parser availability failures are not repairable"),
    )

    response = client.post(
        "/generate",
        json={"project_description": "Build a safe mocked application"},
    )

    assert_failed_job_is_not_accessible(response, jobs_root)
    assert "private parser detail" not in response.text


def test_file_tools_are_isolated_by_project_root(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    with use_project_root(first_root):
        write_file.invoke({"path": "only-first.txt", "content": "first"})
    with use_project_root(second_root):
        write_file.invoke({"path": "only-second.txt", "content": "second"})

    assert (first_root / "only-first.txt").read_text() == "first"
    assert not (first_root / "only-second.txt").exists()
    assert (second_root / "only-second.txt").read_text() == "second"
    assert not (second_root / "only-first.txt").exists()


def test_generation_failure_logs_only_allowlisted_metadata_and_returns_safe_error(
    monkeypatch, tmp_path, caplog
):
    prompt = "Build this sensitive project request"
    sensitive_details = "Provider failure with org_private token=unusual-secret-value"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def failing_invoke(state, config):
        raise RuntimeError(f"{sensitive_details}; prompt={prompt}")

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Generation failed. Check the backend configuration and try again."
    }
    assert "exception_type=RuntimeError" in caplog.text
    assert "job_id=" in caplog.text
    assert prompt not in caplog.text
    assert sensitive_details not in caplog.text
    assert "Provider failure" not in caplog.text
    assert "traceback" not in caplog.text.lower()


def test_groq_rate_limit_preserves_safe_metadata_and_logs_stage_without_details(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    provider_details = "org=org_private key=gsk_private_key raw provider traceback"
    request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    provider_response = Response(
        status_code=429,
        headers={
            "retry-after": "7",
            "x-ratelimit-reset-requests": "13s",
            "x-ratelimit-reset-tokens": "2m",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-remaining-tokens": "41",
        },
        request=request,
    )

    def rate_limited_invoke(state, config):
        error = RateLimitError(
            provider_details,
            response=provider_response,
            body={"error": {"type": "rate_limit_error", "message": provider_details}},
        )
        error.add_note("LangGraph node: planner")
        raise error

    monkeypatch.setattr(api_main.agent, "invoke", rate_limited_invoke)
    monkeypatch.setattr(
        api_main,
        "repair_generated_files",
        lambda _context: pytest.fail("quota failures must not trigger repair"),
    )
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate",
        json={"project_description": "Build a personal to-do application"},
    )

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail == {
        "message": api_main.RATE_LIMIT_DETAIL,
        "error_type": "rate_limit_error",
        "category": "rate_limit",
        "retry_after_seconds": 7,
        "rate_limit_reset_seconds": {
            "reset_requests_seconds": 13,
            "reset_tokens_seconds": 120,
        },
    }
    assert provider_details not in response.text
    assert "org_private" not in response.text
    assert "gsk_private_key" not in response.text
    assert "traceback" not in response.text.lower()
    assert provider_details not in caplog.text
    assert "stage=Planner" in caplog.text
    assert "http_status=429" in caplog.text
    assert "category=rate_limit" in caplog.text
    assert "reset_requests_seconds=13" in caplog.text
    assert "reset_tokens_seconds=120" in caplog.text
    assert "remaining_requests=0" in caplog.text
    assert "remaining_tokens=41" in caplog.text
    assert "RPM" not in response.text
    assert "TPM" not in response.text
    assert "RPD" not in response.text
    assert "TPD" not in response.text


def test_wrapped_coder_rate_limit_keeps_provider_status_and_safe_metadata(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    provider_response = Response(
        status_code=429,
        headers={
            "retry-after": "4",
            "x-ratelimit-reset-requests": "8s",
        },
        request=request,
    )
    provider_detail = "raw provider body containing a sensitive prompt fragment"

    def fake_coder_failure(_state, _config):
        try:
            raise RateLimitError(
                provider_detail,
                response=provider_response,
                body={"error": {"type": "rate_limit_error", "message": provider_detail}},
            )
        except RateLimitError as provider_error:
            wrapped = RuntimeError("untrusted wrapper message")
            wrapped.add_note("LangGraph node: coder")
            raise wrapped from provider_error

    monkeypatch.setattr(api_main.agent, "invoke", fake_coder_failure)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate", json={"project_description": "Build a useful local application"}
    )

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "message": api_main.RATE_LIMIT_DETAIL,
        "error_type": "rate_limit_error",
        "category": "rate_limit",
        "retry_after_seconds": 4,
        "rate_limit_reset_seconds": {"reset_requests_seconds": 8},
    }
    assert provider_detail not in response.text
    assert provider_detail not in caplog.text
    assert "untrusted wrapper message" not in caplog.text
    assert "stage=Coder" in caplog.text
    assert "http_status=429" in caplog.text
    assert "category=rate_limit" in caplog.text


def test_unexpected_coder_exception_logs_safe_class_chain_and_stays_500(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    private_message = "private prompt/source detail that must not be logged"

    class UnexpectedCoderFailure(Exception):
        pass

    def fake_coder_failure(_state, _config):
        try:
            raise ValueError(private_message)
        except ValueError as cause:
            wrapped = UnexpectedCoderFailure(private_message)
            wrapped.add_note("LangGraph node: coder")
            raise wrapped from cause

    monkeypatch.setattr(api_main.agent, "invoke", fake_coder_failure)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate", json={"project_description": "Build a useful local application"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Generation failed. Check the backend configuration and try again."
    }
    assert private_message not in response.text
    assert private_message not in caplog.text
    assert "node=coder" in caplog.text
    assert "exception_type=OtherError" in caplog.text
    assert "exception_class=test_api.test_unexpected_coder_exception_logs_safe_class_chain_and_stays_500.<locals>.UnexpectedCoderFailure" in caplog.text
    assert "builtins.ValueError" in caplog.text
    assert "category=rate_limit" not in caplog.text
    assert "http_status=429" not in caplog.text
    assert_failed_job_is_not_accessible(response, jobs_root)


def test_coder_graph_recursion_failure_fails_safely_without_completion(
    monkeypatch, tmp_path, caplog
):
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def recursion_limited_invoke(_state, _config):
        record_graph_node_execution("coder")
        error = GraphRecursionError("internal graph state detail")
        error.add_note("LangGraph node: coder")
        raise error

    monkeypatch.setattr(api_main.agent, "invoke", recursion_limited_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate", json={"project_description": "Build a useful local application"}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "message": (
                "Generation reached a bounded workflow step limit. "
                "Try a smaller or simpler project description."
            ),
            "category": "workflow_limit",
            "error_type": "graph_recursion_limit",
            "stage": "Coder",
            "configured_limit": MAX_GRAPH_RECURSION_STEPS,
            "configured_limit_unit": "graph_supersteps",
            "observed_node_steps": 1,
        }
    }
    assert "internal graph state detail" not in response.text
    assert "internal graph state detail" not in caplog.text
    assert "stage=Coder" in caplog.text
    assert "error_type=graph_recursion_limit" in caplog.text
    assert "category=rate_limit" not in caplog.text
    assert_failed_job_is_not_accessible(response, jobs_root, expected_status=422)


def test_coder_agent_step_limit_has_safe_api_category_and_message(
    monkeypatch, tmp_path, caplog
):
    from agent.runtime import CoderAgentStepLimitExceeded

    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def coder_limit_invoke(_state, _config):
        raise CoderAgentStepLimitExceeded(MAX_CODER_AGENT_STEPS, 8)

    monkeypatch.setattr(api_main.agent, "invoke", coder_limit_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")
    response = client.post(
        "/generate", json={"project_description": "Build a useful local application"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["category"] == "workflow_limit"
    assert detail["error_type"] == "coder_step_limit"
    assert detail["stage"] == "Coder"
    assert detail["configured_limit"] == MAX_CODER_AGENT_STEPS
    assert detail["configured_limit_unit"] == "model_iterations"
    assert detail["observed_model_iterations"] == 8
    assert "bounded workflow step limit" in detail["message"]
    assert "internal per-task state details" not in response.text
    assert "internal per-task state details" not in caplog.text
    assert "stage=Coder category=workflow_limit error_type=coder_step_limit" in caplog.text
    assert_failed_job_is_not_accessible(response, jobs_root, expected_status=422)


def test_inner_react_graph_limit_is_distinct_from_model_iteration_limit(
    monkeypatch, tmp_path
):
    from agent.runtime import CoderReActRecursionLimitExceeded

    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    def react_recursion_limited(_state, _config):
        raise CoderReActRecursionLimitExceeded(
            MAX_CODER_AGENT_RECURSION_STEPS, 3
        )

    monkeypatch.setattr(api_main.agent, "invoke", react_recursion_limited)
    response = client.post(
        "/generate", json={"project_description": "Build a small local page"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["category"] == "workflow_limit"
    assert detail["error_type"] == "coder_react_recursion_limit"
    assert detail["stage"] == "Coder"
    assert detail["configured_limit"] == MAX_CODER_AGENT_RECURSION_STEPS
    assert detail["configured_limit_unit"] == "graph_supersteps"
    assert detail["observed_model_iterations"] == 3
    assert_failed_job_is_not_accessible(response, jobs_root, expected_status=422)


def test_provider_status_and_safe_category_are_preserved_without_raw_message(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    provider_response = Response(status_code=401, request=request)
    unsafe_message = "authorization Bearer private-token and user prompt contents"

    def unauthorized_invoke(_state, _config):
        raise APIStatusError(
            unsafe_message,
            response=provider_response,
            body={"error": {"type": "authentication_error", "message": unsafe_message}},
        )

    monkeypatch.setattr(api_main.agent, "invoke", unauthorized_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate", json={"project_description": "Build a useful local application"}
    )

    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["error_type"] == "authentication_error"
    assert detail["category"] == "authentication"
    assert "backend authentication" in detail["message"]
    assert unsafe_message not in response.text
    assert unsafe_message not in caplog.text
    assert "http_status=401" in caplog.text
    assert "category=authentication" in caplog.text


def test_completed_job_download_contains_only_allowed_files_and_is_valid_zip(
    monkeypatch, tmp_path
):
    job_id, job_root = make_completed_job(
        monkeypatch,
        tmp_path,
        {
            "README.md": "# Sample project",
            "index.html": "<main>Safe</main>",
            ".env": "GROQ_API_KEY=never-include",
            "settings/.env.local": "PRIVATE=value",
            ".venv/lib/site.py": "environment file",
            "node_modules/example/index.js": "dependency file",
            "cache/generated.js": "cache file",
            "credentials.json": '{"token":"private"}',
            "secrets.yaml": "service_password: private",
            "service-account.json": '{"private_key":"private"}',
            "access-token.json": '{"access_token":"private"}',
            "bundle.zip": "nested archive",
            "src/embedded-key.js": 'const key = "gsk_abcdefghijklmnopqrstuvwxyz123456";',
            "src/key-notes.js": "-----BEGIN PRIVATE KEY-----\\nexample\\n-----END PRIVATE KEY-----",
            "image.png": b"\x89PNG\x00binary",
        },
    )
    outside_file = tmp_path / "outside.js"
    outside_file.write_text("outside the job")
    (job_root / "escape.js").symlink_to(outside_file)
    outside_dir = tmp_path / "outside-assets"
    outside_dir.mkdir()
    (outside_dir / "outside.css").write_text("body { color: red; }")
    (job_root / "linked-assets").symlink_to(outside_dir, target_is_directory=True)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="thought2code-ai-{job_id}.zip"'
    )
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"README.md", "index.html"}
        assert archive.read("README.md") == b"# Sample project"
        assert archive.read("index.html") == b"<main>Safe</main>"
        assert not any(name.startswith(".completed/") for name in archive.namelist())


def test_download_rejects_invalid_missing_and_incomplete_jobs(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    incomplete_id = uuid4().hex
    (jobs_root / incomplete_id).mkdir(parents=True)
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    invalid = client.get("/jobs/not-a-valid-id/download")
    missing = client.get(f"/jobs/{uuid4().hex}/download")
    incomplete = client.get(f"/jobs/{incomplete_id}/download")

    assert invalid.status_code == 400
    assert missing.status_code == 404
    assert incomplete.status_code == 404


def test_download_job_identifier_does_not_accept_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    response = client.get("/jobs/%2E%2E%2Foutside/download")

    assert response.status_code in {400, 404}


def test_download_enforces_total_uncompressed_size_limit(monkeypatch, tmp_path):
    job_id, _ = make_completed_job(
        monkeypatch, tmp_path, {"README.md": "project"}
    )
    monkeypatch.setattr(api_main, "MAX_DOWNLOAD_BYTES", 3)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 413
    assert response.json() == {
        "detail": "This project is too large to download safely."
    }


def test_download_rechecks_actual_file_size_after_listing(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(
        monkeypatch, tmp_path, {"README.md": "initial"}
    )
    original_list = api_main._completed_job_files

    def list_then_grow(path):
        files = original_list(path)
        (path / "README.md").write_bytes(
            b"x" * (api_main.MAX_VIEWABLE_FILE_BYTES + 1)
        )
        return files

    monkeypatch.setattr(api_main, "_completed_job_files", list_then_grow)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == []


def test_download_rejects_symlink_inserted_after_path_validation(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "inside project"}
    )
    outside_file = tmp_path / "outside.html"
    outside_file.write_text("outside project")
    original_safe_path = api_main._safe_job_file

    def swap_for_symlink_after_validation(root, requested_path):
        resolved = original_safe_path(root, requested_path)
        if requested_path == "index.html":
            resolved.unlink()
            resolved.symlink_to(outside_file)
        return resolved

    monkeypatch.setattr(api_main, "_safe_job_file", swap_for_symlink_after_validation)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == []


def test_download_rejects_duplicate_archive_entry_names(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "inside project"}
    )
    file_info = api_main.GeneratedFileInfo(
        path="index.html",
        size_bytes=len("inside project"),
        language="html",
        viewable=True,
    )
    monkeypatch.setattr(api_main, "_completed_job_files", lambda _root: [file_info, file_info])

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 500
    assert response.json() == {"detail": "This project contains duplicate file paths."}


def test_download_entry_filter_rejects_unsafe_paths():
    unsafe_paths = (
        "../outside.js",
        "/absolute.js",
        "C:/outside.js",
        "folder\\outside.js",
        "folder/file:stream.js",
        "folder/../outside.js",
    )
    for path in unsafe_paths:
        file_info = api_main.GeneratedFileInfo(
            path=path,
            size_bytes=1,
            language="javascript",
            viewable=True,
        )
        assert not api_main._archive_path_is_allowed(file_info)


def test_content_secret_check_catches_specific_credentials_without_flagging_generic_config_text():
    assert api_main._content_looks_sensitive(
        b"-----BEGIN PRIVATE KEY-----\\nexample\\n-----END PRIVATE KEY-----"
    )
    assert api_main._content_looks_sensitive(
        b"const key = 'gsk_abcdefghijklmnopqrstuvwxyz123456';"
    )
    assert not api_main._content_looks_sensitive(
        b"const apiKey = 'YOUR_API_KEY'; // Add a key to configure the application"
    )


@pytest.mark.parametrize(
    "content",
    [
        b"sk-proj-" + b"a" * 32,
        b"xoxb-" + b"a" * 24,
        b"eyJ" + b"a" * 16 + b"." + b"b" * 16 + b"." + b"c" * 16,
    ],
)
def test_content_secret_filter_covers_additional_common_token_shapes(content):
    assert api_main._content_looks_sensitive(content)


@pytest.mark.parametrize(
    "path",
    [
        ".aws/credentials",
        ".ssh/id_ecdsa",
        ".azure/profile.json",
        ".gcloud/application_default_credentials.json",
        "credentials.ini",
        "service-account-key.json",
        "firebase-adminsdk.json",
    ],
)
def test_archive_filter_excludes_additional_credential_paths(path):
    file_info = api_main.GeneratedFileInfo(
        path=path,
        size_bytes=1,
        language="text",
        viewable=True,
    )
    assert not api_main._archive_path_is_allowed(file_info)


def test_download_stream_closes_spooled_archive(monkeypatch, tmp_path):
    job_id, _ = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "inside project"}
    )
    created_archives = []
    original_spooled_file = api_main.tempfile.SpooledTemporaryFile

    def tracked_spooled_file(*args, **kwargs):
        archive_file = original_spooled_file(*args, **kwargs)
        created_archives.append(archive_file)
        return archive_file

    monkeypatch.setattr(api_main.tempfile, "SpooledTemporaryFile", tracked_spooled_file)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    assert len(created_archives) == 1
    assert created_archives[0].closed


def test_generation_failure_log_includes_graph_node_without_prompt(
    monkeypatch, tmp_path, caplog
):
    prompt = "Build a private application idea with confidential details"
    secret_message = f"Coder execution failed for {prompt}; token=private-provider-detail"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def failing_invoke(state, config):
        error = RuntimeError(secret_message)
        error.add_note("LangGraph node: coder")
        raise error

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert "node=coder exception_type=RuntimeError" in caplog.text
    assert prompt not in caplog.text
    assert secret_message not in caplog.text
    assert "private-provider-detail" not in caplog.text


def test_generation_logging_rejects_unrecognized_exception_names_and_nodes(
    monkeypatch, tmp_path, caplog
):
    sensitive_class_name = "RuntimeError;token=class-name-secret"
    sensitive_message = "provider response with org-secret and prompt-secret"
    unusual_exception = type(sensitive_class_name, (Exception,), {})
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def failing_invoke(state, config):
        error = unusual_exception(sensitive_message)
        error.add_note("LangGraph node: coder;token=note-secret")
        raise error

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate", json={"project_description": "Build a secure local application"}
    )

    assert response.status_code == 500
    assert "node=unknown exception_type=OtherError" in caplog.text
    assert sensitive_class_name not in caplog.text
    assert sensitive_message not in caplog.text
    assert "org-secret" not in caplog.text
    assert "note-secret" not in caplog.text


def test_local_generation_diagnostics_are_disabled_by_default(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("THOUGHT2CODE_ENV", raising=False)
    monkeypatch.delenv("THOUGHT2CODE_LOCAL_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    assert not api_main._local_diagnostics_enabled(make_http_request())
    caplog.set_level(logging.INFO, logger="thought2code.diagnostics")
    caplog.set_level(logging.ERROR, logger="api.main")

    prompt = "A local mocked generation failure"
    def failing_invoke(state, config):
        raise RuntimeError("mocked failure")

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert "Local generation diagnostic" not in caplog.text
    assert "Generation diagnostic" not in caplog.text


def test_local_generation_diagnostics_require_local_loopback_and_no_hosted_runtime(
    monkeypatch,
):
    monkeypatch.setenv("THOUGHT2CODE_ENV", "local")
    monkeypatch.setenv("THOUGHT2CODE_LOCAL_DIAGNOSTICS", "1")

    assert api_main._local_diagnostics_enabled(make_http_request("::1"))
    assert not api_main._local_diagnostics_enabled(make_http_request("192.0.2.4"))
    assert not api_main._local_diagnostics_enabled(
        make_http_request(server_host="0.0.0.0")
    )

    monkeypatch.setenv("K_SERVICE", "hosted-runtime")
    assert not api_main._local_diagnostics_enabled(make_http_request())


def test_local_generation_diagnostics_log_safe_frames_and_exception_classes_only(
    monkeypatch, tmp_path, caplog
):
    prompt = "private prompt detail that must not be logged"
    exception_message = "provider body contains api_key=private-secret and raw response"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api_main, "_local_diagnostics_enabled", lambda _request: True)

    class ProviderFailure(Exception):
        pass

    def failing_invoke(state, config):
        try:
            raise ValueError(exception_message)
        except ValueError as cause:
            error = ProviderFailure(exception_message)
            error.add_note("LangGraph node: coder")
            raise error from cause

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Generation failed. Check the backend configuration and try again."
    }
    assert "Local generation diagnostic job_id=" in caplog.text
    assert "node=coder" in caplog.text
    assert "ProviderFailure" in caplog.text
    assert "ValueError" in caplog.text
    assert "failing_invoke" in caplog.text
    assert prompt not in caplog.text and prompt not in response.text
    assert exception_message not in caplog.text and exception_message not in response.text
    assert "private-secret" not in caplog.text + response.text
    assert "raw response" not in caplog.text + response.text
    assert "Traceback" not in caplog.text


def test_missing_graph_plan_is_reported_as_failure_with_safe_state_diagnostics(
    monkeypatch, tmp_path, caplog
):
    prompt = "Build a project with enough detail"
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda state, config: {"coder_state": None, "status": "DONE"},
    )
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert response.json()["detail"].startswith("Generation failed.")
    assert "exception_type=ValueError" in caplog.text
    assert "result_keys" not in caplog.text
    assert "coder_state" not in caplog.text
    assert prompt not in caplog.text


def test_completed_job_file_reading_is_isolated(monkeypatch, tmp_path):
    first_id, _ = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "<h1>first job</h1>"}
    )
    jobs_root = api_main.PROJECTS_ROOT
    second_id = uuid4().hex
    second_root = jobs_root / second_id
    second_root.mkdir()
    (second_root / "index.html").write_text("<h1>second job</h1>")
    (jobs_root / ".completed" / f"{second_id}.done").touch()

    listing = client.get(f"/jobs/{first_id}/files")
    content = client.get(f"/jobs/{first_id}/files/index.html")
    cross_job_path = client.get(
        f"/jobs/{first_id}/files/%2E%2E%2F{second_id}%2Findex.html"
    )

    assert listing.status_code == 200
    assert listing.json()["job_id"] == first_id
    assert [file["path"] for file in listing.json()["files"]] == ["index.html"]
    assert content.status_code == 200
    assert content.json()["content"] == "<h1>first job</h1>"
    assert cross_job_path.status_code == 400
    assert second_id not in content.text


def test_job_directory_symlink_cannot_expose_another_completed_job(
    monkeypatch, tmp_path
):
    first_id, first_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "first"}
    )
    jobs_root = api_main.PROJECTS_ROOT
    second_id, second_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "JOB_B_PRIVATE_CONTENT", "image.png": b"image"}
    )
    assert second_id != first_id
    shutil.rmtree(first_root)
    first_root.symlink_to(second_root, target_is_directory=True)

    responses = [
        client.get(f"/jobs/{first_id}/files"),
        client.get(f"/jobs/{first_id}/files/index.html"),
        client.get(f"/jobs/{first_id}/preview-assets/image.png"),
        client.get(f"/jobs/{first_id}/download"),
    ]

    assert all(response.status_code == 404 for response in responses)
    assert all("JOB_B_PRIVATE_CONTENT" not in response.text for response in responses)
    assert str(jobs_root) not in "".join(response.text for response in responses)


def test_completed_marker_directory_symlink_into_job_is_rejected(
    monkeypatch, tmp_path
):
    first_id, _first_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "first"}
    )
    jobs_root = api_main.PROJECTS_ROOT
    _second_id, second_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "JOB_B_PRIVATE_CONTENT"}
    )
    shutil.rmtree(jobs_root / ".completed")
    (jobs_root / ".completed").symlink_to(second_root, target_is_directory=True)
    # Recreate the marker the old resolved-parent check would have accepted.
    (second_root / f"{first_id}.done").touch()

    responses = [
        client.get(f"/jobs/{first_id}/files"),
        client.get(f"/jobs/{first_id}/files/index.html"),
        client.get(f"/jobs/{first_id}/download"),
    ]

    assert all(response.status_code == 404 for response in responses)
    assert all("JOB_B_PRIVATE_CONTENT" not in response.text for response in responses)


@pytest.mark.parametrize("marker_kind", ["symlink", "directory"])
def test_completion_marker_must_be_a_regular_file(
    monkeypatch, tmp_path, marker_kind
):
    job_id, job_root = make_completed_job(
        monkeypatch, tmp_path, {"index.html": "JOB_CONTENT"}
    )
    marker = api_main.PROJECTS_ROOT / ".completed" / f"{job_id}.done"
    marker.unlink()
    if marker_kind == "symlink":
        marker.symlink_to(job_root, target_is_directory=True)
    else:
        marker.mkdir()

    responses = [
        client.get(f"/jobs/{job_id}/files"),
        client.get(f"/jobs/{job_id}/files/index.html"),
        client.get(f"/jobs/{job_id}/download"),
    ]

    assert all(response.status_code == 404 for response in responses)
    assert all("JOB_CONTENT" not in response.text for response in responses)


def test_generation_does_not_write_marker_through_completed_directory_symlink(
    monkeypatch, tmp_path
):
    _second_id, second_root = make_completed_job(
        monkeypatch, tmp_path, {"README.md": "existing job"}
    )
    jobs_root = api_main.PROJECTS_ROOT
    shutil.rmtree(jobs_root / ".completed")
    (jobs_root / ".completed").symlink_to(second_root, target_is_directory=True)

    def fake_invoke(_state, _config):
        write_file.invoke({"path": "app.py", "content": "print('ready')"})
        return generation_state(["app.py"])

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)

    response = client.post(
        "/generate", json={"project_description": "Build a safe sample application"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Generation failed. Check the backend configuration and try again."
    }
    assert list(second_root.glob("*.done")) == []


def test_file_endpoint_rejects_traversal_and_absolute_paths(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"safe.txt": "safe"})

    for requested in ("../outside.txt", "/etc/passwd", "C:/Windows/win.ini", "..\\outside.txt"):
        try:
            api_main._safe_job_file(job_root, requested)
        except HTTPException as error:
            assert error.status_code == 400
        else:
            raise AssertionError(f"Unsafe path was accepted: {requested}")

    traversal_request = client.get(
        f"/jobs/{job_id}/files/%2E%2E%2Foutside.txt"
    )
    assert traversal_request.status_code == 400


def test_file_endpoint_returns_404_for_missing_job_or_file(monkeypatch, tmp_path):
    job_id, _ = make_completed_job(monkeypatch, tmp_path, {})

    missing_file = client.get(f"/jobs/{job_id}/files/missing.md")
    missing_job = client.get(f"/jobs/{uuid4().hex}/files")

    assert missing_file.status_code == 404
    assert missing_job.status_code == 404
    assert str(tmp_path) not in missing_file.text


def test_unmarked_job_is_not_treated_as_completed(monkeypatch, tmp_path):
    jobs_root = tmp_path / "jobs"
    job_id = uuid4().hex
    (jobs_root / job_id).mkdir(parents=True)
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)

    response = client.get(f"/jobs/{job_id}/files")

    assert response.status_code == 404
    assert str(tmp_path) not in response.text


def test_file_endpoint_rejects_symlink_escape(monkeypatch, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside job")
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {})
    link = job_root / "escape.txt"
    link.symlink_to(outside)

    response = client.get(f"/jobs/{job_id}/files/escape.txt")

    assert response.status_code == 400
    assert str(tmp_path) not in response.text


def test_file_endpoint_rejects_symlink_swapped_after_validation(
    monkeypatch, tmp_path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("must never be returned")
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"safe.txt": "inside"})
    original_open = api_main.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "safe.txt" and kwargs.get("dir_fd") is not None and not swapped:
            (job_root / "safe.txt").unlink()
            (job_root / "safe.txt").symlink_to(outside)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(api_main.os, "open", swap_before_open)
    monkeypatch.setattr(api_main, "_secure_archive_open_supported", lambda: True)

    response = client.get(f"/jobs/{job_id}/files/safe.txt")

    assert response.status_code == 404
    assert "must never be returned" not in response.text
    assert str(tmp_path) not in response.text


def test_file_endpoint_reads_open_descriptor_if_path_is_swapped_after_open(
    monkeypatch, tmp_path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside contents")
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"safe.txt": "inside contents"})
    original_open = api_main.os.open
    swapped = False

    def swap_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "safe.txt" and kwargs.get("dir_fd") is not None and not swapped:
            (job_root / "safe.txt").unlink()
            (job_root / "safe.txt").symlink_to(outside)
            swapped = True
        return descriptor

    monkeypatch.setattr(api_main.os, "open", swap_after_open)
    monkeypatch.setattr(api_main, "_secure_archive_open_supported", lambda: True)

    response = client.get(f"/jobs/{job_id}/files/safe.txt")

    assert response.status_code == 200
    assert response.json()["content"] == "inside contents"
    assert "outside contents" not in response.text


def test_file_endpoint_rejects_nonregular_file_inserted_before_open(
    monkeypatch, tmp_path
):
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"safe.txt": "inside"})
    original_open = api_main.os.open
    swapped = False

    def swap_for_fifo(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "safe.txt" and kwargs.get("dir_fd") is not None and not swapped:
            (job_root / "safe.txt").unlink()
            os.mkfifo(job_root / "safe.txt")
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    import os

    monkeypatch.setattr(api_main.os, "open", swap_for_fifo)
    monkeypatch.setattr(api_main, "_secure_archive_open_supported", lambda: True)

    response = client.get(f"/jobs/{job_id}/files/safe.txt")

    assert response.status_code == 400
    assert "fifo" not in response.text.lower()
    assert str(tmp_path) not in response.text


def test_preview_asset_endpoint_serves_small_allowlisted_asset_safely(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(
        monkeypatch, tmp_path, {"assets/logo.png": b"\x89PNG\r\n\x1a\nimage"}
    )

    response = client.get(f"/jobs/{job_id}/preview-assets/assets/logo.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n\x1a\nimage"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_preview_asset_endpoint_rejects_traversal_missing_and_unsupported_files(
    monkeypatch, tmp_path
):
    job_id, _job_root = make_completed_job(
        monkeypatch, tmp_path, {"image.png": b"png", "page.html": "<main />"}
    )

    traversal = client.get(f"/jobs/{job_id}/preview-assets/%2E%2E/outside.png")
    missing = client.get(f"/jobs/{job_id}/preview-assets/missing.png")
    unsupported = client.get(f"/jobs/{job_id}/preview-assets/page.html")

    assert traversal.status_code in {400, 404}
    assert missing.status_code == 404
    assert unsupported.status_code == 400
    assert str(tmp_path) not in traversal.text + missing.text + unsupported.text


def test_preview_asset_endpoint_rejects_symlink_swap_and_executable_files(
    monkeypatch, tmp_path
):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside bytes")
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"assets/image.png": b"inside"})
    original_open = api_main.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "image.png" and kwargs.get("dir_fd") is not None and not swapped:
            target = job_root / "assets" / "image.png"
            target.unlink()
            target.symlink_to(outside)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(api_main.os, "open", swap_before_open)
    monkeypatch.setattr(api_main, "_secure_archive_open_supported", lambda: True)
    response = client.get(f"/jobs/{job_id}/preview-assets/assets/image.png")
    assert response.status_code == 404
    assert b"outside bytes" not in response.content

    monkeypatch.setattr(api_main.os, "open", original_open)
    (job_root / "assets" / "image.png").unlink()
    executable = job_root / "assets" / "run.png"
    executable.write_bytes(b"not executable content")
    executable.chmod(0o755)
    response = client.get(f"/jobs/{job_id}/preview-assets/assets/run.png")
    assert response.status_code == 400


def test_preview_asset_endpoint_enforces_size_on_open_descriptor(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(monkeypatch, tmp_path, {"large.png": b"small"})
    monkeypatch.setattr(api_main, "MAX_PREVIEW_ASSET_BYTES", 4)

    response = client.get(f"/jobs/{job_id}/preview-assets/large.png")

    assert response.status_code == 413
    assert str(job_root) not in response.text


def test_file_endpoint_enforces_size_and_text_file_limits(monkeypatch, tmp_path):
    job_id, job_root = make_completed_job(
        monkeypatch,
        tmp_path,
        {
            "large.txt": b"x" * (api_main.MAX_VIEWABLE_FILE_BYTES + 1),
            "program.exe": b"MZ\x00\x01",
            "run.sh": "#!/bin/sh\nprintf unsafe",
        },
    )
    (job_root / "run.sh").chmod(0o755)

    oversized = client.get(f"/jobs/{job_id}/files/large.txt")
    executable = client.get(f"/jobs/{job_id}/files/program.exe")
    executable_script = client.get(f"/jobs/{job_id}/files/run.sh")
    listing = client.get(f"/jobs/{job_id}/files")

    assert oversized.status_code == 413
    assert executable.status_code == 400
    assert executable_script.status_code == 400
    listed = {file["path"]: file for file in listing.json()["files"]}
    assert listed["large.txt"]["viewable"] is False
    assert listed["program.exe"]["viewable"] is False
    assert listed["run.sh"]["viewable"] is False


def test_file_endpoint_rejects_non_utf8_text(monkeypatch, tmp_path):
    job_id, _ = make_completed_job(monkeypatch, tmp_path, {"notes.txt": b"\xff\xfe"})

    response = client.get(f"/jobs/{job_id}/files/notes.txt")

    assert response.status_code == 400
    assert str(tmp_path) not in response.text

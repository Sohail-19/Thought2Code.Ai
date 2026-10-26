import io
import logging
import zipfile
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
from groq import RateLimitError
from httpx import Request, Response
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest

import api.main as api_main
from agent.tools import use_project_root, write_file


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
    marker_root.mkdir()
    (marker_root / f"{job_id}.done").touch()
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", jobs_root)
    return job_id, job_root


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert api_main.app.title == "Thought2Code.Ai Local API"


def test_generate_rejects_invalid_description():
    response = client.post("/generate", json={"project_description": "short"})

    assert response.status_code == 422


def test_generate_returns_plan_and_job_scoped_files(monkeypatch, tmp_path):
    class FakePlan(BaseModel):
        name: str = "Example"
        description: str = "Example project"
        techstack: str = "HTML, CSS, JavaScript"
        features: list[str] = ["A sample feature"]

    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")

    def fake_invoke(state, config):
        assert state["user_prompt"] == "Build a small example application"
        assert config == {"recursion_limit": 100}
        write_file.invoke({"path": "index.html", "content": "<main>Example</main>"})
        return {"plan": FakePlan()}

    monkeypatch.setattr(api_main.agent, "invoke", fake_invoke)

    response = client.post(
        "/generate",
        json={"project_description": "Build a small example application"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["project"]["name"] == "Example"
    assert body["generated_files"] == ["index.html"]
    assert (tmp_path / "jobs" / body["job_id"] / "index.html").read_text() == "<main>Example</main>"
    assert (tmp_path / "jobs" / ".completed" / f"{body['job_id']}.done").exists()


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


def test_groq_rate_limit_returns_safe_429_without_provider_details(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(api_main, "PROJECTS_ROOT", tmp_path / "jobs")
    provider_details = "org=org_private key=gsk_private_key raw provider traceback"
    request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    provider_response = Response(status_code=429, request=request)

    def rate_limited_invoke(state, config):
        raise RateLimitError(
            provider_details,
            response=provider_response,
            body={"error": {"message": provider_details}},
        )

    monkeypatch.setattr(api_main.agent, "invoke", rate_limited_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post(
        "/generate",
        json={"project_description": "Build a personal to-do application"},
    )

    assert response.status_code == 429
    assert response.json() == {"detail": api_main.RATE_LIMIT_DETAIL}
    assert provider_details not in response.text
    assert "org_private" not in response.text
    assert "gsk_private_key" not in response.text
    assert "traceback" not in response.text.lower()
    assert provider_details not in caplog.text


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

    prompt = "A local mocked generation failure"
    def failing_invoke(state, config):
        raise RuntimeError("mocked failure")

    monkeypatch.setattr(api_main.agent, "invoke", failing_invoke)
    caplog.set_level(logging.ERROR, logger="api.main")

    response = client.post("/generate", json={"project_description": prompt})

    assert response.status_code == 500
    assert "Local generation diagnostic" not in caplog.text


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
    assert "exception_type=RuntimeError" in caplog.text
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

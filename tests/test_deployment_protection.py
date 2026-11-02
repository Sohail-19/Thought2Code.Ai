from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.deployment_protection import (
    GenerationProtection,
    ProtectionStoreUnavailable,
    configured_origins,
)


class FakeRedis:
    """Small deterministic Redis-script fake shared by protection instances."""

    def __init__(self):
        self.counts = {}
        self.slots = set()
        self.keys = []

    def eval(self, script, _key_count, key, *args):
        self.keys.append(key)
        if "INCR" in script:
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
            return [count, 3600]
        if "ZCARD" in script:
            if len(self.slots) >= int(args[0]):
                return 0
            self.slots.add(args[2])
            return 1
        if "ZREM" in script:
            token = args[0]
            existed = token in self.slots
            self.slots.discard(token)
            return int(existed)
        raise AssertionError("unexpected Redis script")


def test_local_limiter_is_per_client_and_bounded(monkeypatch):
    monkeypatch.setenv("THOUGHT2CODE_CLIENT_RATE_LIMIT", "2")
    limiter = GenerationProtection(redis_url=None)

    assert limiter.request_allowed("client-a")
    assert limiter.request_allowed("client-a")
    assert not limiter.request_allowed("client-a")
    assert limiter.request_allowed("client-b")


def test_shared_redis_limits_across_instances_without_storing_ip(monkeypatch):
    monkeypatch.setenv("THOUGHT2CODE_CLIENT_RATE_LIMIT", "2")
    store = FakeRedis()
    first = GenerationProtection(redis_url="redis://test")
    second = GenerationProtection(redis_url="redis://test")
    first.redis = second.redis = store

    assert first.request_allowed("203.0.113.10")
    assert second.request_allowed("203.0.113.10")
    assert not first.request_allowed("203.0.113.10")
    assert store.keys == [
        f"thought2code:rate:{hashlib.sha256(b'203.0.113.10').hexdigest()}"
    ] * 3
    assert all("203.0.113.10" not in key for key in store.keys)


def test_shared_generation_slots_are_global_and_released():
    store = FakeRedis()
    first = GenerationProtection(redis_url="redis://test")
    second = GenerationProtection(redis_url="redis://test")
    first.max_concurrent = second.max_concurrent = 1
    first.redis = second.redis = store

    lease = first.acquire_distributed_slot()
    assert lease
    assert second.acquire_distributed_slot() is None
    first.release_distributed_slot(lease)
    assert second.acquire_distributed_slot()


def test_public_demo_fails_closed_without_shared_store(monkeypatch):
    monkeypatch.setenv("THOUGHT2CODE_PUBLIC_DEMO", "true")
    monkeypatch.delenv("THOUGHT2CODE_REDIS_URL", raising=False)
    limiter = GenerationProtection()

    with pytest.raises(ProtectionStoreUnavailable):
        limiter.request_allowed("client-a")
    with pytest.raises(ProtectionStoreUnavailable):
        limiter.acquire_distributed_slot()


def test_public_demo_fails_closed_when_shared_store_is_unreachable(monkeypatch):
    monkeypatch.setenv("THOUGHT2CODE_PUBLIC_DEMO", "true")
    limiter = GenerationProtection(redis_url="redis://unreachable")

    class UnavailableRedis:
        def eval(self, *_args, **_kwargs):
            raise ConnectionError("private connection details must not escape")

    limiter.redis = UnavailableRedis()
    with pytest.raises(ProtectionStoreUnavailable):
        limiter.request_allowed("client-a")
    with pytest.raises(ProtectionStoreUnavailable):
        limiter.acquire_distributed_slot()


def test_cors_origins_are_explicit_and_reject_wildcards(monkeypatch):
    monkeypatch.setenv(
        "THOUGHT2CODE_ALLOWED_ORIGINS",
        "https://thought2code.example, https://preview.example",
    )
    assert configured_origins() == [
        "https://thought2code.example",
        "https://preview.example",
    ]
    monkeypatch.setenv("THOUGHT2CODE_ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError):
        configured_origins()
    monkeypatch.setenv("THOUGHT2CODE_ALLOWED_ORIGINS", "https://example.test/path")
    with pytest.raises(RuntimeError):
        configured_origins()


def test_render_runtime_defaults_to_protected_public_mode(monkeypatch):
    monkeypatch.delenv("THOUGHT2CODE_PUBLIC_DEMO", raising=False)
    monkeypatch.setenv("RENDER", "true")
    limiter = GenerationProtection(redis_url=None)
    assert limiter.public_demo is True


def test_generation_kill_switch_returns_safe_503(monkeypatch):
    monkeypatch.setattr(api_main.protection, "generation_enabled", False)
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("disabled generation must not invoke the graph"),
    )

    response = TestClient(api_main.app).post(
        "/generate", json={"project_description": "Create a small demo app"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Generation is temporarily disabled. Please try again later."
    }


def test_generation_rate_limit_returns_429_without_invoking_graph(monkeypatch):
    monkeypatch.setattr(api_main.protection, "generation_enabled", True)
    monkeypatch.setattr(api_main.protection, "public_demo", False)
    monkeypatch.setattr(api_main.protection, "request_allowed", lambda _identity: False)
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("rate-limited generation must not invoke the graph"),
    )

    response = TestClient(api_main.app).post(
        "/generate", json={"project_description": "Create a small demo app"}
    )

    assert response.status_code == 429
    assert response.json()["detail"].startswith("Generation request limit reached.")


def test_public_generation_requires_shared_store(monkeypatch):
    monkeypatch.setattr(api_main.protection, "generation_enabled", True)
    monkeypatch.setattr(api_main.protection, "public_demo", True)
    monkeypatch.setattr(api_main.protection, "redis", None)
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("unprotected generation must not invoke the graph"),
    )

    response = TestClient(api_main.app).post(
        "/generate", json={"project_description": "Create a small demo app"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Generation protection is temporarily unavailable. Please try again later."
    }


def test_public_generation_returns_safe_503_when_shared_store_fails(monkeypatch):
    monkeypatch.setattr(api_main.protection, "generation_enabled", True)
    monkeypatch.setattr(api_main.protection, "public_demo", True)
    monkeypatch.setattr(api_main.protection, "redis", object())
    monkeypatch.setattr(
        api_main.protection,
        "request_allowed",
        lambda _identity: (_ for _ in ()).throw(ProtectionStoreUnavailable("private error")),
    )
    monkeypatch.setattr(
        api_main.agent,
        "invoke",
        lambda *_args, **_kwargs: pytest.fail("unprotected generation must not invoke the graph"),
    )

    response = TestClient(api_main.app).post(
        "/generate", json={"project_description": "Create a small demo app"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Generation protection is temporarily unavailable. Please try again later."
    }
    assert "private error" not in response.text

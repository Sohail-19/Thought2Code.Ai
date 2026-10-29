import pytest
from groq import AuthenticationError, BadRequestError, RateLimitError

import agent.runtime as runtime


class FakeModel:
    def __init__(self):
        self.updates = []

    def model_copy(self, *, update):
        self.updates.append(update)
        return self


def test_transient_failure_is_retried_once(monkeypatch):
    class TemporaryFailure(Exception):
        pass

    monkeypatch.setattr(runtime, "TRANSIENT_PROVIDER_ERRORS", (TemporaryFailure,))
    monkeypatch.setattr(runtime, "MAX_TRANSIENT_RETRIES", 1)
    monkeypatch.setattr(runtime, "TRANSIENT_RETRY_BACKOFF_SECONDS", 0)
    calls = []

    def operation(_model):
        calls.append(1)
        if len(calls) == 1:
            raise TemporaryFailure("private detail")
        return "ok"

    assert runtime.invoke_with_transient_retries(FakeModel(), operation) == "ok"
    assert len(calls) == 2


def test_generation_retry_budget_is_shared_across_model_calls(monkeypatch):
    class TemporaryFailure(Exception):
        pass

    monkeypatch.setattr(runtime, "TRANSIENT_PROVIDER_ERRORS", (TemporaryFailure,))
    monkeypatch.setattr(runtime, "MAX_TRANSIENT_RETRIES", 1)
    monkeypatch.setattr(runtime, "TRANSIENT_RETRY_BACKOFF_SECONDS", 0)
    first_calls = []
    second_calls = []

    def first_operation(_model):
        first_calls.append(1)
        if len(first_calls) == 1:
            raise TemporaryFailure()
        return "ok"

    def second_operation(_model):
        second_calls.append(1)
        raise TemporaryFailure()

    with runtime.generation_deadline(30):
        assert runtime.invoke_with_transient_retries(
            FakeModel(), first_operation
        ) == "ok"
        with pytest.raises(TemporaryFailure):
            runtime.invoke_with_transient_retries(FakeModel(), second_operation)

    assert len(first_calls) == 2
    assert len(second_calls) == 1


def test_non_transient_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(runtime, "MAX_TRANSIENT_RETRIES", 2)
    calls = []

    def operation(_model):
        calls.append(1)
        raise ValueError("invalid request")

    with pytest.raises(ValueError):
        runtime.invoke_with_transient_retries(FakeModel(), operation)
    assert len(calls) == 1


def test_provider_quota_authentication_and_invalid_request_are_not_retry_classes():
    assert RateLimitError not in runtime.TRANSIENT_PROVIDER_ERRORS
    assert AuthenticationError not in runtime.TRANSIENT_PROVIDER_ERRORS
    assert BadRequestError not in runtime.TRANSIENT_PROVIDER_ERRORS


def test_expired_deadline_prevents_model_call():
    model = FakeModel()
    calls = []
    with runtime.generation_deadline(0):
        with pytest.raises(runtime.GenerationDeadlineExceeded):
            runtime.invoke_with_transient_retries(model, lambda _model: calls.append(1))
    assert calls == []
    assert model.updates == []


def test_model_call_disables_sdk_retries_and_applies_request_timeout(monkeypatch):
    monkeypatch.setattr(runtime, "MODEL_REQUEST_TIMEOUT_SECONDS", 7.0)
    model = FakeModel()

    runtime.bounded_model(model)

    assert model.updates == [{"request_timeout": 7.0, "max_retries": 0}]

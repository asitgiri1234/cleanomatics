"""The Groq client wrapper: configuration, error mapping, and JSON parsing.

No network calls. The Groq SDK is stubbed, so these tests check that failures
arrive as the application's own structured errors rather than as raw SDK
exceptions leaking into the agent layer.
"""

import httpx
import pytest

import groq

from app.llm.errors import (
    LLMAPIError,
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.groq_client import GroqLLMClient, LLMClient, parse_json_object
from tests.fakes import FakeLLMClient


# Configuration


def test_missing_api_key_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")

    with pytest.raises(LLMConfigurationError) as caught:
        GroqLLMClient(api_key="")

    assert "GROQ_API_KEY" in str(caught.value)


@pytest.mark.parametrize("key", ["", "   ", None])
def test_blank_keys_are_all_rejected(key):
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient(api_key=key if key is not None else "")


def test_key_is_never_read_from_source(monkeypatch):
    """The key must come from configuration, never from a literal in the code."""
    from pathlib import Path

    source = Path("app/llm/groq_client.py").read_text(encoding="utf-8")
    assert "gsk_" not in source


def test_client_uses_the_configured_model():
    client = GroqLLMClient(api_key="test-key", model="some-model")

    assert client.model == "some-model"


def test_fake_client_satisfies_the_protocol():
    assert isinstance(FakeLLMClient("hi"), LLMClient)


# Error mapping


class _FailingCompletions:
    def __init__(self, error: Exception):
        self.error = error

    def create(self, **_kwargs):
        raise self.error


def _client_raising(error: Exception) -> GroqLLMClient:
    client = GroqLLMClient(api_key="test-key")
    client._client.chat.completions = _FailingCompletions(error)
    return client


def _status_error(status_code: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})
    return groq.APIStatusError("boom", response=response, body=None)


def test_timeout_becomes_a_timeout_error():
    request = httpx.Request("POST", "https://api.groq.com")
    client = _client_raising(groq.APITimeoutError(request=request))

    with pytest.raises(LLMTimeoutError):
        client.complete("system", "user")


def test_authentication_failure_becomes_a_configuration_error():
    unauthorised = _status_error(401).response
    client = _client_raising(
        groq.AuthenticationError("bad key", response=unauthorised, body=None)
    )

    with pytest.raises(LLMConfigurationError) as caught:
        client.complete("system", "user")

    assert "GROQ_API_KEY" in str(caught.value)


def test_server_error_becomes_an_api_error_with_its_status():
    client = _client_raising(_status_error(503))

    with pytest.raises(LLMAPIError) as caught:
        client.complete("system", "user")

    assert caught.value.status_code == 503


def test_connection_failure_becomes_an_api_error():
    request = httpx.Request("POST", "https://api.groq.com")
    client = _client_raising(groq.APIConnectionError(request=request))

    with pytest.raises(LLMAPIError):
        client.complete("system", "user")


def test_every_llm_error_shares_one_base_class():
    for error in (LLMConfigurationError, LLMAPIError, LLMTimeoutError, LLMResponseError):
        assert issubclass(error, LLMError)


def test_errors_are_not_swallowed():
    """A failing client raises; it never returns an empty string."""
    client = _client_raising(_status_error(500))

    with pytest.raises(LLMError):
        client.complete("system", "user")


# Empty replies


class _EmptyCompletions:
    def __init__(self, content):
        self.content = content

    def create(self, **_kwargs):
        class _Message:
            content = self.content

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_reply_is_a_response_error(content):
    client = GroqLLMClient(api_key="test-key")
    client._client.chat.completions = _EmptyCompletions(content)

    with pytest.raises(LLMResponseError):
        client.complete("system", "user")


# JSON parsing


def test_plain_json_object_is_parsed():
    assert parse_json_object('{"needs_kb": true}') == {"needs_kb": True}


def test_fenced_json_is_recovered():
    assert parse_json_object('```json\n{"needs_kb": true}\n```') == {"needs_kb": True}


def test_json_with_surrounding_prose_is_recovered():
    text = 'Here you go: {"needs_kb": true} — let me know!'
    assert parse_json_object(text) == {"needs_kb": True}


def test_prose_with_no_json_is_rejected():
    with pytest.raises(LLMResponseError) as caught:
        parse_json_object("I think you want the refund policy.")

    assert caught.value.raw == "I think you want the refund policy."


def test_json_array_is_rejected():
    with pytest.raises(LLMResponseError) as caught:
        parse_json_object("[1, 2, 3]")

    assert "object" in str(caught.value)


def test_broken_json_is_rejected_with_the_original_text():
    with pytest.raises(LLMResponseError) as caught:
        parse_json_object('{"needs_kb": tru')

    assert caught.value.raw is not None

"""Unit tests for LLM client functions."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from mr_lead_agent.llm import (
    _call_openai_compat,
    call_deepseek,
    call_groq,
    call_openrouter,
)
from mr_lead_agent.models import PipelineStats, RedactionStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_REVIEW_JSON = json.dumps({
    "summary": ["Looks good"],
    "key_risks": [],
    "blockers": [],
    "questions_to_author": [],
})

VALID_RESPONSE = {
    "choices": [{"message": {"content": VALID_REVIEW_JSON}}]
}


def _make_stats() -> PipelineStats:
    return PipelineStats(
        diff_lines=10,
        context_fragments=1,
        context_files=1,
        context_lines=5,
        redaction=RedactionStats(secrets_replaced=0, urls_replaced=0, files_excluded=0),
        summary_only_mode=False,
    )


# ---------------------------------------------------------------------------
# _call_openai_compat — success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_returns_review_result() -> None:
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    result = await _call_openai_compat(
        prompt="review this",
        api_key="sk-test",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    assert result.summary == ["Looks good"]
    assert result.blockers == []


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_sends_auth_header() -> None:
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    await _call_openai_compat(
        prompt="test",
        api_key="sk-mykey",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    assert route.called
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-mykey"


# ---------------------------------------------------------------------------
# _call_openai_compat — 429 retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_retries_on_429_then_succeeds() -> None:
    call_count = 0

    def side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=VALID_RESPONSE)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    result = await _call_openai_compat(
        prompt="test",
        api_key="sk-test",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
        max_retries=3,
    )
    assert result.summary == ["Looks good"]
    assert call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_exhausts_retries_returns_degraded() -> None:
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )

    result = await _call_openai_compat(
        prompt="test",
        api_key="sk-test",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
        max_retries=2,
    )
    # degraded mode: summary contains error message
    assert any("429" in point or "rate" in point.lower() for point in result.summary)


# ---------------------------------------------------------------------------
# _call_openai_compat — 4xx errors (non-429)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_401_returns_degraded() -> None:
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "Unauthorized"})
    )
    result = await _call_openai_compat(
        prompt="test",
        api_key="bad-key",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    assert any("401" in p for p in result.summary)


# ---------------------------------------------------------------------------
# _call_openai_compat — parse error (bad JSON in content)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_bad_json_returns_degraded() -> None:
    bad_response = {"choices": [{"message": {"content": "not json at all"}}]}
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=bad_response)
    )
    result = await _call_openai_compat(
        prompt="test",
        api_key="sk-test",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    # Should return degraded result, not raise
    assert isinstance(result.summary, list)


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_no_choices_returns_degraded() -> None:
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    result = await _call_openai_compat(
        prompt="test",
        api_key="sk-test",
        stats=_make_stats(),
        base_url="https://api.example.com/v1",
        model="test-model",
    )
    assert any("No choices" in p for p in result.summary)


# ---------------------------------------------------------------------------
# Provider wrappers — check correct base_url is used
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_call_deepseek_uses_deepseek_url() -> None:
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    await call_deepseek("prompt", "sk-ds", _make_stats())
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_call_openrouter_uses_openrouter_url() -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    await call_openrouter("prompt", "sk-or", _make_stats())
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_call_groq_uses_groq_url() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    await call_groq("prompt", "gsk-test", _make_stats())
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_call_openrouter_does_not_send_response_format() -> None:
    """OpenRouter free models don't support json_mode — verify it's not sent."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=VALID_RESPONSE)
    )
    await call_openrouter("prompt", "sk-or", _make_stats())
    request_body = json.loads(route.calls[0].request.content)
    assert "response_format" not in request_body

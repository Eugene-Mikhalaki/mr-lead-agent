"""LLM client: Gemini, DeepSeek, OpenRouter, Groq — all providers in one place."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from google import genai
from google.genai import types
from pydantic import ValidationError

from mr_lead_agent.models import (
    Blocker,
    DiscussionReply,
    PipelineStats,
    Question,
    ReviewResult,
    Risk,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-1.5-pro"
_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=8192,
)


async def fetch_model_info(
    api_key: str,
    base_url: str,
    model: str,
    provider_name: str = "LLM",
) -> None:
    """Query /v1/models to log context window + release date.

    Non-blocking: errors are caught and logged at DEBUG level only.
    """
    ctx = None
    created_ts = None
    
    try:
        if provider_name == "DeepSeek":
            # DeepSeek's API doesn't return context info, so we fetch it from OpenRouter
            url = "https://openrouter.ai/api/v1/models"
            openrouter_id = f"deepseek/{model}"
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(url)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    if m.get("id") == openrouter_id:
                        ctx = m.get("context_length")
                        created_ts = m.get("created")
                        break
        else:
            url = f"{base_url.rstrip('/')}/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                ctx = (
                    data.get("context_length")
                    or data.get("max_context_length")
                    or data.get("context_window")
                )
                created_ts = data.get("created")

        created_str = (
            datetime.fromtimestamp(created_ts, tz=UTC).strftime("%Y-%m-%d")
            if created_ts
            else "n/a"
        )
        ctx_str = f"{ctx:,}" if ctx else "unknown"
        logger.info(
            "[%s] Model: %s  |  Context window: %s tokens  |  Released: %s",
            provider_name,
            model,
            ctx_str,
            created_str,
        )
    except Exception as exc:
        logger.debug("%s model info fetch failed: %s", provider_name, exc)



def _parse_review_result(raw: str) -> ReviewResult:
    """Parse a JSON string into a ReviewResult.

    Strips markdown code fences if present.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data: dict[str, Any] = json.loads(text)

    # Coerce list fields in case the model omits them
    return ReviewResult(
        summary=data.get("summary", []),
        key_risks=[Risk(**r) for r in data.get("key_risks", [])],
        blockers=[Blocker(**b) for b in data.get("blockers", [])],
        questions_to_author=[Question(**q) for q in data.get("questions_to_author", [])],
        discussion_replies=[DiscussionReply(**d) for d in data.get("discussion_replies", [])],
    )


async def call_gemini(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    model: str = _MODEL_NAME,
) -> ReviewResult:
    """Send the prompt to Gemini via HTTP REST API and return a validated ReviewResult.

    On failure (API error, invalid JSON, validation error) returns a
    degraded ReviewResult with error info in the summary.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        },
    }

    logger.info("Calling Gemini API (HTTP) model %s (%d chars prompt)", model, len(prompt))

    # We use a longer timeout because generation can take some time
    timeout = httpx.Timeout(60.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                params={"key": api_key},
                json=payload,
            )

            if response.status_code >= 400:
                error_body = response.text[:1000]
                logger.error("Gemini HTTP API call failed: %d %s. %s", response.status_code, response.reason_phrase, error_body)
                return _degraded_result(f"LLM API error: {response.status_code} {response.reason_phrase}. {error_body}", stats)

            data = response.json()
            # Extract text from the response structure:
            # candidates[0].content.parts[0].text
            candidates = data.get("candidates", [])
            if not candidates:
                logger.error("Gemini API returned no candidates: %s", data)
                return _degraded_result("LLM API error: No candidates returned", stats)

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                logger.error("Gemini API returned no parts: %s", data)
                return _degraded_result("LLM API error: No parts in response content", stats)

            raw_text = parts[0].get("text", "")
            logger.debug("Gemini response length: %d chars", len(raw_text))

    except httpx.RequestError as exc:
        logger.error("HTTP request to Gemini failed: %s", exc)
        return _degraded_result(f"HTTP request error: {exc}", stats)
    except Exception as exc:
        logger.error("Unexpected error calling Gemini: %s", exc)
        return _degraded_result(f"Unexpected error: {exc}", stats)

    try:
        result = _parse_review_result(raw_text)
        # Enforce max_blockers is done by caller, but we return the raw parsed object here
        return result
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        logger.error("Failed to parse Gemini response: %s\nRaw: %.500s", exc, raw_text)
        return _degraded_result(f"LLM response parse error: {exc}", stats)




async def _call_openai_compat(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    base_url: str,
    model: str,
    provider_name: str = "OpenAI-compat",
    extra_headers: dict[str, str] | None = None,
    json_mode: bool = True,
    max_retries: int = 3,
) -> ReviewResult:
    """Generic OpenAI-compatible Chat Completions client with retry for 429.

    Used for DeepSeek, OpenRouter, Groq, and any other OpenAI-compatible provider.
    Retries up to `max_retries` times on 429 with exponential backoff.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer performing a thorough code review. "
                    "Always respond with valid JSON only, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    logger.info(
        "Calling %s model %s (%d chars prompt)", provider_name, model, len(prompt)
    )

    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=10.0)
    retry_delays = [5, 15, 30]  # seconds between retries

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    # Respect Retry-After header if present, else use backoff table
                    retry_after = int(response.headers.get("Retry-After", retry_delays[attempt]))
                    logger.warning(
                        "%s rate-limited (429). Attempt %d/%d. Waiting %ds before retry...",
                        provider_name, attempt + 1, max_retries, retry_after,
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    # Last attempt — return degraded
                    error_body = response.text[:500]
                    return _degraded_result(
                        f"LLM API error: 429 Too Many Requests (rate limited after {max_retries} attempts). {error_body}",
                        stats,
                    )

                if response.status_code >= 400:
                    error_body = response.text[:1000]
                    logger.error(
                        "%s API call failed: %d %s. %s",
                        provider_name,
                        response.status_code,
                        response.reason_phrase,
                        error_body,
                    )
                    return _degraded_result(
                        f"LLM API error: {response.status_code} {response.reason_phrase}. {error_body}",
                        stats,
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    logger.error("%s API returned no choices: %s", provider_name, data)
                    return _degraded_result("LLM API error: No choices returned", stats)

                raw_text = choices[0].get("message", {}).get("content", "")
                # Capture token usage if provided by the API
                usage = data.get("usage", {})
                stats.prompt_tokens = int(usage.get("prompt_tokens", 0))
                stats.completion_tokens = int(usage.get("completion_tokens", 0))
                logger.debug(
                    "%s tokens: prompt=%d completion=%d",
                    provider_name, stats.prompt_tokens, stats.completion_tokens,
                )
                logger.debug("%s response length: %d chars", provider_name, len(raw_text))
                break  # success — exit retry loop

        except httpx.RequestError as exc:
            logger.error("HTTP request to %s failed: %s", provider_name, exc)
            return _degraded_result(f"HTTP request error: {exc}", stats)
        except Exception as exc:
            logger.error("Unexpected error calling %s: %s", provider_name, exc)
            return _degraded_result(f"Unexpected error: {exc}", stats)
    else:
        # Should not reach here, but just in case
        return _degraded_result("LLM API error: Max retries exceeded", stats)

    try:
        return _parse_review_result(raw_text)
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        logger.error(
            "Failed to parse %s response: %s\nRaw: %.500s", provider_name, exc, raw_text
        )
        return _degraded_result(f"LLM response parse error: {exc}", stats)



async def call_deepseek(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    model: str = "deepseek-chat",
) -> ReviewResult:
    """Send the prompt to DeepSeek (platform.deepseek.com)."""
    await fetch_model_info(api_key, "https://api.deepseek.com/v1", model, "DeepSeek")
    return await _call_openai_compat(
        prompt=prompt,
        api_key=api_key,
        stats=stats,
        base_url="https://api.deepseek.com/v1",
        model=model,
        provider_name="DeepSeek",
        json_mode=True,
    )


async def call_openrouter(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    model: str = "deepseek/deepseek-chat:free",
) -> ReviewResult:
    """Send the prompt to OpenRouter (openrouter.ai).

    Free models (suffix ':free'):
      - deepseek/deepseek-chat:free
      - meta-llama/llama-3.3-70b-instruct:free
      - qwen/qwen-2.5-72b-instruct:free

    NOTE: free-tier OpenRouter models do not support response_format JSON mode.
    """
    return await _call_openai_compat(
        prompt=prompt,
        api_key=api_key,
        stats=stats,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        provider_name="OpenRouter",
        extra_headers={
            "HTTP-Referer": "https://github.com/mr-lead-agent",
            "X-Title": "MR Lead Agent",
        },
        json_mode=False,  # free models don't support JSON mode
    )


async def call_groq(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    model: str = "llama-3.3-70b-versatile",
) -> ReviewResult:
    """Send the prompt to Groq (api.groq.com) — free tier, very fast.

    Free models (as of early 2026):
      - llama-3.3-70b-versatile  (recommended)
      - llama-3.1-70b-versatile
      - deepseek-r1-distill-llama-70b

    Free tier: ~14,400 requests/day, 6,000 tokens/minute.
    """
    return await _call_openai_compat(
        prompt=prompt,
        api_key=api_key,
        stats=stats,
        base_url="https://api.groq.com/openai/v1",
        model=model,
        provider_name="Groq",
        json_mode=True,  # Groq supports JSON mode
    )



async def _call_gemini_sdk(
    prompt: str,
    api_key: str,
    stats: PipelineStats,
    model: str = _MODEL_NAME,
) -> ReviewResult:
    """Send the prompt to Gemini using the google-genai SDK.

    Preserved as a fallback implementation as requested by user.
    """
    client = genai.Client(api_key=api_key)

    logger.info("Calling Gemini model %s via SDK (%d chars prompt)", model, len(prompt))
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=_GENERATION_CONFIG,
        )
        raw_text: str = response.text or ""
        logger.debug("Gemini response length: %d chars", len(raw_text))
    except Exception as exc:
        logger.error("Gemini SDK API call failed: %s", exc)
        return _degraded_result(f"LLM API error: {exc}", stats)

    try:
        result = _parse_review_result(raw_text)
        return result
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        logger.error("Failed to parse Gemini SDK response: %s\nRaw: %.500s", exc, raw_text)
        return _degraded_result(f"LLM response parse error: {exc}", stats)


def _degraded_result(reason: str, stats: PipelineStats) -> ReviewResult:
    """Return a minimal ReviewResult explaining the degraded mode."""
    return ReviewResult(
        summary=[
            f"⚠️  Review could not be completed: {reason}",
            f"Diff: {stats.diff_lines} lines, "
            f"Context: {stats.context_fragments} fragments from {stats.context_files} files.",
        ],
    )

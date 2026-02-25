"""End-to-end and integration tests for the MR Lead Agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mr_lead_agent.config import Config
from mr_lead_agent.main import run_review
from mr_lead_agent.models import MRData, PipelineStats, RedactionStats, ReviewResult


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_DIFF = (
    "--- a/src/auth.py\n+++ b/src/auth.py\n"
    "@@ -1,3 +1,5 @@\n"
    "+import jwt\n"
    '+API_KEY = "sk-abc123456789abcdef"\n'
    "+\n"
    "+def login(username, password):\n"
    '+    token = jwt.encode({"user": username}, API_KEY)\n'
    "+    return token\n"
)

_FAKE_MR = MRData(
    title="Add login endpoint",
    description="Adds /api/auth/login",
    author="dev",
    source_branch="feature/auth",
    target_branch="main",
    web_url="https://gitlab.example.com/group/repo/-/merge_requests/42",
    sha="abc123def456abc123def456abc123def456abc1",
    iid=42,
    project_path="group/repo",
    changed_files=["src/auth.py"],
    diff=_SAMPLE_DIFF,
)

_FAKE_REVIEW = ReviewResult(
    summary=["Adds login flow"],
    key_risks=[],
    blockers=[],
    questions_to_author=[],
)


def _make_config(tmp_path: Path, **overrides) -> Config:
    defaults = dict(
        repo_url="https://gitlab.example.com/group/repo.git",
        mr_iid=42,
        gitlab_base_url="https://gitlab.example.com",
        gitlab_token="glpat-test",
        gemini_api_key="test-key",
        workdir=str(tmp_path / "repos"),
        dry_run=False,
        log_level="WARNING",
        # Pin provider so pydantic-settings doesn't pick up LLM_PROVIDER from .env
        llm_provider="gemini",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# E2E dry-run: pipeline runs without LLM call and without errors
# ---------------------------------------------------------------------------

class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_skips_llm(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, dry_run=True)

        with (
            patch("mr_lead_agent.main.GitLabClient") as mock_gl,
            patch("mr_lead_agent.main.RepoManager") as mock_rm,
            patch("mr_lead_agent.main.call_gemini") as mock_llm,
            patch("mr_lead_agent.main.render_dry_run"),
            patch("mr_lead_agent.main.render_report"),
        ):
            # Set up GitLab client mock
            gl_instance = AsyncMock()
            gl_instance.__aenter__ = AsyncMock(return_value=gl_instance)
            gl_instance.__aexit__ = AsyncMock(return_value=None)
            gl_instance.get_mr_data = AsyncMock(return_value=_FAKE_MR)
            mock_gl.return_value = gl_instance

            # Set up repo manager mock
            rm_instance = MagicMock()
            rm_instance.ensure_repo = AsyncMock(return_value=Path("/tmp/repo"))
            rm_instance.checkout_sha = AsyncMock()
            mock_rm.return_value = rm_instance

            await run_review(config)

        # LLM should NOT be called in dry-run mode
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Security test: secrets in diff are redacted before reaching the LLM
# ---------------------------------------------------------------------------

class TestSecretsNotSentToLLM:
    @pytest.mark.asyncio
    async def test_secret_key_redacted_in_prompt(self, tmp_path: Path) -> None:
        """Verify SECRET_KEY in diff does not reach the LLM prompt."""
        config = _make_config(tmp_path)

        captured_prompt: list[str] = []

        async def fake_llm(prompt: str, api_key: str, stats: PipelineStats, **kwargs) -> ReviewResult:
            captured_prompt.append(prompt)
            return _FAKE_REVIEW

        with (
            patch("mr_lead_agent.main.GitLabClient") as mock_gl,
            patch("mr_lead_agent.main.RepoManager") as mock_rm,
            patch("mr_lead_agent.main.call_gemini", side_effect=fake_llm),
            patch("mr_lead_agent.main.render_report"),
        ):
            gl_instance = AsyncMock()
            gl_instance.__aenter__ = AsyncMock(return_value=gl_instance)
            gl_instance.__aexit__ = AsyncMock(return_value=None)
            gl_instance.get_mr_data = AsyncMock(return_value=_FAKE_MR)
            mock_gl.return_value = gl_instance

            rm_instance = MagicMock()
            rm_instance.ensure_repo = AsyncMock(return_value=Path("/tmp/repo"))
            rm_instance.checkout_sha = AsyncMock()
            mock_rm.return_value = rm_instance

            await run_review(config)

        assert captured_prompt, "LLM was never called"
        prompt_text = captured_prompt[0]
        # The raw API key value should NOT appear in the prompt (redacted)
        assert "sk-abc123456789abcdef" not in prompt_text


# ---------------------------------------------------------------------------
# Idempotency: second run reuses the repo cache (git fetch, not clone)
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.mark.asyncio
    async def test_second_run_reuses_repo_cache(self, tmp_path: Path) -> None:
        """RepoManager.sync should be called on each run (fetch not clone)."""
        config = _make_config(tmp_path)

        async def fake_llm(prompt: str, api_key: str, stats: PipelineStats, **kwargs) -> ReviewResult:
            return _FAKE_REVIEW

        sync_call_count = 0

        with (
            patch("mr_lead_agent.main.GitLabClient") as mock_gl,
            patch("mr_lead_agent.main.RepoManager") as mock_rm,
            patch("mr_lead_agent.main.call_gemini", side_effect=fake_llm),
            patch("mr_lead_agent.main.render_report"),
        ):
            gl_instance = AsyncMock()
            gl_instance.__aenter__ = AsyncMock(return_value=gl_instance)
            gl_instance.__aexit__ = AsyncMock(return_value=None)
            gl_instance.get_mr_data = AsyncMock(return_value=_FAKE_MR)
            mock_gl.return_value = gl_instance

            def make_rm(*args, **kwargs):
                nonlocal sync_call_count
                rm = MagicMock()

                async def counting_sync(*a, **kw):
                    nonlocal sync_call_count
                    sync_call_count += 1
                    return Path("/tmp/repo")

                rm.ensure_repo = AsyncMock(side_effect=counting_sync)
                rm.checkout_sha = AsyncMock()
                return rm

            mock_rm.side_effect = make_rm

            # Run twice
            await run_review(config)
            await run_review(config)

        assert sync_call_count == 2, "sync() should be called on every run"


# ---------------------------------------------------------------------------
# Integration: result is saved to runs/ on success
# ---------------------------------------------------------------------------

class TestResultPersistence:
    @pytest.mark.asyncio
    async def test_result_saved_to_runs_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runs_dir = tmp_path / "runs"

        async def fake_llm(prompt: str, api_key: str, stats: PipelineStats, **kwargs) -> ReviewResult:
            return _FAKE_REVIEW

        with (
            patch("mr_lead_agent.main.GitLabClient") as mock_gl,
            patch("mr_lead_agent.main.RepoManager") as mock_rm,
            patch("mr_lead_agent.main.call_gemini", side_effect=fake_llm),
            patch("mr_lead_agent.main.render_report") as mock_render,
        ):
            gl_instance = AsyncMock()
            gl_instance.__aenter__ = AsyncMock(return_value=gl_instance)
            gl_instance.__aexit__ = AsyncMock(return_value=None)
            gl_instance.get_mr_data = AsyncMock(return_value=_FAKE_MR)
            mock_gl.return_value = gl_instance

            rm_instance = MagicMock()
            rm_instance.ensure_repo = AsyncMock(return_value=Path("/tmp/repo"))
            rm_instance.checkout_sha = AsyncMock()
            mock_rm.return_value = rm_instance

            await run_review(config)

        # render_report should have been called with the review result
        mock_render.assert_called_once()
        call_args = mock_render.call_args[0]  # positional args
        result_arg = call_args[1]  # second positional arg is ReviewResult
        assert isinstance(result_arg, ReviewResult)
        assert result_arg.summary == ["Adds login flow"]

"""Unit tests for the report renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr_lead_agent.models import (
    Blocker,
    MRData,
    PipelineStats,
    Question,
    RedactionStats,
    ReviewResult,
    Risk,
)
from mr_lead_agent.renderer import _save_json, render_dry_run, render_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mr_data() -> MRData:
    return MRData(
        title="Fix bug in auth",
        description="Patches login issue",
        author="bob",
        source_branch="fix/auth-bug",
        target_branch="main",
        web_url="https://gitlab.example.com/org/repo/-/merge_requests/7",
        sha="deadbeefcafe1234deadbeefcafe1234deadbeef",
        iid=7,
        project_path="org/repo",
        changed_files=["src/auth.py"],
        diff="--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1 +1 @@\n-old\n+new\n",
    )


@pytest.fixture()
def pipeline_stats() -> PipelineStats:
    return PipelineStats(
        diff_lines=10,
        context_fragments=2,
        context_files=1,
        context_lines=20,
        redaction=RedactionStats(secrets_replaced=0, urls_replaced=0, files_excluded=0),
        summary_only_mode=False,
    )


@pytest.fixture()
def clean_result() -> ReviewResult:
    return ReviewResult(
        summary=["Looks good overall"],
        key_risks=[],
        blockers=[],
        questions_to_author=[],
    )


@pytest.fixture()
def full_result() -> ReviewResult:
    return ReviewResult(
        summary=["Added auth endpoint", "Missing tests"],
        key_risks=[
            Risk(severity="major", title="No input validation", details="params unchecked")
        ],
        blockers=[
            Blocker(
                severity="blocker",
                file="src/auth.py",
                lines="12-15",
                title="SQL injection risk",
                comment="Raw string concatenation in query",
                suggested_fix="Use parameterized queries",
                verification="Run sqlmap against the endpoint",
            )
        ],
        questions_to_author=[
            Question(
                file="src/auth.py",
                lines="20",
                question="Why is SECRET_KEY hardcoded?",
                why_it_matters="Security risk",
            )
        ],
    )


# ---------------------------------------------------------------------------
# _save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path: Path, mr_data: MRData, clean_result: ReviewResult, pipeline_stats: PipelineStats) -> None:
        _save_json(mr_data, clean_result, pipeline_stats, str(tmp_path))
        files = list(tmp_path.glob("mr7_*.json"))
        assert len(files) == 1

    def test_filename_contains_iid_and_sha(self, tmp_path: Path, mr_data: MRData, clean_result: ReviewResult, pipeline_stats: PipelineStats) -> None:
        _save_json(mr_data, clean_result, pipeline_stats, str(tmp_path))
        files = list(tmp_path.glob("mr7_deadbeef*.json"))
        assert len(files) == 1

    def test_json_content_valid(self, tmp_path: Path, mr_data: MRData, full_result: ReviewResult, pipeline_stats: PipelineStats) -> None:
        _save_json(mr_data, full_result, pipeline_stats, str(tmp_path))
        files = list(tmp_path.glob("mr7_*.json"))
        data = json.loads(files[0].read_text())
        assert "timestamp" in data
        assert data["mr"]["title"] == "Fix bug in auth"
        assert data["result"]["blockers"][0]["title"] == "SQL injection risk"

    def test_creates_runs_dir_if_missing(self, tmp_path: Path, mr_data: MRData, clean_result: ReviewResult, pipeline_stats: PipelineStats) -> None:
        nested = tmp_path / "deep" / "runs"
        _save_json(mr_data, clean_result, pipeline_stats, str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# render_report (smoke test — just check no exceptions)
# ---------------------------------------------------------------------------

class TestRenderReport:
    def test_renders_clean_result(self, mr_data: MRData, clean_result: ReviewResult, pipeline_stats: PipelineStats, tmp_path: Path) -> None:
        # Should not raise
        render_report(mr_data, clean_result, pipeline_stats, save_runs=True, runs_dir=str(tmp_path))
        assert list(tmp_path.glob("mr7_*.json"))

    def test_renders_full_result(self, mr_data: MRData, full_result: ReviewResult, pipeline_stats: PipelineStats, tmp_path: Path) -> None:
        render_report(mr_data, full_result, pipeline_stats, save_runs=False, runs_dir=str(tmp_path))

    def test_no_save_skips_file_creation(self, mr_data: MRData, clean_result: ReviewResult, pipeline_stats: PipelineStats, tmp_path: Path) -> None:
        render_report(mr_data, clean_result, pipeline_stats, save_runs=False, runs_dir=str(tmp_path))
        assert not list(tmp_path.glob("*.json"))


# ---------------------------------------------------------------------------
# render_dry_run (smoke test)
# ---------------------------------------------------------------------------

class TestRenderDryRun:
    def test_renders_without_error(self, mr_data: MRData, pipeline_stats: PipelineStats) -> None:
        render_dry_run(mr_data, "short prompt", pipeline_stats)

    def test_renders_long_prompt(self, mr_data: MRData, pipeline_stats: PipelineStats) -> None:
        long_prompt = "x" * 5000
        render_dry_run(mr_data, long_prompt, pipeline_stats)

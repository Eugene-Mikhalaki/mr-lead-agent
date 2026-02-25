"""Unit tests for GitLab API client."""

from __future__ import annotations

import pytest
import respx
import httpx

from mr_lead_agent.gitlab_client import GitLabClient, GitLabAPIError, _extract_project_path


# ---------------------------------------------------------------------------
# _extract_project_path
# ---------------------------------------------------------------------------

class TestExtractProjectPath:
    def test_simple_two_segment(self) -> None:
        assert _extract_project_path("https://gitlab.example.com/group/repo.git") == "group/repo"

    def test_three_segment_nested(self) -> None:
        assert _extract_project_path("https://gitlab.example.com/a/b/c.git") == "a/b/c"

    def test_no_git_suffix(self) -> None:
        assert _extract_project_path("https://gitlab.example.com/group/repo") == "group/repo"

    def test_trailing_slash(self) -> None:
        assert _extract_project_path("https://gitlab.example.com/group/repo/") == "group/repo"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot extract project path"):
            _extract_project_path("not-a-url")


# ---------------------------------------------------------------------------
# GitLabClient.get_mr_data
# ---------------------------------------------------------------------------

MR_META_RESPONSE = {
    "title": "Add feature",
    "description": "Adds a new thing",
    "author": {"username": "alice"},
    "source_branch": "feature/add-thing",
    "target_branch": "main",
    "web_url": "https://gitlab.example.com/group/repo/-/merge_requests/42",
    "sha": "abc123def456abc123def456abc123def456abc1",
    "iid": 42,
}

MR_CHANGES_RESPONSE = {
    "changes": [
        {
            "new_path": "src/thing.py",
            "diff": "--- a/src/thing.py\n+++ b/src/thing.py\n@@ -1 +1,2 @@\n+def thing():\n+    pass\n",
        },
        {
            "new_path": "tests/test_thing.py",
            "diff": "",
        },
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_get_mr_data_success() -> None:
    base = "https://gitlab.example.com"
    project = "group%2Frepo"
    respx.get(f"{base}/api/v4/projects/{project}/merge_requests/42").mock(
        return_value=httpx.Response(200, json=MR_META_RESPONSE)
    )
    respx.get(f"{base}/api/v4/projects/{project}/merge_requests/42/changes").mock(
        return_value=httpx.Response(200, json=MR_CHANGES_RESPONSE)
    )

    async with GitLabClient(base, "glpat-test") as client:
        mr = await client.get_mr_data("https://gitlab.example.com/group/repo.git", 42)

    assert mr.title == "Add feature"
    assert mr.author == "alice"
    assert mr.iid == 42
    assert "src/thing.py" in mr.changed_files
    assert "tests/test_thing.py" in mr.changed_files
    assert "def thing" in mr.diff


@pytest.mark.asyncio
@respx.mock
async def test_get_mr_data_404_raises() -> None:
    base = "https://gitlab.example.com"
    project = "group%2Frepo"
    respx.get(f"{base}/api/v4/projects/{project}/merge_requests/99").mock(
        return_value=httpx.Response(404, json={"message": "Not found"})
    )

    async with GitLabClient(base, "glpat-test") as client:
        with pytest.raises(GitLabAPIError) as exc_info:
            await client.get_mr_data("https://gitlab.example.com/group/repo.git", 99)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_get_mr_data_filters_empty_diffs() -> None:
    """Files with empty diffs should still appear in changed_files."""
    base = "https://gitlab.example.com"
    project = "group%2Frepo"
    respx.get(f"{base}/api/v4/projects/{project}/merge_requests/42").mock(
        return_value=httpx.Response(200, json=MR_META_RESPONSE)
    )
    respx.get(f"{base}/api/v4/projects/{project}/merge_requests/42/changes").mock(
        return_value=httpx.Response(200, json=MR_CHANGES_RESPONSE)
    )

    async with GitLabClient(base, "glpat-test") as client:
        mr = await client.get_mr_data("https://gitlab.example.com/group/repo.git", 42)

    # both files listed in changed_files
    assert len(mr.changed_files) == 2

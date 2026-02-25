"""Shared test fixtures."""

from __future__ import annotations

import pytest

from mr_lead_agent.config import Config
from mr_lead_agent.models import ContextFragment, MRData


@pytest.fixture()
def sample_mr() -> MRData:
    return MRData(
        title="Add user auth endpoint",
        description="Adds /api/auth/login",
        author="dev",
        source_branch="feature/auth",
        target_branch="main",
        web_url="https://gitlab.example.com/group/repo/-/merge_requests/42",
        sha="abc123def456abc123def456abc123def456abc1",
        iid=42,
        project_path="group/repo",
        changed_files=["src/auth.py", "tests/test_auth.py"],
        diff=(
            "--- a/src/auth.py\n+++ b/src/auth.py\n"
            "@@ -1,3 +1,10 @@\n"
            "+import jwt\n"
            "+SECRET_KEY = 'super_secret_key_12345'\n"
            "+\n"
            "+def login(username, password):\n"
            "+    token = jwt.encode({'user': username}, SECRET_KEY)\n"
            "+    return token\n"
        ),
    )


@pytest.fixture()
def minimal_config() -> Config:
    return Config(
        repo_url="https://gitlab.example.com/group/repo.git",
        mr_iid=42,
        gitlab_base_url="https://gitlab.example.com",
        gitlab_token="glpat-test",
        gemini_api_key="test-key",
    )


@pytest.fixture()
def sample_fragment() -> ContextFragment:
    return ContextFragment(
        file_path="src/auth.py",
        line_start=10,
        line_end=20,
        code_excerpt="def login(username, password):\n    pass\n",
        token_match="login",
    )

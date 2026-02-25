"""Tests for secret redaction and file filtering."""

from __future__ import annotations

import pytest

from mr_lead_agent.redaction import (
    redact_secrets,
    redact_internal_urls,
    should_exclude_file,
)


class TestShouldExcludeFile:
    def test_excludes_env_file(self) -> None:
        assert should_exclude_file(".env", [], []) is True

    def test_excludes_pem_file(self) -> None:
        assert should_exclude_file("certs/server.pem", [], []) is True

    def test_excludes_key_file(self) -> None:
        assert should_exclude_file("keys/id_rsa", [], []) is True

    def test_excludes_user_deny_glob(self) -> None:
        assert should_exclude_file("config/secrets.yaml", ["**/secrets.yaml"], []) is True

    def test_allows_normal_python_file(self) -> None:
        assert should_exclude_file("src/auth.py", [], []) is False

    def test_allow_dirs_filters_outside(self) -> None:
        # Only src/ is allowed — tests/ should be excluded
        assert should_exclude_file("tests/test_auth.py", [], ["src/"]) is True

    def test_allow_dirs_permits_inside(self) -> None:
        assert should_exclude_file("src/auth.py", [], ["src/"]) is False


class TestRedactSecrets:
    def test_masks_api_key_assignment(self) -> None:
        text = 'API_KEY = "sk-abc123456789"'
        result, stats = redact_secrets(text)
        assert "sk-abc123456789" not in result
        assert stats.secrets_replaced >= 1

    def test_masks_token_assignment(self) -> None:
        text = "token: glpat-abcdefghij1234"
        result, stats = redact_secrets(text)
        assert "glpat-abcdefghij1234" not in result

    def test_masks_secret_assignment(self) -> None:
        text = "SECRET=my_super_secret_value"
        result, stats = redact_secrets(text)
        assert "my_super_secret_value" not in result

    def test_preserves_non_secret_text(self) -> None:
        text = "def calculate_total(items):\n    return sum(items)"
        result, stats = redact_secrets(text)
        assert result == text
        assert stats.secrets_replaced == 0

    def test_redacted_marker_present(self) -> None:
        text = "password: hunter2"
        result, stats = redact_secrets(text)
        assert "REDACTED" in result


class TestRedactInternalUrls:
    def test_masks_internal_domain(self) -> None:
        text = "See http://gitlab.internal/group/repo for details"
        result, stats = redact_internal_urls(text, ["gitlab.internal"])
        assert "gitlab.internal" not in result
        assert stats.urls_replaced == 1

    def test_no_domains_no_change(self) -> None:
        text = "https://example.com/path"
        result, stats = redact_internal_urls(text, [])
        assert result == text
        assert stats.urls_replaced == 0

    def test_external_url_not_masked(self) -> None:
        text = "https://pypi.org/project/requests"
        result, stats = redact_internal_urls(text, ["corp.internal"])
        assert result == text

"""Tests for token extraction from diffs."""

from __future__ import annotations

from mr_lead_agent.retrieval import extract_tokens


SAMPLE_DIFF = """\
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,10 @@
+import jwt
+
+SECRET_KEY = 'super_secret'
+
+def login(username, password):
+    token = jwt.encode({'user': username}, SECRET_KEY)
+    return token
"""


class TestExtractTokens:
    def test_extracts_identifiers(self) -> None:
        tokens = extract_tokens(SAMPLE_DIFF, [])
        assert "login" in tokens
        assert "username" in tokens
        assert "jwt" in tokens

    def test_extracts_filename_parts(self) -> None:
        tokens = extract_tokens(SAMPLE_DIFF, [])
        assert "auth.py" in tokens or "src" in tokens

    def test_trigger_words_included_when_present(self) -> None:
        tokens = extract_tokens(SAMPLE_DIFF, ["secret", "token"])
        # "secret" appears in diff (SECRET_KEY), "token" appears too
        assert "secret" in tokens or "SECRET_KEY" in tokens

    def test_trigger_words_not_included_when_absent(self) -> None:
        tokens = extract_tokens(SAMPLE_DIFF, ["kafka"])
        assert "kafka" not in tokens

    def test_no_stop_words(self) -> None:
        tokens = extract_tokens(SAMPLE_DIFF, [])
        assert "def" not in tokens
        assert "return" not in tokens
        assert "import" not in tokens

    def test_empty_diff_returns_empty(self) -> None:
        assert extract_tokens("", []) == []

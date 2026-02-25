"""Security filtering and secret redaction for content sent to the LLM."""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for secret detection
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # Key/token/secret assignments: KEY=value or key: value
    re.compile(
        r'(?i)(?P<key>(?:api[_-]?key|token|secret|password|passwd|pwd|auth)'
        r'\s*[:=]\s*)["\']?(?P<val>[A-Za-z0-9+/=_\-]{4,})["\']?',
        re.MULTILINE,
    ),
    # Bearer tokens in HTTP headers
    re.compile(r'(?i)bearer\s+(?P<val>[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+)', re.MULTILINE),
    # RSA/private key blocks
    re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----', re.DOTALL),
]

# Default deny-glob patterns (always excluded regardless of user config)
_DEFAULT_DENY_GLOBS: list[str] = [
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_ed25519",
    "**/*.p12",
    "**/*.pfx",
    "**/secrets/**",
    "**/.secrets",
]


@dataclass
class RedactionStats:
    """Counts of redacted items."""

    secrets_replaced: int = 0
    urls_replaced: int = 0
    files_excluded: int = 0
    patterns: list[str] = field(default_factory=list)


def _glob_match(path: str, pattern: str) -> bool:
    """Match a file path against a glob pattern (supporting ** prefix)."""
    # Also try matching without the leading **/ prefix
    stripped_pattern = pattern
    while stripped_pattern.startswith("**/"):
        stripped_pattern = stripped_pattern[3:]
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
        path.lstrip("/"), stripped_pattern
    )


def should_exclude_file(
    path: str,
    deny_globs: list[str],
    allow_dirs: list[str],
) -> bool:
    """Return True if the file should be excluded from LLM context.

    Args:
        path: Repository-relative file path.
        deny_globs: Glob patterns to always exclude.
        allow_dirs: If non-empty, only these directory prefixes are allowed.
    """
    combined = list(_DEFAULT_DENY_GLOBS) + deny_globs
    for pattern in combined:
        if _glob_match(path, pattern):
            logger.debug("Excluding file (deny glob %r): %s", pattern, path)
            return True

    if allow_dirs:
        allowed = any(path.startswith(d.rstrip("/") + "/") or path == d for d in allow_dirs)
        if not allowed:
            logger.debug("Excluding file (not in allow-dirs): %s", path)
            return True

    return False


def redact_secrets(text: str) -> tuple[str, RedactionStats]:
    """Mask secret values in text.

    Returns the redacted text and statistics about what was replaced.
    """
    stats = RedactionStats()
    result = text

    for pattern in _SECRET_PATTERNS:
        def _replace(m: re.Match[str]) -> str:
            stats.secrets_replaced += 1
            # Keep the key name, mask only the value
            if "key" in m.groupdict() and m.group("key"):
                return m.group("key") + "***REDACTED***"
            return "***REDACTED***"

        new_result = pattern.sub(_replace, result)
        if new_result != result:
            stats.patterns.append(pattern.pattern[:60])
        result = new_result

    if stats.secrets_replaced:
        logger.info("Redacted %d secret(s)", stats.secrets_replaced)

    return result, stats


def redact_internal_urls(
    text: str,
    internal_domains: list[str],
) -> tuple[str, RedactionStats]:
    """Mask URLs that match any of the given internal domain suffixes.

    Args:
        text: Text to process.
        internal_domains: List of domain substrings, e.g. ['gitlab.internal', '.corp.com'].
    """
    stats = RedactionStats()
    if not internal_domains:
        return text, stats

    pattern = re.compile(
        r'https?://(?:[^/\s\'"<>]*(?:' + "|".join(re.escape(d) for d in internal_domains) + r')[^\s\'"<>]*)',
        re.IGNORECASE,
    )

    def _replace_url(m: re.Match[str]) -> str:
        stats.urls_replaced += 1
        return "http://***INTERNAL-URL-REDACTED***"

    result = pattern.sub(_replace_url, text)
    if stats.urls_replaced:
        logger.info("Redacted %d internal URL(s)", stats.urls_replaced)

    return result, stats

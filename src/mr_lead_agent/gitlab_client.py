"""Async GitLab API client for fetching MR metadata and diffs."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from mr_lead_agent.models import MRData

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(30.0)


class GitLabAPIError(Exception):
    """Raised when the GitLab API returns an unexpected response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"GitLab API error {status_code}: {message}")


def _extract_project_path(repo_url: str) -> str:
    """Extract 'group/repo' path from an HTTPS GitLab repo URL.

    Examples:
        https://gitlab.example.com/group/repo.git  →  group/repo
        https://gitlab.example.com/a/b/c.git      →  a/b/c
    """
    # Strip trailing .git and split on the host
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Everything after the third slash (scheme://host/path)
    match = re.match(r"https?://[^/]+/(.+)", url)
    if not match:
        raise ValueError(f"Cannot extract project path from URL: {repo_url!r}")
    return match.group(1)


class GitLabClient:
    """Async client for the GitLab REST API v4."""

    def __init__(self, base_url: str, token: str, ssl_verify: bool = True) -> None:
        self._base = base_url.rstrip("/") + "/api/v4"
        self._token = token
        self._client = httpx.AsyncClient(
            headers={"PRIVATE-TOKEN": token},
            timeout=_DEFAULT_TIMEOUT,
            verify=ssl_verify,
        )

    async def __aenter__(self) -> GitLabClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        logger.debug("GET %s params=%s", url, params)
        resp = await self._client.get(url, params=params)
        if resp.status_code >= 400:
            raise GitLabAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    async def get_mr_data(self, repo_url: str, mr_iid: int) -> MRData:
        """Fetch MR metadata plus unified diff and return a MRData object."""
        project_path = _extract_project_path(repo_url)
        encoded = quote_plus(project_path)

        # --- MR metadata ---
        logger.info("Fetching MR !%d metadata for %s", mr_iid, project_path)
        meta: dict[str, Any] = await self._get(
            f"/projects/{encoded}/merge_requests/{mr_iid}"
        )
        logger.debug(
            "MR meta: title=%r author=%r source=%r→%r sha=%s state=%s",
            meta.get("title"),
            meta.get("author", {}).get("username"),
            meta.get("source_branch"),
            meta.get("target_branch"),
            str(meta.get("sha", ""))[:12],
            meta.get("state"),
        )

        # --- Changed files list ---
        logger.info("Fetching MR !%d changes", mr_iid)
        changes_data: dict[str, Any] = await self._get(
            f"/projects/{encoded}/merge_requests/{mr_iid}/changes"
        )
        changes = changes_data.get("changes", [])
        changed_files: list[str] = [c["new_path"] for c in changes]
        logger.debug(
            "Changes API: %d files, overflow=%s",
            len(changed_files),
            changes_data.get("overflow", False),
        )
        logger.debug("Changed files: %s", changed_files)

        # Build unified diff from per-file diffs
        diff_parts: list[str] = []
        for change in changes:
            raw_diff: str = change.get("diff", "")
            old_path: str = change.get("old_path", change["new_path"])
            new_path: str = change["new_path"]
            diff_parts.append(
                f"--- a/{old_path}\n+++ b/{new_path}\n{raw_diff}"
            )
        unified_diff = "\n".join(diff_parts)

        sha: str = meta.get("sha", "") or meta.get("diff_refs", {}).get("head_sha", "")
        logger.debug("Unified diff size: %d chars, %d lines", len(unified_diff), unified_diff.count("\n"))

        return MRData(
            title=meta.get("title", ""),
            description=meta.get("description") or "",
            author=meta.get("author", {}).get("username", "unknown"),
            source_branch=meta.get("source_branch", ""),
            target_branch=meta.get("target_branch", ""),
            web_url=meta.get("web_url", ""),
            sha=sha,
            iid=mr_iid,
            project_path=project_path,
            changed_files=changed_files,
            diff=unified_diff,
        )

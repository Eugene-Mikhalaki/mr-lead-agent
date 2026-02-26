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

    async def get_mr_notes(
        self, repo_url: str, mr_iid: int, diff: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch all discussion threads from a GitLab MR.

        Uses /discussions to capture threaded replies, file path, line number,
        and extracts a ±1 line code snippet from the diff for inline comments.

        Returns list of dicts representing threads (each has a 'notes' list).
        """
        project_path = _extract_project_path(repo_url)
        encoded = quote_plus(project_path)

        logger.info("Fetching MR !%d discussions", mr_iid)
        discussions: list[dict[str, Any]] = await self._get(
            f"/projects/{encoded}/merge_requests/{mr_iid}/discussions",
            params={"per_page": 100},
        )

        # Pre-parse diff into {file_path: {line_no: [lines_around]}}
        diff_lines_map = _build_diff_lines_map(diff) if diff else {}

        threads: list[dict[str, Any]] = []
        for discussion in discussions:
            thread_notes: list[dict[str, Any]] = []
            for note in discussion.get("notes", []):
                if note.get("system", False):
                    continue

                position: dict[str, Any] = note.get("position") or {}
                file_path: str = position.get("new_path") or position.get("old_path") or ""
                line: int = int(position.get("new_line") or position.get("old_line") or 0)

                # Extract code snippet from diff
                code_snippet = ""
                if file_path and line and file_path in diff_lines_map:
                    code_snippet = diff_lines_map[file_path].get(line, "")

                thread_notes.append({
                    "author": note.get("author", {}).get("username", ""),
                    "body": note.get("body", ""),
                    "created_at": note.get("created_at", ""),
                    "resolved": note.get("resolved", False),
                    "file_path": file_path,
                    "line": line,
                    "code_snippet": code_snippet,
                })

            if thread_notes:
                threads.append({"notes": thread_notes})

        total_notes = sum(len(t["notes"]) for t in threads)
        logger.info("Fetched %d threads (%d notes) from %d discussions",
                     len(threads), total_notes, len(discussions))
        return threads


def _build_diff_lines_map(diff: str) -> dict[str, dict[int, str]]:
    """Parse unified diff into {file_path: {new_line_no: '±1 line snippet'}}.

    Used to attach small code context to inline MR comments.
    """
    result: dict[str, list[tuple[int, str]]] = {}
    current_file = ""
    current_new_line = 0

    for raw_line in diff.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:]
            if current_file not in result:
                result[current_file] = []
        elif raw_line.startswith("@@ "):
            # Parse @@ -old,count +new,count @@
            parts = raw_line.split("+")
            if len(parts) >= 2:
                try:
                    current_new_line = int(parts[1].split(",")[0]) - 1
                except (ValueError, IndexError):
                    current_new_line = 0
        elif raw_line.startswith("-"):
            continue  # deleted line — doesn't increment new line counter
        else:
            current_new_line += 1
            if current_file:
                # Store line content (strip the leading '+' or ' ')
                content = raw_line[1:] if raw_line.startswith("+") else raw_line
                result.setdefault(current_file, []).append((current_new_line, content))

    # Build ±1 line snippets
    snippets: dict[str, dict[int, str]] = {}
    for file_path, lines in result.items():
        line_map: dict[int, str] = {ln: content for ln, content in lines}
        snippet_map: dict[int, str] = {}
        for ln, content in lines:
            context_lines = []
            for offset in (-1, 0, 1):
                target = ln + offset
                if target in line_map:
                    marker = "→ " if offset == 0 else "  "
                    context_lines.append(f"{target}: {marker}{line_map[target]}")
            snippet_map[ln] = "\n".join(context_lines)
        snippets[file_path] = snippet_map

    return snippets

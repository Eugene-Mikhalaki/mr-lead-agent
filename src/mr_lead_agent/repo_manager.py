"""Async repository manager: clone, fetch, and checkout git repositories."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 300  # seconds (increased from 120s for large monorepos)


async def _run_git(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run a git command asynchronously and return stdout."""
    cmd = ["git", *args]
    logger.debug("Running: %s (cwd=%s, env_keys=%s)", " ".join(cmd), cwd, list(env.keys()) if env else [])

    # Merge custom env into current system environment
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_GIT_TIMEOUT
        )
    except TimeoutError as err:
        proc.kill()
        raise RuntimeError(
            f"git {args[0]} timed out after {_GIT_TIMEOUT}s"
        ) from err

    if proc.returncode != 0:
        error = stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"git {args[0]} failed (exit {proc.returncode}): {error}"
        )

    return stdout.decode(errors="replace").strip()


class RepoManager:
    """Manages a local git clone of the target repository."""

    def __init__(self, workdir: str, ssl_verify: bool = True) -> None:
        self._workdir = Path(workdir)
        self._env = {}
        if not ssl_verify:
            self._env["GIT_SSL_NO_VERIFY"] = "true"

    @property
    def repo_path(self) -> Path:
        return self._workdir

    async def ensure_repo(self, repo_url: str) -> Path:
        """Clone the repo if it doesn't exist yet, otherwise fetch updates.

        Returns the path to the local repository.
        """
        if (self._workdir / ".git").exists():
            logger.info("Repo exists at %s — running git fetch", self._workdir)
            await _run_git(
                "fetch", "--all", "--prune", cwd=self._workdir, env=self._env
            )
        else:
            logger.info(
                "Cloning %s → %s", repo_url, self._workdir
            )
            self._workdir.mkdir(parents=True, exist_ok=True)
            await _run_git(
                "clone", repo_url, str(self._workdir), env=self._env
            )
        return self._workdir

    async def checkout_sha(self, sha: str) -> None:
        """Checkout a specific commit SHA (detached HEAD)."""
        logger.info("Checking out SHA %s", sha[:12])
        # Fetch the specific SHA in case it's not yet reachable
        try:
            await _run_git("fetch", "origin", sha, cwd=self._workdir, env=self._env)
        except RuntimeError:
            # Some GitLab setups don't allow fetching arbitrary SHAs by ref;
            # the SHA may already be present after fetch --all.
            logger.debug("Fetch by SHA failed — SHA may already be present")
        await _run_git("checkout", sha, "--", cwd=self._workdir, env=self._env)

    async def checkout_branch(self, branch: str) -> None:
        """Checkout a remote branch."""
        logger.info("Checking out branch %s", branch)
        await _run_git(
            "checkout", "-B", branch, f"origin/{branch}",
            cwd=self._workdir,
            env=self._env,
        )

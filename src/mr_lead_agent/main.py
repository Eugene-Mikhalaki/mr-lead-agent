"""CLI entry point and main review pipeline."""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings

# Suppress ArbitraryTypeWarning emitted by google-genai on import (their internal
# Pydantic models use `any` as a type — not our code, not actionable).
warnings.filterwarnings(
    "ignore",
    message=".*is not a Python type.*",
    category=UserWarning,
)

# Load .env into os.environ BEFORE click reads envvar= options.
# Must run before our package imports (which trigger google-genai import).
# python-dotenv is already available as a dep of pydantic-settings.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)  # override=False: real env vars take priority over .env

import click  # noqa: E402

from mr_lead_agent.config import Config  # noqa: E402
from mr_lead_agent.gitlab_client import GitLabAPIError, GitLabClient  # noqa: E402
from mr_lead_agent.llm_gemini import (  # noqa: E402
    call_deepseek,
    call_gemini,
    call_groq,
    call_openrouter,
)
from mr_lead_agent.models import PipelineStats, RedactionStats  # noqa: E402
from mr_lead_agent.prompt_builder import build_prompt  # noqa: E402
from mr_lead_agent.redaction import (  # noqa: E402
    redact_internal_urls,
    redact_secrets,
    should_exclude_file,
)
from mr_lead_agent.renderer import render_dry_run, render_report  # noqa: E402
from mr_lead_agent.repo_manager import RepoManager  # noqa: E402
from mr_lead_agent.retrieval import extract_tokens, search_context  # noqa: E402


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers unless DEBUG is explicitly requested
    if numeric > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--repo-url", required=True, envvar="REPO_URL", help="HTTPS URL of the GitLab repository")
@click.option("--mr-iid", required=True, type=int, envvar="MR_IID", help="MR IID (project-scoped)")
@click.option("--gitlab-base-url", required=True, envvar="GITLAB_BASE_URL", help="GitLab base URL")
@click.option("--gitlab-token", required=True, envvar="GITLAB_TOKEN", help="GitLab personal access token")
@click.option("--gemini-api-key", default="", envvar="GEMINI_API_KEY", help="Google Gemini API key")
@click.option("--model", default="gemini-2.0-flash", show_default=True, envvar="GEMINI_MODEL", help="Google Gemini model to use")
@click.option("--llm-provider", default="gemini", show_default=True, envvar="LLM_PROVIDER", type=click.Choice(["gemini", "deepseek", "openrouter", "groq"], case_sensitive=False), help="LLM provider")
@click.option("--deepseek-api-key", default="", envvar="DEEPSEEK_API_KEY", help="DeepSeek API key")
@click.option("--deepseek-model", default="deepseek-chat", show_default=True, envvar="DEEPSEEK_MODEL", help="DeepSeek model")
@click.option("--openrouter-api-key", default="", envvar="OPENROUTER_API_KEY", help="OpenRouter API key")
@click.option("--openrouter-model", default="qwen/qwen3-coder:free", show_default=True, envvar="OPENROUTER_MODEL", help="OpenRouter model")
@click.option("--groq-api-key", default="", envvar="GROQ_API_KEY", help="Groq API key (free tier)")
@click.option("--groq-model", default="llama-3.3-70b-versatile", show_default=True, envvar="GROQ_MODEL", help="Groq model")
@click.option("--target-branch", default="main", show_default=True, envvar="TARGET_BRANCH")
@click.option("--workdir", default=None, envvar="WORKDIR", help="Override repo cache directory")
@click.option("--max-blockers", default=10, show_default=True, type=int)
@click.option("--max-context-fragments", default=12, show_default=True, type=int)
@click.option("--max-fragment-lines", default=160, show_default=True, type=int)
@click.option("--allow-dirs", multiple=True, envvar="ALLOW_DIRS", help="Allowed source directories")
@click.option("--deny-globs", multiple=True, envvar="DENY_GLOBS", help="File glob patterns to exclude")
@click.option("--dry-run", is_flag=True, default=False, help="Print prompt & stats, skip LLM call")
@click.option("--no-verify-ssl", is_flag=True, default=False, envvar="NO_VERIFY_SSL", help="Disable SSL certificate verification (for self-signed certs)")
@click.option("--log-level", default="INFO", show_default=True)
@click.version_option()
def cli(
    repo_url: str,
    mr_iid: int,
    gitlab_base_url: str,
    gitlab_token: str,
    gemini_api_key: str,
    model: str,
    llm_provider: str,
    deepseek_api_key: str,
    deepseek_model: str,
    openrouter_api_key: str,
    openrouter_model: str,
    groq_api_key: str,
    groq_model: str,
    target_branch: str,
    workdir: str | None,
    max_blockers: int,
    max_context_fragments: int,
    max_fragment_lines: int,
    allow_dirs: tuple[str, ...],
    deny_globs: tuple[str, ...],
    dry_run: bool,
    no_verify_ssl: bool,
    log_level: str,
) -> None:
    """Lead Review Agent — automated GitLab MR code review via Gemini."""
    _setup_logging(log_level)

    config = Config(
        repo_url=repo_url,
        mr_iid=mr_iid,
        gitlab_base_url=gitlab_base_url,
        gitlab_token=gitlab_token,
        gemini_api_key=gemini_api_key,
        model=model,
        llm_provider=llm_provider,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        target_branch=target_branch,
        workdir=workdir,
        max_blockers=max_blockers,
        max_context_fragments=max_context_fragments,
        max_fragment_lines=max_fragment_lines,
        allow_dirs=list(allow_dirs),
        deny_globs=list(deny_globs),
        dry_run=dry_run,
        log_level=log_level,
        no_verify_ssl=no_verify_ssl,
    )

    asyncio.run(run_review(config))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_review(config: Config) -> None:
    """Execute the full MR review pipeline."""
    log = logging.getLogger(__name__)
    stats = PipelineStats()

    # ------------------------------------------------------------------
    # Step 1: Fetch MR data from GitLab
    # ------------------------------------------------------------------
    log.info("[Step 1/5] Fetching MR data from GitLab")
    try:
        async with GitLabClient(
            config.gitlab_base_url,
            config.gitlab_token,
            ssl_verify=not config.no_verify_ssl,
        ) as gl:
            mr_data = await gl.get_mr_data(config.repo_url, config.mr_iid)
    except GitLabAPIError as exc:
        _fail(f"GitLab API error: {exc}")
    except Exception as exc:
        _fail(f"Failed to fetch MR data: {exc}")
        return  # unreachable but satisfies type checker

    stats.diff_lines = mr_data.diff.count("\n")
    stats.summary_only_mode = stats.diff_lines > config.max_diff_lines_full_mode
    log.info("MR !%d — %d diff lines, %d files", mr_data.iid, stats.diff_lines, len(mr_data.changed_files))

    # ------------------------------------------------------------------
    # Step 2: Clone / fetch repository
    # ------------------------------------------------------------------
    log.info("[Step 2/5] Syncing local repository")
    log.debug("Workdir: %s", config.effective_workdir)
    try:
        repo_mgr = RepoManager(
            config.effective_workdir,
            ssl_verify=not config.no_verify_ssl,
        )
        repo_path = await repo_mgr.ensure_repo(config.repo_url)
        log.debug("Repo path: %s", repo_path)
        if mr_data.sha:
            await repo_mgr.checkout_sha(mr_data.sha)
    except RuntimeError as exc:
        _fail(f"Git error: {exc}")
        return

    # ------------------------------------------------------------------
    # Step 3: Retrieval — lexical context search
    # ------------------------------------------------------------------
    log.info("[Step 3/5] Running lexical retrieval")
    try:
        tokens = extract_tokens(mr_data.diff, config.trigger_words)
        log.debug("Extracted %d tokens: %s", len(tokens), tokens[:20])
        raw_fragments = await search_context(repo_path, tokens, config)
        log.debug("Retrieved %d raw fragments", len(raw_fragments))
    except Exception as exc:
        log.warning("Retrieval failed (continuing without context): %s", exc)
        raw_fragments = []

    # ------------------------------------------------------------------
    # Step 4: Redaction — filter and mask sensitive content
    # ------------------------------------------------------------------
    log.info("[Step 4/5] Applying redaction")
    redaction = RedactionStats()

    # Filter excluded files from the diff context
    safe_fragments = []
    for frag in raw_fragments:
        if should_exclude_file(frag.file_path, config.deny_globs, config.allow_dirs):
            redaction.files_excluded += 1
        else:
            excerpt, sec_stats = redact_secrets(frag.code_excerpt)
            redaction.secrets_replaced += sec_stats.secrets_replaced
            excerpt, url_stats = redact_internal_urls(excerpt, [])
            redaction.urls_replaced += url_stats.urls_replaced
            safe_fragments.append(frag.model_copy(update={"code_excerpt": excerpt}))

    # Redact the diff itself
    clean_diff, diff_sec = redact_secrets(mr_data.diff)
    redaction.secrets_replaced += diff_sec.secrets_replaced
    mr_data = mr_data.model_copy(update={"diff": clean_diff})

    stats.redaction = redaction
    stats.context_fragments = len(safe_fragments)
    stats.context_files = len({f.file_path for f in safe_fragments})
    stats.context_lines = sum(
        f.line_end - f.line_start for f in safe_fragments
    )

    # ------------------------------------------------------------------
    # Step 5: Build prompt (and stop here if dry-run)
    # ------------------------------------------------------------------
    log.info("[Step 5/5] Building prompt")
    prompt = build_prompt(mr_data, safe_fragments, config)

    if config.dry_run:
        render_dry_run(mr_data, prompt, stats)
        return

    # ------------------------------------------------------------------
    # Step 6: Call LLM
    # ------------------------------------------------------------------
    if config.llm_provider == "deepseek":
        log.info("[Step 6] Calling DeepSeek (%s)", config.deepseek_model)
        result = await call_deepseek(
            prompt, config.deepseek_api_key, stats, model=config.deepseek_model
        )
    elif config.llm_provider == "openrouter":
        log.info("[Step 6] Calling OpenRouter (%s)", config.openrouter_model)
        result = await call_openrouter(
            prompt, config.openrouter_api_key, stats, model=config.openrouter_model
        )
    elif config.llm_provider == "groq":
        log.info("[Step 6] Calling Groq (%s)", config.groq_model)
        result = await call_groq(
            prompt, config.groq_api_key, stats, model=config.groq_model
        )
    else:
        log.info("[Step 6] Calling Gemini (%s)", config.model)
        result = await call_gemini(prompt, config.gemini_api_key, stats, model=config.model)

    # Enforce max_blockers
    if len(result.blockers) > config.max_blockers:
        result = result.model_copy(
            update={"blockers": result.blockers[: config.max_blockers]}
        )

    # ------------------------------------------------------------------
    # Step 7: Render output
    # ------------------------------------------------------------------
    render_report(mr_data, result, stats, save_runs=config.save_runs)


def _fail(message: str) -> None:
    logging.getLogger(__name__).error(message)
    sys.exit(1)

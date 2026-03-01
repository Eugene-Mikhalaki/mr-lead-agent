"""Configuration management via Pydantic Settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """All runtime configuration for the review pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- TLS ---
    no_verify_ssl: bool = Field(
        False,
        description="Disable SSL certificate verification (for self-signed certs).",
    )

    # --- Required ---
    repo_url: str = Field(..., description="HTTPS URL of the GitLab repository")
    mr_iid: int = Field(..., description="MR IID (project-scoped)")
    gitlab_base_url: str = Field(..., description="GitLab base URL")
    gitlab_token: str = Field(..., description="GitLab personal access token")
    gemini_api_key: str = Field("", description="Google Gemini API key")
    model: str = Field("gemini-2.0-flash", description="Google Gemini model to use")

    # --- DeepSeek ---
    deepseek_api_key: str = Field("", description="DeepSeek API key")
    deepseek_model: str = Field("deepseek-chat", description="DeepSeek model to use")

    # --- OpenRouter ---
    openrouter_api_key: str = Field("", description="OpenRouter API key")
    openrouter_model: str = Field(
        "qwen/qwen3-coder:free",
        description="OpenRouter model (append ':free' for free tier)",
    )

    # --- Groq ---
    groq_api_key: str = Field("", description="Groq API key (free tier available)")
    groq_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model (llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, ...)",
    )

    # --- LLM provider selector ---
    llm_provider: str = Field(
        "gemini",
        description="LLM provider: 'gemini' | 'deepseek' | 'openrouter' | 'groq'",
    )

    # --- Optional with defaults from spec ---
    target_branch: str = Field("main", description="Target branch name")
    workdir: str | None = Field(None, description="Override working directory for repo cache")
    max_blockers: int = Field(10, ge=1, le=50)
    max_fragment_lines: int = Field(160, ge=10, le=500)
    max_diff_lines_full_mode: int = Field(3000, ge=100)

    # --- Dynamic token budget ---
    max_prompt_tokens: int = Field(
        120_000, description="Total prompt token budget",
    )
    max_context_tokens: int = Field(
        60_000, description="Ceiling for RETRIEVED CONTEXT section tokens",
    )
    token_rate: float = Field(
        0.35, description="Chars-to-tokens coefficient for budget estimation",
    )

    # Retrieval trigger words (configurable)
    trigger_words: list[str] = Field(
        default=[
            "timeout", "retry", "transaction", "alembic", "migration",
            "auth", "permission", "token", "secret", "sql", "kafka",
        ]
    )

    # Security filters
    allow_dirs: list[str] = Field(default_factory=list)
    deny_globs: list[str] = Field(
        default=[
            "**/.env", "**/*.pem", "**/*.key", "**/id_rsa",
            "**/id_ed25519", "**/*.p12", "**/*.pfx",
        ]
    )

    # Output
    dry_run: bool = Field(False, description="Print prompt & stats, skip LLM call")
    debug: bool = Field(False, description="Save prompt to runs/prompts/ for inspection")
    log_level: str = Field("INFO")
    save_runs: bool = Field(True, description="Save result JSON to ./runs/")

    # Review settings
    reviewer_username: str = Field(
        "",
        description="Your GitLab username — comments from this user are excluded from LLM prompt",
    )
    review_language: str = Field(
        "en",
        description="Language for review output: 'ru' or 'en'",
    )
    review_rules_file: str = Field(
        "./review_rules.md",
        description="Path to markdown file with coding standards and review checklist",
    )

    @property
    def effective_workdir(self) -> str:
        """Compute workdir from repo_url if not explicitly set."""
        if self.workdir:
            return self.workdir
        # Extract project name: https://host/group/repo.git → repo
        name = self.repo_url.rstrip("/").rstrip(".git").rsplit("/", 1)[-1]
        return f"./repos/{name}"

"""Pydantic models for MR data, context fragments, and review results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MRData(BaseModel):
    """Metadata and diff for a GitLab Merge Request."""

    title: str
    description: str
    author: str
    source_branch: str
    target_branch: str
    web_url: str
    sha: str
    iid: int
    project_path: str
    changed_files: list[str] = Field(default_factory=list)
    diff: str = ""


class ContextFragment(BaseModel):
    """A code snippet retrieved from the repository for LLM context."""

    file_path: str
    line_start: int
    line_end: int
    code_excerpt: str
    token_match: str = ""
    fragment_type: str = "usage"  # "definition" | "pydantic_model" | "usage"
    priority: int = 50           # 0=highest, 100=lowest


class RedactionStats(BaseModel):
    """Statistics about redacted content."""

    secrets_replaced: int = 0
    urls_replaced: int = 0
    files_excluded: int = 0


class Risk(BaseModel):
    """A key risk identified in the MR."""

    severity: str  # "major" | "minor"
    title: str
    details: str


class Blocker(BaseModel):
    """A blocking issue in the MR."""

    severity: str = "blocker"
    file: str
    lines: str
    title: str
    comment: str = ""        # optional — LLM may omit it
    suggested_fix: str = ""
    verification: str = ""


class Question(BaseModel):
    """A question to the MR author."""

    file: str = ""
    lines: str = ""
    question: str
    why_it_matters: str


class MRNote(BaseModel):
    """A discussion comment from the MR."""

    author: str
    body: str
    created_at: str = ""
    resolved: bool = False
    file_path: str = ""    # non-empty for inline code comments
    line: int = 0          # source line for inline code comments
    code_snippet: str = "" # ±1 line of code around the commented line


class MRDiscussion(BaseModel):
    """A threaded discussion from the MR (one or more notes)."""

    notes: list[MRNote] = Field(default_factory=list)


class DiscussionReply(BaseModel):
    """LLM's response to a developer's comment/question in the MR."""

    original_author: str
    original_comment: str = ""
    reply: str


class ReviewResult(BaseModel):
    """Structured review result from the LLM."""

    summary: list[str] = Field(default_factory=list)
    key_risks: list[Risk] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    questions_to_author: list[Question] = Field(default_factory=list)
    discussion_replies: list[DiscussionReply] = Field(default_factory=list)


class PipelineStats(BaseModel):
    """Execution stats for the review pipeline."""

    diff_lines: int = 0
    context_fragments: int = 0
    context_files: int = 0
    context_lines: int = 0
    redaction: RedactionStats = Field(default_factory=RedactionStats)
    summary_only_mode: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

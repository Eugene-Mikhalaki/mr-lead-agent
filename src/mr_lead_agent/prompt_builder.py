"""Prompt builder: assembles the LLM prompt from MR data and context."""

from __future__ import annotations

import json
import logging

from mr_lead_agent.config import Config
from mr_lead_agent.models import ContextFragment, MRData

logger = logging.getLogger(__name__)

_ROLE_POLICY = """\
You are a senior tech-lead performing a code review of a GitLab Merge Request.
Your role:
- Be thorough and skeptical, but constructive.
- Do NOT comment on code style, formatting, or cosmetic issues.
- Every blocker MUST have a verifiable, specific rationale (not vague concerns).
- Do NOT invent context or assume things not present in the provided fragments.
- Prioritise security, correctness, and reliability issues.
"""

_OUTPUT_SCHEMA = {
    "summary": ["string — 2 to 7 bullet points summarising the change"],
    "key_risks": [
        {
            "severity": "major | minor",
            "title": "short title",
            "details": "explanation",
        }
    ],
    "blockers": [
        {
            "severity": "blocker",
            "file": "path/to/file.py",
            "lines": "start-end",
            "title": "short title",
            "comment": "detailed comment",
            "suggested_fix": "suggested change (optional)",
            "verification": "how to verify this is fixed",
        }
    ],
    "questions_to_author": [
        {
            "file": "path/to/file.py or empty",
            "lines": "line range or empty",
            "question": "the question",
            "why_it_matters": "why this is important",
        }
    ],
}

_SUMMARY_ONLY_NOTE = (
    "NOTE: The diff is very large (>{limit} lines). "
    "Focus on the summary and key risks. "
    "Limit blockers to the most critical issues only."
)


def build_prompt(
    mr_data: MRData,
    context_fragments: list[ContextFragment],
    config: Config,
) -> str:
    """Assemble the full LLM prompt.

    Returns a string prompt and logs size statistics.
    """
    diff_lines = mr_data.diff.count("\n")
    summary_only = diff_lines > config.max_diff_lines_full_mode

    parts: list[str] = []

    # --- 1. Role / Policy ---
    parts.append("## ROLE & POLICY\n" + _ROLE_POLICY)

    # --- 2. MR Metadata ---
    meta_block = f"""\
## MR METADATA
Title: {mr_data.title}
Author: {mr_data.author}
Source branch: {mr_data.source_branch} → {mr_data.target_branch}
URL: {mr_data.web_url}
SHA: {mr_data.sha}

Description:
{mr_data.description or "(no description)"}

Changed files ({len(mr_data.changed_files)}):
{chr(10).join(f"  - {f}" for f in mr_data.changed_files)}
"""
    parts.append(meta_block)

    # --- 3. Diff ---
    if summary_only:
        note = _SUMMARY_ONLY_NOTE.format(limit=config.max_diff_lines_full_mode)
        diff_section = f"## DIFF\n{note}\n\n{mr_data.diff}"
    else:
        diff_section = f"## DIFF\n{mr_data.diff}"
    parts.append(diff_section)

    # --- 4. Retrieved context ---
    if context_fragments:
        ctx_parts = ["## RETRIEVED CONTEXT\n"]
        for i, frag in enumerate(context_fragments, 1):
            ctx_parts.append(
                f"### [{i}] FILE: {frag.file_path}  LINES: {frag.line_start}-{frag.line_end}"
                f"  (matched: {frag.token_match!r})\n"
                f"```\n{frag.code_excerpt}\n```\n"
            )
        parts.append("\n".join(ctx_parts))
    else:
        parts.append("## RETRIEVED CONTEXT\n(none)")

    # --- 5. Output contract ---
    schema_json = json.dumps(_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)
    output_block = f"""\
## OUTPUT CONTRACT
Return ONLY valid JSON matching this schema (no markdown fences, no extra text):

{schema_json}

Constraints:
- blockers: at most {config.max_blockers} items
- summary: 2-7 items
- questions_to_author: 0-10 items
"""
    parts.append(output_block)

    prompt = "\n\n".join(parts)
    logger.info(
        "Prompt built: %d chars, %d diff lines, %d context fragments, summary_only=%s",
        len(prompt), diff_lines, len(context_fragments), summary_only,
    )
    return prompt

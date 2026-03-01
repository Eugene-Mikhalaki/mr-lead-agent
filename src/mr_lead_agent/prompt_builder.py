"""Prompt builder: assembles the LLM prompt from MR data and context."""

from __future__ import annotations

import json
import logging

from mr_lead_agent.config import Config
from mr_lead_agent.models import ContextFragment, MRData, MRDiscussion

logger = logging.getLogger(__name__)

_ROLE_POLICY = """\
You are a senior tech-lead performing a code review of a GitLab Merge Request.

## Instructions

1. Analyse only new/changed code for the task, but study related old code if it is relevant to the changes.
2. Reference exact line numbers for every issue found.
3. **Formulate ALL remarks as questions to the developer** — never as assertions or commands.
   Example: ❓ "Could there be a race condition here if two requests arrive simultaneously?"
   NOT: "This has a race condition."
4. Group remarks by file.
5. Prioritise: critical → important → style.

## Rules

- Do NOT comment on trivial style or formatting issues unless they violate the provided coding rules.
- Every blocker MUST have a verifiable, specific rationale (not vague concerns).
- Do NOT invent context or assume things not present in the provided fragments.
- Prioritise security, correctness, and reliability issues.
"""

_LANGUAGE_INSTRUCTIONS = {
    "ru": (
        "IMPORTANT: Write ALL your output (summary, risks, blockers, questions, replies) "
        "in Russian language. All field values in the JSON must be in Russian."
    ),
    "en": "Write all output in English.",
}

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
            "title": "short title as a question (e.g. 'Could this cause a race condition?')",
            "comment": "detailed comment formulated as a question",
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
    "discussion_replies": [
        {
            "original_author": "username of the comment author",
            "original_comment": "first 100 chars of the original comment",
            "reply": "your answer to their question/comment",
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
    discussions: list[MRDiscussion] | None = None,
    rules_content: str | None = None,
) -> str:
    """Assemble the full LLM prompt.

    Returns a string prompt and logs size statistics.
    """
    diff_lines = mr_data.diff.count("\n")
    summary_only = diff_lines > config.max_diff_lines_full_mode

    parts: list[str] = []

    # --- 1. Role / Policy ---
    parts.append("## ROLE & POLICY\n" + _ROLE_POLICY)

    # --- 2. Language instruction ---
    lang_key = config.review_language.lower() if config.review_language else "en"
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(lang_key, _LANGUAGE_INSTRUCTIONS["en"])
    parts.append(f"## LANGUAGE\n{lang_instruction}")

    # --- 3. Coding rules (if provided) ---
    if rules_content:
        parts.append(f"## CODING RULES & CHECKLIST\n\n{rules_content}")

    # --- 4. MR Metadata ---
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

    # --- 5. MR Discussion Threads ---
    if discussions:
        reviewer = config.reviewer_username
        disc_block = ["## MR DISCUSSION THREADS\n"]
        disc_block.append(
            "Below are threaded discussions from the MR. "
            "Each thread groups related replies together. "
            "Reply to developer questions/comments in the `discussion_replies` section of your output.\n"
        )
        if reviewer:
            disc_block.append(
                f"**@{reviewer}** is the reviewer (you are acting on their behalf). "
                "Do NOT reply to their comments — they are provided as context only. "
                "Reply ONLY to comments from other participants (developers).\n"
            )
        for t_idx, thread in enumerate(discussions, 1):
            disc_block.append(f"### Thread #{t_idx}")
            # Show code snippet from the first note that has one
            first_inline = next(
                (n for n in thread.notes if n.file_path and n.line), None
            )
            if first_inline:
                loc = f"`{first_inline.file_path}` line {first_inline.line}"
                disc_block.append(f"📍 Location: {loc}")
                if first_inline.code_snippet:
                    disc_block.append(f"```\n{first_inline.code_snippet}\n```")

            for n_idx, note in enumerate(thread.notes):
                tag = " [REVIEWER]" if reviewer and note.author == reviewer else ""
                indent = "  ↳ " if n_idx > 0 else ""
                date_str = note.created_at[:10] if note.created_at else ""
                disc_block.append(
                    f"{indent}**@{note.author}**{tag} ({date_str}):\n{indent}> {note.body}\n"
                )
            disc_block.append("")  # blank line between threads
        parts.append("\n".join(disc_block))

    # --- 6. Diff ---
    if summary_only:
        note = _SUMMARY_ONLY_NOTE.format(limit=config.max_diff_lines_full_mode)
        diff_section = f"## DIFF\n{note}\n\n{mr_data.diff}"
    else:
        diff_section = f"## DIFF\n{mr_data.diff}"
    parts.append(diff_section)

    # --- 7. Output contract ---
    schema_json = json.dumps(_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)
    output_block = f"""\
## OUTPUT CONTRACT
Return ONLY valid JSON matching this schema (no markdown fences, no extra text):

{schema_json}

Constraints:
- blockers: at most {config.max_blockers} items
- summary: 2-7 items
- questions_to_author: 0-10 items
- discussion_replies: one reply per developer comment (if any comments provided above)
- All blocker titles and comments must be formulated as questions
"""
    parts.append(output_block)

    # --- Base prompt (everything except RETRIEVED CONTEXT) ---
    base_prompt = "\n\n".join(parts)

    # --- 8. Dynamic budget — fill RETRIEVED CONTEXT ---
    base_tokens = int(len(base_prompt) * config.token_rate)
    context_budget_tokens = min(
        config.max_prompt_tokens - base_tokens,
        config.max_context_tokens,
    )
    context_budget_chars = int(context_budget_tokens / config.token_rate)

    selected: list[ContextFragment] = []
    used_chars = 0

    if context_fragments:
        # Fragments are already sorted by priority from retrieval
        for frag in context_fragments:
            # Header overhead: "### [N] FILE: ... LINES: ... (type: ...)\n```\n...\n```\n"
            frag_chars = len(frag.code_excerpt) + 100
            if used_chars + frag_chars > context_budget_chars:
                break
            selected.append(frag)
            used_chars += frag_chars

    if selected:
        ctx_parts = ["## RETRIEVED CONTEXT\n"]
        for i, frag in enumerate(selected, 1):
            type_label = frag.fragment_type
            ctx_parts.append(
                f"### [{i}] FILE: {frag.file_path}  LINES: {frag.line_start}-{frag.line_end}"
                f"  (type: {type_label}, matched: {frag.token_match!r})\n"
                f"```\n{frag.code_excerpt}\n```\n"
            )
        parts.append("\n".join(ctx_parts))
    else:
        parts.append("## RETRIEVED CONTEXT\n(none)")

    prompt = "\n\n".join(parts)

    total_notes = sum(len(t.notes) for t in discussions) if discussions else 0
    used_tokens = int(used_chars * config.token_rate)
    logger.info(
        "Prompt built: %d chars (~%d tokens), %d diff lines, "
        "%d context fragments selected (%d/%d budget tokens), "
        "%d discussion threads (%d notes), summary_only=%s",
        len(prompt), int(len(prompt) * config.token_rate),
        diff_lines, len(selected), used_tokens, context_budget_tokens,
        len(discussions) if discussions else 0, total_notes, summary_only,
    )
    return prompt

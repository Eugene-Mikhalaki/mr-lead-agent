"""Lexical retrieval: extract tokens from diff and search repo with ripgrep."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from mr_lead_agent.config import Config
from mr_lead_agent.models import ContextFragment
from mr_lead_agent.redaction import should_exclude_file

logger = logging.getLogger(__name__)

# Regex for identifiers: \b[A-Za-z_][A-Za-z0-9_]{2,}\b
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")

# Words to skip (too generic to be useful)
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "def", "class", "self", "return", "import", "from", "with",
        "None", "True", "False", "pass", "raise", "async", "await",
        "elif", "else", "and", "not", "for", "while", "try", "except",
        "finally", "lambda", "yield", "global", "assert",
    ]
)


def extract_tokens(diff: str, trigger_words: list[str]) -> list[str]:
    """Extract unique tokens from a unified diff.

    Includes:
    - File/directory names from diff headers
    - Identifiers from changed lines (added/removed)
    - Configured trigger words present in the diff
    """
    tokens: set[str] = set()

    # File paths from diff headers
    for line in diff.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line.split("/", 1)[-1].strip()
            tokens.update(Path(path).parts)

        # Changed lines only
        elif line.startswith(("+", "-")) and not line.startswith(("---", "+++")):
            for match in _IDENT_RE.finditer(line[1:]):
                word = match.group(1)
                if word not in _STOP_WORDS and len(word) <= 80:
                    tokens.add(word)

    # Always include trigger words found in the diff
    diff_lower = diff.lower()
    for word in trigger_words:
        if word.lower() in diff_lower:
            tokens.add(word)

    logger.debug("Extracted %d tokens from diff", len(tokens))
    return sorted(tokens)


async def _run_rg(
    pattern: str,
    repo_path: Path,
    context_lines: int = 4,
) -> list[dict[str, object]]:
    """Run ripgrep and return matches with surrounding context lines.

    Each result dict contains:
        - path: absolute file path string
        - line_number: line number of the match itself
        - line_start: first line of the excerpt window
        - line_end: last line of the excerpt window
        - excerpt: multi-line string (context before + match + context after)
    """
    cmd = [
        "rg",
        "--json",
        "--context", str(context_lines),
        "--max-count", "5",
        "--type-add", "code:*.{py,js,ts,sql,yaml,yml,json,md}",
        "--type", "code",
        "--",
        pattern,
        str(repo_path),
    ]
    logger.debug("rg pattern=%r", pattern)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        logger.warning("ripgrep timed out for pattern %r", pattern)
        return []

    # Parse all events into a flat list
    events: list[dict[str, object]] = []
    for raw_line in stdout.splitlines():
        try:
            obj = json.loads(raw_line)
            events.append(obj)
        except json.JSONDecodeError:
            continue

    # Group events by file block and build excerpts around each match
    results: list[dict[str, object]] = []
    current_path: str = ""
    # Buffer: list of (type, line_number, text) for the current file block
    block: list[tuple[str, int, str]] = []

    for evt in events:
        t = str(evt.get("type", ""))
        data = evt.get("data", {})
        assert isinstance(data, dict)

        if t == "begin":
            current_path = str(data.get("path", {}).get("text", ""))  # type: ignore[union-attr]
            block = []
        elif t in ("context", "match"):
            line_num = int(str(data.get("line_number", 0)))
            line_text: str = str(data.get("lines", {}).get("text", ""))  # type: ignore[union-attr]
            block.append((t, line_num, line_text))
        elif t == "end":
            # Find all match positions in block and build an excerpt for each
            for idx, (btype, bline, _) in enumerate(block):
                if btype != "match":
                    continue
                # Take context_lines before and after in the block
                start = max(0, idx - context_lines)
                end = min(len(block), idx + context_lines + 1)
                window = block[start:end]
                excerpt = "".join(text for _, _, text in window)
                results.append({
                    "path": current_path,
                    "line_number": bline,
                    "line_start": window[0][1],
                    "line_end": window[-1][1],
                    "excerpt": excerpt,
                })
            block = []

    return results


def _deduplicate(
    fragments: list[ContextFragment],
    max_per_file: int,
) -> list[ContextFragment]:
    """Remove near-duplicate fragments and enforce per-file limits."""
    seen_keys: set[tuple[str, int]] = set()
    per_file: dict[str, int] = {}
    result: list[ContextFragment] = []

    for frag in fragments:
        key = (frag.file_path, frag.line_start // 20)  # bucket by ~20-line window
        if key in seen_keys:
            continue
        count = per_file.get(frag.file_path, 0)
        if count >= max_per_file:
            continue
        seen_keys.add(key)
        per_file[frag.file_path] = count + 1
        result.append(frag)

    return result


async def search_context(
    repo_path: Path,
    tokens: list[str],
    config: Config,
) -> list[ContextFragment]:
    """Run ripgrep for each token and return deduplicated context fragments."""
    all_fragments: list[ContextFragment] = []
    total_chars = 0

    for token in tokens:
        if total_chars >= config.max_context_chars:
            logger.info("Context char limit reached, stopping retrieval")
            break

        matches = await _run_rg(rf"\b{re.escape(token)}\b", repo_path)

        for match_data in matches:
            file_path_abs = str(match_data.get("path", ""))
            # Make path relative to repo root
            try:
                rel_path = str(Path(file_path_abs).relative_to(repo_path))
            except ValueError:
                rel_path = file_path_abs

            if should_exclude_file(rel_path, config.deny_globs, config.allow_dirs):
                continue

            excerpt: str = str(match_data.get("excerpt", ""))
            if not excerpt.strip():
                continue

            line_start: int = int(str(match_data.get("line_start", 1)))
            line_end: int = int(str(match_data.get("line_end", line_start)))

            # Enforce per-fragment line limit
            excerpt_trimmed = "\n".join(
                excerpt.splitlines()[: config.max_fragment_lines]
            )

            frag = ContextFragment(
                file_path=rel_path,
                line_start=line_start,
                line_end=line_end,
                code_excerpt=excerpt_trimmed,
                token_match=token,
            )
            all_fragments.append(frag)
            total_chars += len(excerpt_trimmed)

    deduped = _deduplicate(all_fragments, config.max_fragments_per_file)
    limited = deduped[: config.max_context_fragments]
    logger.info(
        "Retrieval: %d fragments (from %d raw), %d tokens searched",
        len(limited), len(all_fragments), len(tokens),
    )
    return limited

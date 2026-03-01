"""Lexical retrieval: extract tokens from diff and search repo with ripgrep + AST."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from mr_lead_agent.ast_extractor import (
    extract_dockerfile_block,
    extract_python_definitions,
    extract_yaml_block,
)
from mr_lead_agent.config import Config
from mr_lead_agent.models import ContextFragment
from mr_lead_agent.redaction import should_exclude_file

logger = logging.getLogger(__name__)

# Regex for identifiers: \b[A-Za-z_][A-Za-z0-9_]{4,}\b (min 5 chars)
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{4,})\b")

# Words to skip (too generic to be useful)
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "def", "class", "self", "return", "import", "from", "with",
        "None", "True", "False", "pass", "raise", "async", "await",
        "elif", "else", "and", "not", "for", "while", "try", "except",
        "finally", "lambda", "yield", "global", "assert",
        # Generic code words
        "get", "set", "list", "json", "info", "log", "stat", "var",
        "idx", "bin", "host", "test", "image", "always", "build",
        "cache", "cli", "curl", "bash", "update", "next", "version",
        # Docker-compose/YAML noise
        "api", "docker", "compose", "condition", "container_name",
        "depends_on", "services", "healthcheck", "restart",
        "environment", "ports", "volumes", "networks", "depends",
        # Tokens that produce too many irrelevant matches
        "__init__", "localhost", "redis", "interval", "timeout",
        "retries", "start_period", "python", "sleep", "start_time",
        "dpage", "login", "install", "systemhooks",
    ]
)

# File extensions for structured extraction
_PYTHON_EXTS = {".py"}
_YAML_EXTS = {".yml", ".yaml"}
_DOCKER_NAMES = {"Dockerfile", "dockerfile"}


def extract_tokens(
    diff: str, trigger_words: list[str], changed_files: list[str] | None = None,
) -> list[str]:
    """Extract unique tokens from a unified diff.

    Includes:
    - File/directory names from diff headers
    - Identifiers from changed lines (added/removed)
    - Configured trigger words present in the diff

    Excludes tokens that are directory segments of changed_files (too broad).
    """
    tokens: set[str] = set()

    # Collect path segments from changed files to filter overly-broad tokens
    _changed_path_segments: set[str] = set()
    for cf in (changed_files or []):
        for part in Path(cf).parts:
            seg = part.replace(".py", "").replace(".yml", "").replace(".yaml", "")
            if len(seg) >= 5:
                _changed_path_segments.add(seg)
            # Also add underscore variant of hyphenated segments
            # e.g. 'squad-upload-aggregate-data' -> 'upload_aggregate_data'
            if "-" in seg:
                uscore = seg.replace("-", "_")
                if len(uscore) >= 5:
                    _changed_path_segments.add(uscore)
                # sub-segments after first hyphen part (strip prefix like 'squad-')
                parts_by_dash = seg.split("-")
                if len(parts_by_dash) > 1:
                    suffix = "_".join(parts_by_dash[1:])
                    if len(suffix) >= 5:
                        _changed_path_segments.add(suffix)

    # File paths from diff headers
    for line in diff.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line.split("/", 1)[-1].strip()
            tokens.update(Path(path).parts)

        # Changed lines only
        elif line.startswith(("+", "-")) and not line.startswith(("---", "+++")):
            for match in _IDENT_RE.finditer(line[1:]):
                word = match.group(1)
                if word not in _STOP_WORDS and not word.isupper():
                    tokens.add(word)

    # Add trigger words that appear in the diff
    diff_lower = diff.lower()
    for tw in trigger_words:
        if tw.lower() in diff_lower:
            tokens.add(tw)

    # Remove tokens that are directory segments of changed files
    # (they match every import in that package, producing massive noise)
    tokens -= _changed_path_segments

    logger.debug("Extracted %d tokens from diff", len(tokens))
    return sorted(tokens)


async def _run_rg(
    pattern: str,
    repo_path: Path,
    context_lines: int = 9,
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


def _overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return the fraction of the smaller range covered by overlap."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_len = overlap_end - overlap_start
    smaller_len = min(a_end - a_start, b_end - b_start)
    return overlap_len / max(smaller_len, 1)


def _deduplicate(fragments: list[ContextFragment]) -> list[ContextFragment]:
    """Remove duplicate, subset, and heavily-overlapping fragments.

    Three-pass approach:
    1. Skip fragments fully covered by a previously accepted one
    2. Skip fragments that overlap >50% with a previously accepted one
    3. Bucket dedup for remaining near-duplicates in the same 20-line window
    """
    accepted: list[ContextFragment] = []

    for frag in fragments:
        dominated = False
        for prev in accepted:
            if prev.file_path != frag.file_path:
                continue
            # Full subset
            if prev.line_start <= frag.line_start and prev.line_end >= frag.line_end:
                dominated = True
                break
            # Heavy overlap: drop the smaller (later-priority) one
            if _overlap_ratio(
                prev.line_start, prev.line_end,
                frag.line_start, frag.line_end,
            ) >= 0.4:
                dominated = True
                break
        if dominated:
            continue
        accepted.append(frag)

    # Bucket dedup for remaining near-duplicates
    seen_keys: set[tuple[str, int]] = set()
    result: list[ContextFragment] = []
    for frag in accepted:
        key = (frag.file_path, frag.line_start // 20)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(frag)

    return result


def _classify_file(file_path: str) -> str:
    """Classify file type for extraction strategy."""
    p = Path(file_path)
    if p.suffix in _PYTHON_EXTS:
        return "python"
    if p.suffix in _YAML_EXTS:
        return "yaml"
    if p.name in _DOCKER_NAMES or "dockerfile" in p.name.lower():
        return "dockerfile"
    return "other"


async def search_context(
    repo_path: Path,
    tokens: list[str],
    config: Config,
    changed_files: list[str] | None = None,
) -> list[ContextFragment]:
    """Search repository for context fragments with smart extraction.

    Pipeline:
    1. For each token, try AST-based extraction first (Python definitions)
    2. Fall back to ripgrep for usages
    3. Apply structured block extraction for YAML/Docker
    4. Tag each fragment with type + priority
    5. Sort by priority, deduplicate
    6. Return all fragments (budget trimming happens in prompt_builder)
    """
    if changed_files is None:
        changed_files = []

    diff_tokens = set(tokens)
    all_fragments: list[ContextFragment] = []
    seen_definitions: set[tuple[str, str]] = set()  # (file, token) already extracted via AST

    # ------------------------------------------------------------------
    # Pass 1: AST-based extraction for Python files in the repo
    # ------------------------------------------------------------------
    # Scan all Python files for definition matches
    python_files: list[Path] = []
    for ext in _PYTHON_EXTS:
        python_files.extend(repo_path.rglob(f"*{ext}"))

    for py_file in python_files:
        try:
            rel_path = str(py_file.relative_to(repo_path))
        except ValueError:
            continue
        if should_exclude_file(rel_path, config.deny_globs, config.allow_dirs):
            continue

        fragments = extract_python_definitions(
            repo_path, rel_path, diff_tokens, diff_tokens, changed_files,
        )
        for frag in fragments:
            # Enforce per-fragment line limit
            lines_count = frag.code_excerpt.count("\n") + 1
            if lines_count > config.max_fragment_lines:
                frag.code_excerpt = "\n".join(
                    frag.code_excerpt.splitlines()[: config.max_fragment_lines]
                ) + "\n    # ... (truncated)"

            all_fragments.append(frag)
            seen_definitions.add((rel_path, frag.token_match))

    logger.info("AST pass: found %d definitions", len(all_fragments))

    # seen_blocks tracks purely inclusive ranges to avoid subsets
    seen_blocks: set[tuple[str, int, int]] = set()

    # Register all Pass 1 fragments in seen_blocks
    for frag in all_fragments:
        seen_blocks.add((frag.file_path, frag.line_start, frag.line_end))

    def _is_covered(fpath: str, start: int, end: int) -> bool:
        """Check if [start, end] is fully enclosed in any already seen block."""
        for sf, ss, se in list(seen_blocks):  # iterate copy
            if sf == fpath:
                if ss <= start and se >= end:
                    return True
        return False

    def _add_fragment(frag: ContextFragment) -> None:
        """Add fragment if not purely redundant, and register its bounds."""
        if not _is_covered(frag.file_path, frag.line_start, frag.line_end):
            all_fragments.append(frag)
            seen_blocks.add((frag.file_path, frag.line_start, frag.line_end))

    # ------------------------------------------------------------------
    # Pass 2: ripgrep for usages (tokens not already found via AST)
    # ------------------------------------------------------------------
    # Build set of changed file paths for fast lookup
    changed_set = set(changed_files)

    for token in tokens:
        matches = await _run_rg(rf"\b{re.escape(token)}\b", repo_path)

        for match_data in matches:
            file_path_abs = str(match_data.get("path", ""))
            try:
                rel_path = str(Path(file_path_abs).relative_to(repo_path))
            except ValueError:
                rel_path = file_path_abs

            if should_exclude_file(rel_path, config.deny_globs, config.allow_dirs):
                continue

            # Skip files already in the diff — they are visible in the DIFF section
            if rel_path in changed_set:
                continue

            # Skip test/example directories — they don't help understand MR code
            _rel_lower = rel_path.lower()
            if "/tests/" in _rel_lower or "/test/" in _rel_lower or "/example" in _rel_lower:
                continue

            # Skip tiny __init__.py files (just re-exports, not real context)
            rel_p = Path(rel_path)
            if rel_p.name == "__init__.py":
                try:
                    full = repo_path / rel_path
                    line_count = full.read_text(errors="replace").count("\n") + 1
                    if line_count <= 10:
                        continue
                except OSError:
                    pass

            # Skip if we already have an AST definition from this file for this token
            if (rel_path, token) in seen_definitions:
                continue

            match_line = int(str(match_data.get("line_number", 1)))
            file_type = _classify_file(rel_path)

            # Per-fragment budget for block extraction (rough estimate)
            per_frag_budget = int(config.max_context_tokens / 10 / config.token_rate)

            if file_type == "yaml":
                frag = extract_yaml_block(
                    repo_path, rel_path, match_line, per_frag_budget,
                    token, changed_files,
                )
                if frag:
                    _add_fragment(frag)
                continue

            if file_type == "dockerfile":
                frag = extract_dockerfile_block(
                    repo_path, rel_path, match_line, per_frag_budget,
                    token, changed_files,
                )
                if frag:
                    _add_fragment(frag)
                continue

            # Default: use ripgrep excerpt (±9 lines)
            excerpt = str(match_data.get("excerpt", ""))
            if not excerpt.strip():
                continue

            line_start = int(str(match_data.get("line_start", 1)))
            line_end = int(str(match_data.get("line_end", line_start)))

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
                fragment_type="usage",
                priority=50,
            )
            _add_fragment(frag)

    # ------------------------------------------------------------------
    # Sort by priority, deduplicate
    # ------------------------------------------------------------------
    all_fragments.sort(key=lambda f: (f.priority, f.file_path, f.line_start))
    deduped = _deduplicate(all_fragments)

    logger.info(
        "Retrieval: %d fragments (from %d raw), %d tokens searched, "
        "types: %d definitions, %d usages",
        len(deduped), len(all_fragments), len(tokens),
        sum(1 for f in deduped if f.fragment_type != "usage"),
        sum(1 for f in deduped if f.fragment_type == "usage"),
    )
    return deduped

"""AST-based code extraction for intelligent context retrieval.

Supports:
- Python: full function/class bodies via ast.parse()
- YAML/docker-compose: indent-based block extraction
- Dockerfile: instruction-based block extraction
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from pathlib import Path

from mr_lead_agent.models import ContextFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------

def _get_source_lines(file_path: Path) -> list[str] | None:
    """Read file and return lines, or None on error."""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        logger.debug("Cannot read %s", file_path)
        return None


def _extract_node_source(lines: list[str], node: ast.AST) -> str:
    """Extract source lines for an AST node (1-indexed start/end)."""
    start = node.lineno - 1  # type: ignore[attr-defined]
    end = node.end_lineno  # type: ignore[attr-defined]
    if end is None:
        end = start + 1
    return "\n".join(lines[start:end])


def _is_pydantic_model(node: ast.ClassDef) -> bool:
    """Check if a class inherits from BaseModel (simple heuristic)."""
    for base in node.bases:
        name = ""
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        if name == "BaseModel":
            return True
    return False


def _trim_large_class(
    lines: list[str],
    class_node: ast.ClassDef,
    diff_tokens: set[str],
    max_lines: int = 150,
) -> str:
    """For classes > max_lines, extract only relevant methods.

    Always includes: class signature, __init__, and methods whose
    names appear in diff_tokens. Omitted methods are replaced with
    a comment listing their names.
    """
    class_start = class_node.lineno - 1
    class_end = class_node.end_lineno or (class_start + 1)
    class_lines = lines[class_start:class_end]

    if len(class_lines) <= max_lines:
        return "\n".join(class_lines)

    # Partition class body into methods
    parts: list[tuple[str, str, bool]] = []  # (name, source, included)
    # Class signature = first line(s) up to first method/attribute
    first_method_line = class_end  # fallback
    for child in ast.iter_child_nodes(class_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first_method_line = min(first_method_line, child.lineno - 1)

    # Class header (signature + class-level code before first method)
    header = "\n".join(lines[class_start:first_method_line])

    included_methods: list[str] = []
    omitted_names: list[str] = []

    for child in ast.iter_child_nodes(class_node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        method_name: str = child.name
        method_source = _extract_node_source(lines, child)

        if method_name == "__init__" or method_name in diff_tokens:
            included_methods.append(method_source)
        else:
            omitted_names.append(method_name)

    result_parts = [header]
    if included_methods:
        result_parts.extend(included_methods)
    if omitted_names:
        indent = "    "
        result_parts.append(
            f"{indent}# ... ({len(omitted_names)} methods omitted: "
            f"{', '.join(omitted_names)})"
        )

    return "\n\n".join(result_parts)


def extract_python_definitions(
    repo_path: Path,
    file_path: str,
    tokens: set[str],
    diff_tokens: set[str],
    changed_files: list[str],
) -> list[ContextFragment]:
    """Extract full function/class definitions matching tokens from a Python file.

    Returns ContextFragment items with type='definition' or 'pydantic_model'.
    """
    full_path = repo_path / file_path
    if not full_path.exists() or not file_path.endswith(".py"):
        return []

    lines = _get_source_lines(full_path)
    if lines is None:
        return []

    try:
        tree = ast.parse("\n".join(lines), filename=file_path)
    except SyntaxError:
        logger.debug("SyntaxError parsing %s, skipping AST extraction", file_path)
        return []

    is_same_module = file_path in changed_files
    fragments: list[ContextFragment] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in tokens:
            if _is_pydantic_model(node):
                frag_type = "pydantic_model"
                priority = 30
            else:
                frag_type = "definition"
                priority = 10 if is_same_module else 20

            class_end = node.end_lineno or node.lineno
            class_lines_count = class_end - node.lineno + 1

            if class_lines_count > 150:
                source = _trim_large_class(lines, node, diff_tokens)
            else:
                source = _extract_node_source(lines, node)

            fragments.append(ContextFragment(
                file_path=file_path,
                line_start=node.lineno,
                line_end=class_end,
                code_excerpt=source,
                token_match=node.name,
                fragment_type=frag_type,
                priority=priority,
            ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in tokens:
                # Skip methods inside classes — handled by class extraction
                parent_is_class = False
                for parent_node in ast.walk(tree):
                    if isinstance(parent_node, ast.ClassDef):
                        for child in ast.iter_child_nodes(parent_node):
                            if child is node:
                                parent_is_class = True
                                break
                if parent_is_class:
                    continue

                func_end = node.end_lineno or node.lineno
                source = _extract_node_source(lines, node)
                priority = 10 if is_same_module else 20

                fragments.append(ContextFragment(
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=func_end,
                    code_excerpt=source,
                    token_match=node.name,
                    fragment_type="definition",
                    priority=priority,
                ))

    return fragments


# ---------------------------------------------------------------------------
# YAML / docker-compose block extraction
# ---------------------------------------------------------------------------

def _find_yaml_block(lines: list[str], target_line: int) -> tuple[int, int]:
    """Find the enclosing YAML block by indentation.

    Returns (start, end) line indices (0-indexed, exclusive end).
    For top-level keys under 'services:', finds the full service block.
    """
    if target_line >= len(lines) or target_line < 0:
        return (max(0, target_line - 15), min(len(lines), target_line + 15))

    # Find the indentation of the target line
    target_indent = len(lines[target_line]) - len(lines[target_line].lstrip())

    # Walk up to find the block start (same or lower indent, non-empty)
    block_start = target_line
    for i in range(target_line - 1, -1, -1):
        line = lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent < target_indent:
            block_start = i
            target_indent = indent
            break
        block_start = i

    # Walk down to find block end
    block_end = target_line + 1
    for i in range(target_line + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            block_end = i + 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= target_indent and line.strip():
            break
        block_end = i + 1

    return (block_start, block_end)


def extract_yaml_block(
    repo_path: Path,
    file_path: str,
    match_line: int,
    per_frag_budget_chars: int,
    token_match: str,
    changed_files: list[str],
) -> ContextFragment | None:
    """Extract the enclosing YAML block around a matched line."""
    full_path = repo_path / file_path
    lines = _get_source_lines(full_path)
    if lines is None:
        return None

    start, end = _find_yaml_block(lines, match_line - 1)  # convert to 0-indexed
    block_lines = lines[start:end]
    block_text = "\n".join(block_lines)

    # If block too large, fall back to ±15 lines
    if len(block_text) > per_frag_budget_chars:
        ctx = 15
        s = max(0, (match_line - 1) - ctx)
        e = min(len(lines), (match_line - 1) + ctx + 1)
        block_text = "\n".join(lines[s:e])
        start, end = s, e

    is_same_module = file_path in changed_files
    return ContextFragment(
        file_path=file_path,
        line_start=start + 1,
        line_end=end,
        code_excerpt=block_text,
        token_match=token_match,
        fragment_type="definition",
        priority=10 if is_same_module else 20,
    )


# ---------------------------------------------------------------------------
# Dockerfile block extraction
# ---------------------------------------------------------------------------

_DOCKERFILE_INSTRUCTIONS = re.compile(
    r"^(FROM|RUN|COPY|ADD|CMD|ENTRYPOINT|ENV|EXPOSE|VOLUME|"
    r"WORKDIR|USER|ARG|ONBUILD|LABEL|STOPSIGNAL|HEALTHCHECK|SHELL)\b",
    re.IGNORECASE,
)


def _find_dockerfile_block(lines: list[str], target_line: int) -> tuple[int, int]:
    """Find the Dockerfile instruction block containing target_line (0-indexed)."""
    if target_line >= len(lines):
        target_line = len(lines) - 1

    # Walk up to find instruction start
    block_start = target_line
    for i in range(target_line, -1, -1):
        if _DOCKERFILE_INSTRUCTIONS.match(lines[i].strip()):
            block_start = i
            break

    # Walk down to find instruction end (next instruction or EOF)
    block_end = len(lines)
    for i in range(target_line + 1, len(lines)):
        stripped = lines[i].strip()
        if _DOCKERFILE_INSTRUCTIONS.match(stripped):
            block_end = i
            break

    return (block_start, block_end)


def extract_dockerfile_block(
    repo_path: Path,
    file_path: str,
    match_line: int,
    per_frag_budget_chars: int,
    token_match: str,
    changed_files: list[str],
) -> ContextFragment | None:
    """Extract the enclosing Dockerfile instruction block."""
    full_path = repo_path / file_path
    lines = _get_source_lines(full_path)
    if lines is None:
        return None

    start, end = _find_dockerfile_block(lines, match_line - 1)
    block_lines = lines[start:end]
    block_text = "\n".join(block_lines)

    if len(block_text) > per_frag_budget_chars:
        ctx = 15
        s = max(0, (match_line - 1) - ctx)
        e = min(len(lines), (match_line - 1) + ctx + 1)
        block_text = "\n".join(lines[s:e])
        start, end = s, e

    is_same_module = file_path in changed_files
    return ContextFragment(
        file_path=file_path,
        line_start=start + 1,
        line_end=end,
        code_excerpt=block_text,
        token_match=token_match,
        fragment_type="definition",
        priority=10 if is_same_module else 20,
    )

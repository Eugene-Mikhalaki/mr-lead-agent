"""Rich-powered report renderer for MR review results."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mr_lead_agent.models import MRData, PipelineStats, ReviewResult

logger = logging.getLogger(__name__)
console = Console()


def _severity_color(severity: str) -> str:
    return {
        "blocker": "bold red",
        "major": "yellow",
        "minor": "cyan",
    }.get(severity.lower(), "white")


def render_report(
    mr_data: MRData,
    result: ReviewResult,
    stats: PipelineStats,
    save_runs: bool = True,
    runs_dir: str = "./runs",
) -> None:
    """Print the review report to stdout using Rich formatting."""

    # --- Header ---
    console.print()
    console.rule(f"[bold blue]MR Review: {mr_data.title}", style="blue")
    console.print(f"[dim]URL:[/dim]    {mr_data.web_url}")
    console.print(f"[dim]Author:[/dim] {mr_data.author}")
    console.print(f"[dim]SHA:[/dim]    {mr_data.sha[:12] if mr_data.sha else 'n/a'}")

    # --- Context stats ---
    stat_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    stat_table.add_column("key", style="dim", width=24)
    stat_table.add_column("value")
    stat_table.add_row("Diff lines", str(stats.diff_lines))
    stat_table.add_row("Context fragments", str(stats.context_fragments))
    stat_table.add_row("Context files", str(stats.context_files))
    stat_table.add_row("Secrets redacted", str(stats.redaction.secrets_replaced))
    stat_table.add_row("URLs redacted", str(stats.redaction.urls_replaced))
    stat_table.add_row("Files excluded", str(stats.redaction.files_excluded))
    if stats.prompt_tokens:
        stat_table.add_row("Prompt tokens", f"{stats.prompt_tokens:,}")
        stat_table.add_row("Completion tokens", f"{stats.completion_tokens:,}")
        total = stats.prompt_tokens + stats.completion_tokens
        stat_table.add_row("[dim]Total tokens[/dim]", f"[dim]{total:,}[/dim]")
    if stats.summary_only_mode:
        stat_table.add_row("[yellow]Mode[/yellow]", "[yellow]summary-only (large diff)[/yellow]")
    console.print(Panel(stat_table, title="[dim]Pipeline Stats", border_style="dim"))

    # --- 1. Summary ---
    console.rule("[bold green]Summary", style="green")
    for point in result.summary:
        console.print(f"  • {point}")

    # --- 2. Key Risks ---
    if result.key_risks:
        console.rule("[bold yellow]Key Risks", style="yellow")
        for risk in result.key_risks:
            color = _severity_color(risk.severity)
            console.print(f"  [{color}][{risk.severity.upper()}][/{color}] {risk.title}")
            console.print(f"         {risk.details}", style="dim")
    else:
        console.rule("[dim]Key Risks", style="dim")
        console.print("  [dim](none identified)[/dim]")

    # --- 3. Blockers ---
    if result.blockers:
        console.rule("[bold red]Blockers", style="red")
        for i, bl in enumerate(result.blockers, 1):
            console.print(
                f"\n  [bold red][{i}] {bl.title}[/bold red]"
                f"  [dim]{bl.file}:{bl.lines}[/dim]"
            )
            console.print(f"      {bl.comment}")
            if bl.suggested_fix:
                console.print(f"      [cyan]Fix:[/cyan] {bl.suggested_fix}")
            if bl.verification:
                console.print(f"      [dim]Verify:[/dim] {bl.verification}")
    else:
        console.rule("[dim]Blockers", style="dim")
        console.print("  [green](no blockers)[/green]")

    # --- 4. Questions to Author ---
    if result.questions_to_author:
        console.rule("[bold magenta]Questions to Author", style="magenta")
        for i, q in enumerate(result.questions_to_author, 1):
            loc = f"[dim]{q.file}:{q.lines}[/dim]  " if q.file else ""
            console.print(f"\n  [bold magenta][{i}][/bold magenta] {loc}{q.question}")
            console.print(f"      [dim]Why: {q.why_it_matters}[/dim]")
    else:
        console.rule("[dim]Questions to Author", style="dim")
        console.print("  [dim](none)[/dim]")

    console.rule(style="dim")

    # --- Save JSON ---
    if save_runs:
        _save_json(mr_data, result, stats, runs_dir)


def _save_json(
    mr_data: MRData,
    result: ReviewResult,
    stats: PipelineStats,
    runs_dir: str,
) -> None:
    """Save the full result as a JSON file in runs_dir."""
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    sha_short = mr_data.sha[:12] if mr_data.sha else "unknown"
    filename = path / f"mr{mr_data.iid}_{sha_short}.json"
    payload = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "mr": mr_data.model_dump(),
        "stats": stats.model_dump(),
        "result": result.model_dump(),
    }
    filename.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Result saved to %s", filename)
    console.print(f"[dim]Result saved → {filename}[/dim]")


def render_dry_run(
    mr_data: MRData,
    prompt: str,
    stats: PipelineStats,
) -> None:
    """Print dry-run output: prompt and stats, without calling the LLM."""
    console.print()
    console.rule("[bold yellow]DRY RUN — Prompt Preview", style="yellow")
    console.print(f"[dim]MR:[/dim] {mr_data.web_url}")
    console.print(f"[dim]Prompt size:[/dim] {len(prompt):,} chars")
    console.print(f"[dim]Diff lines:[/dim] {stats.diff_lines:,}")
    console.print(f"[dim]Context fragments:[/dim] {stats.context_fragments}")
    console.rule("[dim]Prompt (first 3000 chars)", style="dim")
    console.print(prompt[:3000])
    if len(prompt) > 3000:
        console.print(f"\n[dim]... ({len(prompt) - 3000:,} more chars)[/dim]")

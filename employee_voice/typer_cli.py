"""
Typer + Rich CLI for the Employee Voice Analytics toolkit.

Installs the `eva` console script alongside the legacy `employee-voice`
shim. Subcommands:

  eva analyze       — run the pipeline on a CSV / XLSX / Delta file
  eva generate-sample — produce a synthetic HR feedback dataset
  eva dashboard     — Rich terminal dashboard for a finished fact table
  eva redact        — PII-scrub a CSV without running the rest of the pipeline
  eva version       — print the package version

The CLI is intentionally thin: each subcommand delegates to the
in-package pipeline / data generators. The argparse-based
`employee_voice.cli:main` stays in place for backward compatibility.
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Employee Voice Analytics — sentiment, category, emotion, themes, attrition risk.",
)
console = Console()
err_console = Console(stderr=True)

log = logging.getLogger("eva")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"employee-voice-analytics [bold cyan]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    pass


# ---------- analyze ----------------------------------------------------

@app.command()
def analyze(
    input_path: Path = typer.Option(..., "--input", "-i", help="CSV / XLSX / JSON / Parquet / Delta path."),
    outdir: Path = typer.Option("output", "--outdir", "-o", help="Output directory."),
    text_column: Optional[str] = typer.Option(None, "--text-column", help="Override the comment column."),
    no_topics: bool = typer.Option(False, "--no-topics", help="Skip topic discovery."),
    redact_pii: bool = typer.Option(False, "--redact-pii", help="PII-scrub before analysis."),
    pii_backend: str = typer.Option("regex", "--pii-backend", help="regex or presidio."),
    k_anonymity: int = typer.Option(5, "--k-anonymity", help="Min responses per slice before quoting."),
    summary: bool = typer.Option(False, "--summary", help="Generate LLM exec summary (Azure OpenAI)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full pipeline and write the four output artifacts."""
    from .pipeline import run
    from .privacy import PIIScrubber, apply_k_anonymity, KAnonymityConfig

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not input_path.exists():
        err_console.print(f"[red]Input not found:[/red] {input_path}")
        raise typer.Exit(code=2)

    with Progress(SpinnerColumn(), TextColumn("[bold]Running layers 0-5..."), console=console, transient=True):
        artifacts = run(
            input_path=str(input_path),
            outdir=str(outdir),
            text_column=text_column,
            run_topics=not no_topics,
        )
        fact = artifacts["fact"]

    if redact_pii:
        scrubber = PIIScrubber(backend=pii_backend)
        fact["CleanComment"] = scrubber.scrub_series(fact["Comment"])

    if k_anonymity and k_anonymity > 1:
        group_cols = [c for c in ("BU", "Dept") if c in fact.columns]
        if group_cols:
            fact, _slice_summary = apply_k_anonymity(
                fact, group_cols,
                KAnonymityConfig(threshold=k_anonymity),
            )
            out_path = Path(outdir) / "fact_employee_feedback.csv"
            fact.to_csv(out_path, index=False)

    if summary:
        from .summarizer import generate_summary
        out = generate_summary(fact, str(outdir))
        if out:
            console.print(f"  [dim]Exec summary:[/dim] {out}")

    console.print(f"[bold green]Done.[/bold green] Wrote 4 artifacts to [cyan]{outdir}[/]")
    risk_counts = fact["RiskBand"].value_counts() if "RiskBand" in fact.columns else {}
    table = Table(title="Risk band distribution", show_header=True, header_style="bold")
    table.add_column("Band", style="cyan")
    table.add_column("Count", justify="right")
    for band in ("High", "Medium", "Low"):
        table.add_row(band, str(int(risk_counts.get(band, 0))))
    console.print(table)


# ---------- generate-sample --------------------------------------------

@app.command()
def generate_sample(
    rows: int = typer.Option(250, "--rows", "-n", help="Number of rows to generate."),
    out: Path = typer.Option("data/sample_feedback_synth.csv", "--out", "-o", help="Output CSV path."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Generate a fully synthetic HR feedback dataset across 4 personas."""
    from .synth_data import generate_synthetic_dataset

    with Progress(SpinnerColumn(), TextColumn(f"[bold]Generating {rows} rows..."), console=console, transient=True):
        df = generate_synthetic_dataset(n=rows, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[bold green]Wrote[/bold green] {len(df)} rows to [cyan]{out}[/]")


# ---------- dashboard --------------------------------------------------

@app.command()
def dashboard(
    fact_path: Path = typer.Option(..., "--fact", "-f", help="Path to fact_employee_feedback.csv."),
    by: str = typer.Option("BU", "--by", help="Group-by column for the risk table."),
    top_n: int = typer.Option(10, "--top", help="Top N rows in the risk table."),
) -> None:
    """Render a Rich terminal dashboard from a finished fact table."""
    import pandas as pd
    if not fact_path.exists():
        err_console.print(f"[red]Fact file not found:[/red] {fact_path}")
        raise typer.Exit(code=2)
    df = pd.read_csv(fact_path)
    _render_dashboard(df, by=by, top_n=top_n)


def _render_dashboard(df, by: str, top_n: int) -> None:
    """The actual Rich rendering. Separated for testability."""
    console.rule("[bold]Employee Voice — Dashboard")

    # 1. Sentiment distribution.
    sent = df["Sentiment"].value_counts() if "Sentiment" in df.columns else {}
    t = Table(title="Sentiment distribution", show_header=True, header_style="bold")
    t.add_column("Label", style="cyan")
    t.add_column("Count", justify="right")
    for label in ("positive", "neutral", "negative", "No Comment"):
        t.add_row(label, str(int(sent.get(label, 0))))
    console.print(t)

    # 2. Risk band distribution.
    risk = df["RiskBand"].value_counts() if "RiskBand" in df.columns else {}
    t = Table(title="Risk band distribution", show_header=True, header_style="bold")
    t.add_column("Band", style="cyan")
    t.add_column("Count", justify="right")
    for band in ("High", "Medium", "Low"):
        t.add_row(band, str(int(risk.get(band, 0))))
    console.print(t)

    # 3. Top themes.
    if "TopicName" in df.columns:
        topics = df["TopicName"].value_counts().head(top_n)
        t = Table(title=f"Top {top_n} themes", show_header=True, header_style="bold")
        t.add_column("Topic", style="cyan")
        t.add_column("Docs", justify="right")
        for name, n in topics.items():
            t.add_row(str(name)[:60], str(int(n)))
        console.print(t)

    # 4. By-column breakdown.
    if by in df.columns and "RiskScore" in df.columns:
        grouped = (
            df.groupby(by)
            .agg(
                FeedbackCount=("FeedbackID", "count"),
                AvgRisk=("RiskScore", "mean"),
                HighRisk=("RiskBand", lambda s: int((s == "High").sum())),
            )
            .sort_values("AvgRisk", ascending=False)
            .head(top_n)
            .reset_index()
        )
        t = Table(title=f"By {by} (top {top_n} by AvgRisk)", show_header=True, header_style="bold")
        t.add_column(by, style="cyan")
        t.add_column("FeedbackCount", justify="right")
        t.add_column("AvgRisk", justify="right")
        t.add_column("HighRisk", justify="right")
        for _, row in grouped.iterrows():
            t.add_row(
                str(row[by]),
                str(int(row["FeedbackCount"])),
                f"{float(row['AvgRisk']):.1f}",
                str(int(row["HighRisk"])),
            )
        console.print(t)


# ---------- redact ----------------------------------------------------

@app.command()
def redact(
    input_path: Path = typer.Option(..., "--input", "-i", help="CSV / XLSX with a Comment column."),
    out: Path = typer.Option(..., "--out", "-o", help="Output CSV."),
    text_column: str = typer.Option("Comment", "--text-column", help="Column to redact."),
    backend: str = typer.Option("regex", "--backend", help="regex or presidio."),
) -> None:
    """PII-scrub a CSV without running the rest of the pipeline."""
    import pandas as pd
    from .privacy import PIIScrubber

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        err_console.print(f"[red]Column not found:[/red] {text_column}")
        raise typer.Exit(code=2)
    scrubber = PIIScrubber(backend=backend)
    with Progress(SpinnerColumn(), TextColumn("[bold]Scrubbing PII..."), console=console, transient=True):
        df[f"{text_column}_redacted"] = scrubber.scrub_series(df[text_column].fillna("").astype(str))
    df.to_csv(out, index=False)
    console.print(f"[bold green]Wrote[/bold green] redacted CSV to [cyan]{out}[/]")


# ---------- entry point -----------------------------------------------

def cli_main() -> int:
    """Console-script entry point. Returns 0 on success, non-zero on error."""
    try:
        app()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(cli_main())

"""
main.py
--------
CLI entry point for the Document Layout Analysis pipeline.

Commands
--------
  python main.py run   <pdf_folder>          — run inference
  python main.py train <annotations.csv>     — train LightGBM model
  python main.py demo  <single.pdf>          — process one PDF, show stats
  python main.py stats <pdf_folder>          — count PDFs + pages
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app     = Console()
cli     = typer.Typer(
    name="doc-layout",
    help="High-throughput document layout analysis pipeline.",
    add_completion=False,
)


# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "data/results/pipeline.log",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )


# ─── Commands ─────────────────────────────────────────────────────────────────

@cli.command("run")
def cmd_run(
    pdf_folder: str = typer.Argument(..., help="Folder with PDFs to process"),
    output_dir: str = typer.Option("data/results", help="Output directory"),
    max_pdfs:   int = typer.Option(None, help="Limit number of PDFs"),
    recursive:  bool = typer.Option(True, help="Recurse into sub-folders"),
    verbose:    bool = typer.Option(False, "--verbose", "-v"),
):
    """Run layout analysis on a folder of PDFs."""
    _setup_logging(verbose)
    from pipeline import LayoutPipeline

    pipe  = LayoutPipeline.from_saved(output_dir=output_dir)
    stats = pipe.run(pdf_folder, recursive=recursive, max_pdfs=max_pdfs)
    _print_stats(stats)


@cli.command("train")
def cmd_train(
    annotations_csv: str = typer.Argument(..., help="CSV with labelled boxes"),
    pdf_root:        str = typer.Option(".", help="Root folder for PDF paths in CSV"),
    max_samples:     int = typer.Option(None, help="Cap training samples"),
    output_dir:      str = typer.Option("data/results", help="Output directory"),
    verbose:         bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Train the LightGBM classifier from a labelled CSV.

    CSV must have columns: pdf_path, page, x, y, w, h, label
    """
    _setup_logging(verbose)
    from pipeline import LayoutPipeline

    pipe = LayoutPipeline(output_dir=output_dir)
    pipe.train_from_annotations(annotations_csv, pdf_root, max_samples)
    logger.success("Training complete!")


@cli.command("demo")
def cmd_demo(
    pdf_path:   str = typer.Argument(..., help="Single PDF to analyse"),
    output_dir: str = typer.Option("data/results", help="Output directory"),
    verbose:    bool = typer.Option(False, "--verbose", "-v"),
):
    """Process a single PDF and show per-page region counts."""
    _setup_logging(verbose)
    from pipeline import LayoutPipeline

    pipe  = LayoutPipeline.from_saved(output_dir=output_dir)
    stats = pipe.run(Path(pdf_path).parent, max_pdfs=1)
    _print_stats(stats)


@cli.command("stats")
def cmd_stats(
    pdf_folder: str = typer.Argument(..., help="Folder to scan"),
    recursive:  bool = typer.Option(True),
):
    """Count PDFs and pages in a folder without processing them."""
    from ingestion.pdf_loader import PDFLoader

    folder = Path(pdf_folder)
    loader = PDFLoader()
    result = loader.count_folder(folder, recursive)

    tbl = Table(title=f"Corpus stats — {folder}")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value",  style="green", justify="right")
    tbl.add_row("PDFs found",    f"{result['pdfs']:,}")
    tbl.add_row("Total pages",   f"{result['pages']:,}")
    tbl.add_row("Unreadable",    f"{result['errors']:,}")
    app.print(tbl)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _print_stats(stats: dict) -> None:
    tbl = Table(title="Pipeline Results")
    tbl.add_column("Metric",  style="cyan")
    tbl.add_column("Value",   style="green", justify="right")

    rows = [
        ("PDFs processed",        f"{stats.get('pdfs', 0):,}"),
        ("Pages processed",       f"{stats.get('pages', 0):,}"),
        ("Regions extracted",     f"{stats.get('regions', 0):,}"),
        ("Proposal reduction",    f"{stats.get('proposal_reduction_pct', 0):.1f}%"),
        ("Throughput (pages/s)",  f"{stats.get('pages_per_second', 0):.1f}"),
        ("Elapsed (s)",           f"{stats.get('elapsed_s', 0):.1f}"),
        ("Errors",                f"{stats.get('errors', 0)}"),
    ]
    for k, v in rows:
        tbl.add_row(k, v)

    app.print(tbl)


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()

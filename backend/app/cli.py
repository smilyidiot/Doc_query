"""Command-line interface for DocQuery.

    python -m app.cli ask "What is retrieval-augmented generation?"
    python -m app.cli ask "How does FastAPI validate input?" --k 2
    python -m app.cli sources
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import qa
from .index import VectorIndex, load_documents

cli = typer.Typer(add_completion=False, help="DocQuery — ask questions over your documents.")
_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def _index(docs_dir: Path) -> VectorIndex:
    documents = load_documents(docs_dir)
    if not documents:
        typer.secho(f"No .md/.txt documents found in {docs_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return VectorIndex().build(documents)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    k: int = typer.Option(3, "--k", help="Number of passages to retrieve."),
    docs: Path = typer.Option(_DOCS, "--docs", help="Folder of .md/.txt documents."),
) -> None:
    """Answer a question from the indexed documents."""
    result = qa.answer(_index(docs), question, k=k)
    typer.secho(f"\nQ: {result['question']}", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"\nA: {result['answer']}\n", fg=typer.colors.WHITE)
    typer.secho(f"Confidence: {result['confidence']:.2f}", dim=True)
    typer.secho("Sources:", bold=True)
    for s in result["sources"]:
        typer.echo(f"  [{s['score']:.2f}] {s['source']}: {s['snippet']}")


@cli.command()
def sources(docs: Path = typer.Option(_DOCS, "--docs")) -> None:
    """List the documents that would be indexed."""
    idx = _index(docs)
    for name in sorted({c.source for c in idx.chunks}):
        typer.echo(f"  {name}")
    typer.secho(f"\n{idx.size} chunks across {len(set(c.source for c in idx.chunks))} documents.", dim=True)


if __name__ == "__main__":
    cli()

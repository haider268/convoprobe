"""Command line interface. Commands are added milestone by milestone."""

from __future__ import annotations

import typer

from convoprobe import __version__

app = typer.Typer(
    help="Stress-test conversational agents with simulated users.", no_args_is_help=True
)


@app.callback()
def _root() -> None:
    """convoprobe command line."""


@app.command()
def version() -> None:
    """Print the installed convoprobe version."""
    typer.echo(__version__)


def main() -> None:
    """Console script entry point."""
    app()

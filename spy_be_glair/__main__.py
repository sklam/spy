import asyncio
import sys
from functools import wraps

import typer

from spy.cli._runners import _run_command
from spy.vendored.dataclass_typer import dataclass_typer
from spy_be_glair.commands import Glair_Args, glair


@wraps(glair)
def _sync_glair(args: Glair_Args) -> None:
    asyncio.run(_run_command(glair, args))


app = typer.Typer(
    pretty_exceptions_enable=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.command()(dataclass_typer(_sync_glair))


if __name__ == "__main__":
    app()

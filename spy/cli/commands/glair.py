from dataclasses import dataclass
from pathlib import Path
from typing import (
    Annotated,
    Optional,
)

from typer import Option

from spy.analyze.importing import ImportAnalyzer
from spy.backend.glair.glairbackend import GlairBackend
from spy.cli._runners import init_vm
from spy.cli.commands.build import get_build_dir
from spy.cli.commands.shared_args import (
    Base_Args,
    Filename_Required_Args,
)


@dataclass
class _glair_mixin:
    irdump: Annotated[
        bool,
        Option("--irdump", help="Dump the generated GLAIR IR to stdout"),
    ] = False

    build_dir: Annotated[
        Optional[Path],
        Option(
            "-b",
            "--build-dir",
            help="Directory to store generated files (defaults to build/ next to the "
            ".spy file)",
            show_default=False,
        ),
    ] = None


@dataclass
class Glair_Args(Base_Args, _glair_mixin, Filename_Required_Args): ...


async def glair(args: Glair_Args) -> None:
    """
    Generate GLAIR IR and optionally dump it to stdout
    """
    modname = args.filename.stem
    vm = await init_vm(args)

    importer = ImportAnalyzer(vm, modname, use_spyc=not args.no_spyc)
    importer.parse_all()
    importer.import_all()

    vm.ast_color_map = {}
    vm.redshift(error_mode=args.error_mode)

    build_dir = get_build_dir(args)  # type: ignore[arg-type]

    glair_backend = GlairBackend(vm, modname, build_dir, dump_glair=args.irdump)
    glair_backend.glairwrite()

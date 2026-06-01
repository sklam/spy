import py.path

from spy.highlight import highlight_src
from spy.vm.object import W_Type
from spy.vm.vm import SPyVM
from spy_be_glair.glairmodwriter import GlairModule, GlairModuleWriter
from spy_be_glair.glairstructwriter import GlairStructDefs, GlairStructWriter


class GlairBackend:
    """
    Convert SPy modules into GLAIR files.
    """

    vm: SPyVM
    main_modname: str
    build_dir: py.path.local
    dump_glair: bool
    glair_structdefs: dict[str, GlairStructDefs]
    glair_modules: dict[str, GlairModule]
    glairfiles: list[py.path.local]

    def __init__(
        self,
        vm: SPyVM,
        main_modname: str,
        build_dir: py.path.local,
        *,
        dump_glair: bool,
    ) -> None:
        self.vm = vm
        self.main_modname = main_modname
        self.build_dir = build_dir
        self.build_dir.join("src").ensure(dir=True)
        self.dump_glair = dump_glair
        self.glair_structdefs = {}
        self.glair_modules = {}
        self.glairfiles = []

    def split_fqns(self) -> None:
        """
        Split the global FQNs into GlairModule and GlairStructDefs objects,
        mirroring the same logic as CBackend.split_fqns().
        """
        bdir = self.build_dir
        # Create a GlairModule for each non-builtin SPy module
        for modname, w_mod in self.vm.modules_w.items():
            if w_mod.is_builtin():
                continue
            assert w_mod.filepath is not None
            spyfile = py.path.local(w_mod.filepath)
            basename = spyfile.purebasename
            glairfile = bdir.join("src", f"{basename}.glair")
            glair_mod = GlairModule(
                modname=modname,
                spyfile=spyfile,
                glairfile=glairfile,
                content=[],
            )
            self.glair_modules[modname] = glair_mod

        # Single GlairStructDefs for all type declarations
        self.glair_structdefs["globals"] = GlairStructDefs(
            glairfile=bdir.join("src", "spy_structdefs.glair"),
            content=[],
        )

        # Assign each FQN to the appropriate GlairModule or GlairStructDefs
        for fqn, w_obj in self.vm.globals_w.items():
            if fqn.is_module():
                continue
            modname = fqn.modname
            w_mod = self.vm.modules_w[modname]
            if isinstance(w_obj, W_Type):
                irtag = self.vm.get_irtag(fqn)
                if w_mod.filepath is None and irtag.tag != "mlir.type":
                    continue
                self.glair_structdefs["globals"].content.append((fqn, w_obj))
            elif w_mod.filepath is not None:
                self.glair_modules[modname].content.append((fqn, w_obj))

    def _write_prelude(self) -> None:
        preludefile = self.build_dir.join("src", "_prelude.glair")
        preludefile.write("module _prelude;\n\n@builtin\nextern fn abort() -> void;\n")
        self.glairfiles.append(preludefile)
        if self.dump_glair:
            print()
            print(f"---- {preludefile} ----")
            print(preludefile.read())

    def glairwrite(self) -> None:
        """
        Convert all non-builtin modules into .glair files.
        """
        self.vm.linearize_all()
        self.split_fqns()
        self._write_prelude()

        # Emit spy_structdefs.glair
        for glair_structdefs in self.glair_structdefs.values():
            writer = GlairStructWriter(self.vm, glair_structdefs)
            writer.write_glair_source()
            self.glairfiles.append(glair_structdefs.glairfile)
            if self.dump_glair:
                print()
                print(f"---- {glair_structdefs.glairfile} ----")
                # Use C highlighter as a reasonable approximation for GLAIR
                print(highlight_src("C", glair_structdefs.glairfile.read()))

        # Emit <module>.glair files
        for glair_mod in self.glair_modules.values():
            writer = GlairModuleWriter(self.vm, glair_mod)
            writer.write_glair_source()
            self.glairfiles.append(glair_mod.glairfile)
            if self.dump_glair:
                print()
                print(f"---- {glair_mod.glairfile} ----")
                print(highlight_src("C", glair_mod.glairfile.read()))

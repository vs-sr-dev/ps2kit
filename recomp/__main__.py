"""Recompile an EE executable to C++.

    python -m ps2kit.recomp GAME.ELF --out build/recomp [--functions map.tsv]
                                    [--per-file 8000] [--no-comments]

Writes, into --out:
    funcs.h          prototypes of every recompiled entry point
    recomp_NNN.cpp   the functions, split by instruction count
    table.cpp        the sorted address -> function table for indirect calls
    CMakeLists.txt   the library, the link check, the runtime
    report.txt       units, entries, switch tables, anything not handled

The function map comes from the executable's own symbols
(ps2kit.elf.functions(): .symtab and .mdebug) or from a TSV (va, size, name,
source). The generated code needs ps2kit/runtime/ee.h.
"""
import argparse
import os
import time

from .. import ee
from ..elf import Elf
from . import emit as E
from .program import Program

RUNTIME = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "runtime"))


class Out:
    """A generated file, written only if its content changed, so that a
    rebuild after a small change recompiles only what it touched."""

    def __init__(self, path):
        self.path, self.parts = path, []

    def write(self, s):
        self.parts.append(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        text = "".join(self.parts)
        try:
            with open(self.path, encoding="utf-8", newline="") as f:
                if f.read() == text:
                    return
        except OSError:
            pass
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(text)


class FnCtx:
    """Control-flow answers for one entry's body."""

    def __init__(self, prog, entry, entries):
        self.prog, self.entry, self.entries = prog, entry, entries
        self.unit, _ = prog.body(entry)
        self.tables = prog.tables

    def jump(self, t, src):
        if self.entry <= t < self.unit.end:
            if t > src:
                return "goto L_%08X;" % t
            return "EE_POLL(c); goto L_%08X;" % t          # a loop's back-edge: a safe point
        if t in self.entries:
            return "{ %s(c); return; }" % E.fname(t)
        return '{ ee_unimplemented(c, 0x%08Xu, "branch to unknown target"); return; }' % t

    def call(self, t):
        if t in self.entries:
            return "%s(c);" % E.fname(t)
        return 'ee_unimplemented(c, 0x%08Xu, "call to unknown target");' % t


def _function(prog, e, entries, comments, unsupported):
    unit, body = prog.body(e)
    fn = FnCtx(prog, e, entries)
    labels = prog.labels(e)
    for i in body:                      # a jump into a delay slot: the slot is emitted twice
        if i.has_delay_slot and i.va + 4 in labels:
            labels.add(i.va + 8)
    title = unit.name if e == unit.start else "%s+0x%X" % (unit.name, e - unit.start)
    src = " (%s)" % unit.source if unit.source and e == unit.start else ""
    lines = ["// %s%s" % (title, src), "void %s(EEContext& c) {" % E.fname(e)]

    def code(i):
        try:
            return E.emit(i)
        except E.Unsupported as ex:
            unsupported[str(ex)] = unsupported.get(str(ex), 0) + 1
            return ['ee_unimplemented(c, 0x%08Xu, "%s");' % (i.va, ex)]

    def note(i):
        return "  // %08X %s" % (i.va, ee.disasm(i)) if comments else ""

    k, terminated = 0, False
    while k < len(body):
        i = body[k]
        if i.va in labels:
            lines.append("L_%08X:" % i.va)
        if i.has_delay_slot:
            slot = body[k + 1] if k + 1 < len(body) else None
            slot_code = code(slot) if slot is not None else []
            try:
                cf = E.control_flow(i, slot_code, fn)
            except E.Unsupported as ex:
                unsupported[str(ex)] = unsupported.get(str(ex), 0) + 1
                cf = ['ee_unimplemented(c, 0x%08Xu, "%s"); return;' % (i.va, ex)]
            lines.append("    " + cf[0] + note(i) + (note(slot) if slot is not None else ""))
            lines.extend("    " + x for x in cf[1:])
            terminated = E.terminates(i)
            if slot is not None and slot.va in labels:
                if not terminated:
                    lines.append("    goto L_%08X;" % (slot.va + 4))
                lines.append("L_%08X:" % slot.va)
                lines.extend("    " + x for x in slot_code)
                terminated = False
            k += 2
            continue
        c = code(i)
        if c:
            lines.append("    " + c[0] + note(i))
            lines.extend("    " + x for x in c[1:])
        elif comments:
            lines.append("    ;" + note(i))
        terminated = False
        k += 1
    if unit.end in labels:
        lines.append("L_%08X:" % unit.end)
        terminated = False
    if not terminated:
        nxt = unit.end
        if nxt in entries:
            lines.append("    %s(c); return;  // falls into the next function" % E.fname(nxt))
        else:
            lines.append('    ee_unimplemented(c, 0x%08Xu, "fell off the end"); return;' % nxt)
    lines.append("}\n")
    return lines, len(body)


def generate(elf, functions, out, per_file=8000, comments=True, log=print):
    t0 = time.time()
    prog = Program(elf, functions, log=log)
    entries = set(prog.entries)
    os.makedirs(out, exist_ok=True)
    unsupported, files, n_ins = {}, [], 0
    buf, count, idx = [], 0, 0

    def flush():
        nonlocal buf, count, idx
        if not buf:
            return
        name = "recomp_%03d.cpp" % idx
        with Out(os.path.join(out, name)) as f:
            f.write('#include "funcs.h"\n\n')
            f.write("\n".join(buf))
        files.append(name)
        buf, count, idx = [], 0, idx + 1

    for e in prog.entries:
        lines, n = _function(prog, e, entries, comments, unsupported)
        buf.append("\n".join(lines))
        count += n
        n_ins += n
        if count >= per_file:
            flush()
    flush()

    with Out(os.path.join(out, "funcs.h")) as f:
        f.write('#pragma once\n#include "ee.h"\n\n')
        for e in prog.entries:
            f.write("void %s(EEContext& c);\n" % E.fname(e))
    with Out(os.path.join(out, "table.cpp")) as f:
        f.write('#include "funcs.h"\n#include <cstddef>\n\n')
        f.write("extern const EEFuncEntry g_ee_funcs[] = {\n")
        for e in prog.entries:
            u = prog.unit_at(e)
            name = u.name if e == u.start else "%s+0x%X" % (u.name, e - u.start)
            f.write('    {0x%08Xu, %s, "%s"},\n' % (e, E.fname(e), name.replace("\\", "\\\\")))
        f.write("};\n")
        f.write("extern const size_t g_ee_nfuncs = %d;\n" % len(prog.entries))
    with Out(os.path.join(out, "CMakeLists.txt")) as f:
        rt = RUNTIME.replace("\\", "/")
        srcs = "\n    ".join(files + ["table.cpp"])
        f.write("""cmake_minimum_required(VERSION 3.20)
project(ps2kit_recomp CXX)
set(CMAKE_CXX_STANDARD 20)
set(PS2KIT_RUNTIME "%s")
add_library(recomp STATIC
    %s)
target_include_directories(recomp PUBLIC ${PS2KIT_RUNTIME})
target_compile_options(recomp PRIVATE -Wno-unused-label -Wno-unused-variable -fno-strict-aliasing)
include(${PS2KIT_RUNTIME}/runtime.cmake)
add_executable(linkcheck ${PS2KIT_RUNTIME}/linkcheck.cpp)
target_link_libraries(linkcheck ps2kit_core ps2kit_stub recomp)
if(DEFINED PS2KIT_EXTRA)
  include(${PS2KIT_EXTRA})
endif()
""" % (rt, srcs))
    report = [
        "units            %d (%d gaps)" % (len(prog.units), sum(1 for u in prog.units if u.name.startswith("unk_"))),
        "entries          %d (%d inside units)" % (len(prog.entries),
                                                    sum(1 for e in prog.entries if prog.unit_at(e).start != e)),
        "instructions     %d (%d undecoded)" % (n_ins, len(prog.undecoded)),
        "switch tables    %d resolved, %d unresolved" % (len(prog.tables), len(prog.unresolved_tables)),
        "bad targets      %d" % len(prog.bad_targets),
        "unsupported      %s" % (unsupported or "none"),
        "files            %d" % len(files),
        "time             %.1f s" % (time.time() - t0),
    ]
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
        for a, t in prog.bad_targets:
            f.write("bad target %08X -> %08X\n" % (a, t))
        for a in prog.unresolved_tables:
            f.write("unresolved table at %08X\n" % a)
    log("\n".join(report))
    return prog


def load_map(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            out[int(p[0], 16)] = (p[2] if len(p) > 2 else "", int(p[1]), p[3] if len(p) > 3 else "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--functions", metavar="TSV", help="function map (va, size, name, source)")
    ap.add_argument("--per-file", type=int, default=8000)
    ap.add_argument("--no-comments", action="store_true")
    a = ap.parse_args()
    elf = Elf(a.elf)
    functions = load_map(a.functions) if a.functions else elf.functions()
    generate(elf, functions, a.out, a.per_file, not a.no_comments)


if __name__ == "__main__":
    main()

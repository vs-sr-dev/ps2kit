"""EE executable loading, address mapping and cross-references.

    python -m ps2kit.elf SCES_502.40 --info
    python -m ps2kit.elf SCES_502.40 --xref 0x270490
    python -m ps2kit.elf SCES_502.40 --callers 0x1C6CE0
    python -m ps2kit.elf SCES_502.40 --read 0x25E8B0 64

PS2 retail executables are stripped, so the most useful question is usually
"who uses this address?". The EE has no PC-relative data loads: a 32-bit
address is built with `lui` followed by `addiu`/`ori` or folded into a
load/store offset. xref() tracks the last `lui` per register and reports every
completed pair that lands on the target.
"""
import argparse
import struct
import sys

# opcodes whose 16-bit immediate completes a lui (I-type, rs = base register)
_ADDIU, _ORI = 0x09, 0x0D
_MEM = {0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x28, 0x29, 0x2B, 0x31, 0x35,
        0x37, 0x39, 0x3D, 0x3F, 0x1E, 0x1F}          # lb..sd, lwc1/swc1, lq/sq
_LUI = 0x0F


class Segment:
    def __init__(self, vaddr, offset, filesz, memsz, flags):
        self.vaddr, self.offset = vaddr, offset
        self.filesz, self.memsz, self.flags = filesz, memsz, flags

    def __repr__(self):
        return "Segment(va=%08X off=%06X filesz=%06X memsz=%06X)" % (
            self.vaddr, self.offset, self.filesz, self.memsz)


class Elf:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        d = self.data
        if d[:4] != b"\x7fELF":
            raise ValueError("not an ELF file")
        self.entry, phoff, shoff = struct.unpack_from("<III", d, 0x18)
        phnum, = struct.unpack_from("<H", d, 0x2C)
        shnum, shstrndx = struct.unpack_from("<HH", d, 0x30)
        self.segments = []
        for i in range(phnum):
            p = struct.unpack_from("<8I", d, phoff + 32 * i)
            if p[0] == 1:                               # PT_LOAD
                self.segments.append(Segment(p[2], p[1], p[4], p[5], p[6]))
        self.sections = {}
        if shnum and shoff:
            sh = [struct.unpack_from("<10I", d, shoff + 40 * i) for i in range(shnum)]
            base = sh[shstrndx][4]
            for s in sh:
                name = d[base + s[0]:d.index(b"\0", base + s[0])].decode("latin-1")
                if name:
                    self.sections[name] = (s[3], s[4], s[5])
        # The loaded image is the first segment with file content.
        self.main = next(s for s in self.segments if s.filesz)

    def comment(self):
        """Compiler identification from .comment, if the linker kept it."""
        if ".comment" not in self.sections:
            return None
        _, off, size = self.sections[".comment"]
        return [p.decode("latin-1") for p in self.data[off:off + size].split(b"\0") if p]

    def va_to_off(self, va):
        for s in self.segments:
            if s.filesz and s.vaddr <= va < s.vaddr + s.filesz:
                return va - s.vaddr + s.offset
        raise ValueError("address %08X is not backed by the file" % va)

    def read(self, va, n):
        o = self.va_to_off(va)
        return self.data[o:o + n]

    def u32(self, va):
        return struct.unpack_from("<I", self.data, self.va_to_off(va))[0]

    def words(self, lo=None, hi=None):
        """Yield (va, word) over the main segment, or over [lo, hi)."""
        s = self.main
        lo = s.vaddr if lo is None else lo
        hi = s.vaddr + s.filesz if hi is None else hi
        o = self.va_to_off(lo)
        n = (hi - lo) // 4
        for i, w in enumerate(struct.unpack_from("<%dI" % n, self.data, o)):
            yield lo + 4 * i, w

    def xref(self, targets, window=64, span=4):
        """Find instructions that complete a lui pair onto any target.

        `span` widens each target into [t, t+span), so a field inside a
        structure still matches. Returns [(site_va, full_address, lui_va)].
        """
        targets = list(targets)
        hits = []
        lui = {}
        for va, w in self.words():
            op, rs, rt, imm = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 0xFFFF
            if op == _LUI:
                lui[rt] = (imm << 16, va)
                continue
            if (op in _MEM or op in (_ADDIU, _ORI)) and rs in lui:
                hi, lva = lui[rs]
                if va - lva > window:
                    continue
                simm = imm - 0x10000 if (imm & 0x8000 and op != _ORI) else imm
                full = (hi + simm) & 0xFFFFFFFF
                for t in targets:
                    if t <= full < t + span:
                        hits.append((va, full, lva))
        return hits

    def callers(self, targets):
        """Every `jal` onto one of the targets: [(site_va, target)].

        Ghidra misses some of these when a function is only reached through
        code it did not disassemble; a plain scan does not."""
        want = {(t >> 2) & 0x3FFFFFF: t for t in targets}
        return [(va, want[w & 0x3FFFFFF]) for va, w in self.words()
                if w >> 26 == 0x03 and w & 0x3FFFFFF in want]

    def find_bytes(self, needle):
        """Every virtual address at which `needle` occurs in the main segment."""
        s, out, i = self.main, [], self.data.find(needle)
        while i != -1:
            if s.offset <= i < s.offset + s.filesz:
                out.append(i - s.offset + s.vaddr)
            i = self.data.find(needle, i + 1)
        return out

    def symbols(self):
        """.symtab entries as (name, value, size, bind, type, shndx); [] if stripped.

        type 2 is a function, 1 an object; bind 0 local, 1 global, 2 weak."""
        if ".symtab" not in self.sections or ".strtab" not in self.sections:
            return []
        _, off, size = self.sections[".symtab"]
        _, soff, _ = self.sections[".strtab"]
        d, out = self.data, []
        for i in range(size // 16):
            n, v, sz, info, _oth, shn = struct.unpack_from("<IIIBBH", d, off + 16 * i)
            name = d[soff + n:d.index(b"\0", soff + n)].decode("latin-1")
            out.append((name, v, sz, info >> 4, info & 15, shn))
        return out

    def functions(self):
        """Every function start the executable names: {va: (name, size, source)}.

        From .symtab (functions with their sizes) and, when present, .mdebug
        (procedures, which add the static copies of C++ inlines that .symtab
        lacks, and the source file of each). A start known only to .mdebug is
        sized up to the next start. `source` is '' when .mdebug is absent."""
        from . import mdebug
        out = {}
        for name, v, sz, _bind, typ, _shn in self.symbols():
            if typ == 2 and v not in out:
                out[v] = [name, sz, ""]
        md = mdebug.from_elf(self)
        if md:
            for p in md.procs():
                if p.address in out:
                    out[p.address][2] = out[p.address][2] or p.file.name
                else:
                    out[p.address] = [p.name, 0, p.file.name]
        starts = sorted(out)
        text_end = {}
        for name, (va, _off, size) in self.sections.items():
            if size and va:
                text_end[name] = va + size
        for i, va in enumerate(starts):
            if out[va][1] == 0:
                nxt = starts[i + 1] if i + 1 < len(starts) else None
                end = min([e for e in text_end.values() if e > va] + ([nxt] if nxt else []))
                out[va][1] = end - va
        return {va: tuple(v) for va, v in out.items()}

    def coverage(self, funcs, section=".text"):
        """How well `funcs` tile a section: (gaps, overlaps), gaps as
        [(va, size, all_zero)], overlaps as [(va, name, bytes)]."""
        lo, off, size = self.sections[section]
        hi, pos, gaps, overlaps = lo + size, lo, [], []
        for va in sorted(funcs):
            if not lo <= va < hi:
                continue
            name, sz, _ = funcs[va]
            if va > pos:
                gaps.append((pos, va - pos, not any(self.read(pos, va - pos))))
            elif va < pos:
                overlaps.append((va, name, pos - va))
            pos = max(pos, va + sz)
        if pos < hi:
            gaps.append((pos, hi - pos, not any(self.read(pos, hi - pos))))
        return gaps, overlaps

    def stats(self):
        """Instruction-mix counters that say a lot about the toolchain."""
        c = {"jr_ra": 0, "jal_targets": set(), "cop2": 0, "mmi": 0, "lq_sq": 0}
        for _, w in self.words():
            op = w >> 26
            if w == 0x03E00008:
                c["jr_ra"] += 1
            elif op == 0x03:
                c["jal_targets"].add(w & 0x3FFFFFF)
            elif op == 0x12:
                c["cop2"] += 1
            elif op == 0x1C:
                c["mmi"] += 1
            elif op in (0x1E, 0x1F):
                c["lq_sq"] += 1
        c["jal_targets"] = len(c["jal_targets"])
        return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elf")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--xref", nargs="+", metavar="VA")
    ap.add_argument("--callers", nargs="+", metavar="VA", help="jal sites onto these functions")
    ap.add_argument("--read", nargs=2, metavar=("VA", "N"))
    ap.add_argument("--functions", metavar="TSV", nargs="?", const="-",
                    help="function map from .symtab/.mdebug and its coverage of .text; "
                         "TSV (va, size, name, source) to a file, or to stdout with -")
    a = ap.parse_args()
    e = Elf(a.elf)
    if a.functions:
        f = e.functions()
        gaps, overlaps = e.coverage(f)
        code_gaps = [g for g in gaps if not g[2]]
        print("%d functions; .text gaps %d (%d bytes), %d of them not all zero (%d bytes); "
              "overlaps %d" % (len(f), len(gaps), sum(g[1] for g in gaps), len(code_gaps),
                               sum(g[1] for g in code_gaps), len(overlaps)),
              file=sys.stderr)
        for va, size, _ in code_gaps:
            print("  gap %08X %d bytes" % (va, size), file=sys.stderr)
        for va, name, n in overlaps:
            print("  overlap %08X %s %d bytes" % (va, name, n), file=sys.stderr)
        rows = ["%08X\t%d\t%s\t%s" % (va, f[va][1], f[va][0], f[va][2]) for va in sorted(f)]
        if a.functions == "-":
            print("\n".join(rows))
        else:
            with open(a.functions, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(rows) + "\n")
    if a.info:
        print("entry   %08X" % e.entry)
        print("comment %s" % e.comment())
        for s in e.segments:
            print("  ", s)
        for k, v in e.stats().items():
            print("  %-12s %d" % (k, v))
    if a.xref:
        for va, full, lva in e.xref(int(x, 0) for x in a.xref):
            print("%08X  -> %08X  (lui at %08X)" % (va, full, lva))
    if a.callers:
        for va, t in e.callers(int(x, 0) for x in a.callers):
            print("%08X  jal %08X" % (va, t))
    if a.read:
        va, n = int(a.read[0], 0), int(a.read[1], 0)
        b = e.read(va, n)
        for i in range(0, len(b), 16):
            print("%08X  %s" % (va + i, b[i:i + 16].hex(" ")))


if __name__ == "__main__":
    main()

"""ECOFF symbolic debug information (.mdebug), as the SDK's ee-gcc 2.9x left it.

    python -m ps2kit.mdebug SLES_500.57 --files
    python -m ps2kit.mdebug SLES_500.57 --procs

Some PS2 executables shipped with their .mdebug section: the MIPS ECOFF
symbol table that GNU as writes for MIPS targets, kept by ld in ELF files.
It names every source file (with the path it was built from) and every
procedure, including the static copies of C++ inline functions that GCC 2.9x
emits once per translation unit and that never reach .symtab.

Layout (all offsets in the header are file offsets, not section offsets):

HDRR   96 bytes at the start of the section: magic 0x7009, vstamp, then
       (count, offset) pairs for line numbers, dense numbers, procedure
       descriptors, local symbols, optimisation entries, auxiliary symbols,
       local strings, external strings, file descriptors, relative file
       descriptors and external symbols.
FDR    72 bytes per source file: address, name (local string index, relative
       to the file's own string base), the file's string base and size, its
       first local symbol and count, line numbers, first procedure (16-bit)
       and count (16-bit), aux and rfd bases, flags, line table offset/size.
PDR    52 bytes per procedure: address relative to its file's, the index of
       its stProc symbol (relative to the file's first symbol), line base,
       saved register masks and offsets, frame size, frame and return
       registers, line range, line table offset.
SYMR   12 bytes per local symbol: string index, value, then st (6 bits:
       1 global, 2 static, 3 param, 4 local, 5 label, 6 proc, 7 block,
       8 end, 9 member, 10 typedef, 11 file, 14 static proc), sc (5 bits,
       storage class), 1 reserved bit, 20-bit index.
"""
import argparse
import struct

ST_NAMES = {0: "nil", 1: "global", 2: "static", 3: "param", 4: "local",
            5: "label", 6: "proc", 7: "block", 8: "end", 9: "member",
            10: "typedef", 11: "file", 14: "staticproc", 15: "constant"}

_HDR_KEYS = ("ilineMax cbLine cbLineOffset idnMax cbDnOffset ipdMax cbPdOffset "
             "isymMax cbSymOffset ioptMax cbOptOffset iauxMax cbAuxOffset "
             "issMax cbSsOffset issExtMax cbSsExtOffset ifdMax cbFdOffset "
             "crfd cbRfdOffset iextMax cbExtOffset").split()


class File:
    def __init__(self, index, name, address, procs, isym_base, csym):
        self.index, self.name, self.address = index, name, address
        self.procs = procs                 # [Proc]
        self.isym_base, self.csym = isym_base, csym

    def __repr__(self):
        return "<File %s @%08X %d procs>" % (self.name, self.address, len(self.procs))


class Proc:
    def __init__(self, name, address, file, frame, frame_reg, pc_reg, regmask, fregmask):
        self.name, self.address, self.file = name, address, file
        self.frame, self.frame_reg, self.pc_reg = frame, frame_reg, pc_reg
        self.regmask, self.fregmask = regmask, fregmask

    def __repr__(self):
        return "<Proc %s @%08X>" % (self.name, self.address)


class Mdebug:
    def __init__(self, data, offset):
        """`data` is the whole ELF file, `offset` the file offset of .mdebug."""
        self.data = data
        magic, vstamp = struct.unpack_from("<hh", data, offset)
        if magic & 0xFFFF != 0x7009:
            raise ValueError("no ECOFF symbolic header (magic %04X)" % (magic & 0xFFFF))
        self.h = dict(zip(_HDR_KEYS, struct.unpack_from("<23i", data, offset + 4)))
        self.files = []
        h = self.h
        for i in range(h["ifdMax"]):
            f = struct.unpack_from("<IIIiiiiiiihH", data, h["cbFdOffset"] + 72 * i)
            adr, rss, iss_base, _cb_ss, isym_base, csym = f[:6]
            ipd_first, cpd = f[10], f[11]
            name = self._str(iss_base, rss) if rss != 0xFFFFFFFF else ""
            fobj = File(i, name, adr, [], isym_base, csym)
            for j in range(cpd):
                p = struct.unpack_from("<IiiIiiIiihhiiI", data, h["cbPdOffset"] + 52 * (ipd_first + j))
                padr, isym, _iline, regmask, _regoff, _iopt, fregmask, _fregoff, frame, \
                    frame_reg, pc_reg = p[:11]
                iss, value, _bits = struct.unpack_from("<IiI", data, h["cbSymOffset"] + 12 * (isym_base + isym))
                # the stProc symbol's value is the absolute address; padr is relative to the file
                fobj.procs.append(Proc(self._str(iss_base, iss), value & 0xFFFFFFFF, fobj,
                                       frame, frame_reg, pc_reg, regmask, fregmask))
            self.files.append(fobj)

    def _str(self, base, index):
        o = self.h["cbSsOffset"] + base + index
        return self.data[o:self.data.index(b"\0", o)].decode("latin-1")

    def procs(self):
        for f in self.files:
            yield from f.procs

    def symbols(self, f):
        """(name, value, st, sc, index) of a file's local symbols."""
        for k in range(f.csym):
            iss, value, bits = struct.unpack_from("<IiI", self.data, self.h["cbSymOffset"] + 12 * (f.isym_base + k))
            yield self._str_of(f, iss), value, bits & 0x3F, (bits >> 6) & 0x1F, bits >> 12

    def _str_of(self, f, iss):
        fd = struct.unpack_from("<III", self.data, self.h["cbFdOffset"] + 72 * f.index)
        return self._str(fd[2], iss)


def from_elf(elf):
    """The Mdebug of a ps2kit.elf.Elf, or None."""
    if ".mdebug" not in elf.sections:
        return None
    return Mdebug(elf.data, elf.sections[".mdebug"][1])


def main():
    from .elf import Elf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--files", action="store_true", help="source files with procedure counts")
    ap.add_argument("--procs", action="store_true", help="every procedure: address, frame, file")
    a = ap.parse_args()
    m = from_elf(Elf(a.elf))
    if m is None:
        raise SystemExit("no .mdebug section")
    print("%d files, %d procedures, %d local symbols" % (
        len(m.files), m.h["ipdMax"], m.h["isymMax"]))
    if a.files:
        for f in m.files:
            print("%08X %4d  %s" % (f.address, len(f.procs), f.name))
    if a.procs:
        for p in m.procs():
            print("%08X frame %4d  %-50s %s" % (p.address, p.frame, p.name, p.file.name))


if __name__ == "__main__":
    main()

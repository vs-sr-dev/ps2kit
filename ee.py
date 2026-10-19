"""The Emotion Engine's R5900 instruction set: decode, disassemble, census.

    python -m ps2kit.ee GAME.ELF --at 0x200008 [--count 32]
    python -m ps2kit.ee GAME.ELF --census [--functions map.tsv]
    python -m ps2kit.ee GAME.ELF --against listing.tsv     # compare with Ghidra

The R5900 is a MIPS III core (64-bit GPRs, here 128 bits wide) with Sony's
additions: the MMI group (opcode 0x1C: 128-bit integer SIMD, a second
HI1/LO1 pipeline, three-operand MULT/MADD), LQ/SQ (128-bit loads and
stores), a single-precision-only FPU with an accumulator (ADDA, MADD, ...),
COP2 as VU0 in macro mode, and COP0 extras (EI/DI, the performance
counter). Not all MIPS III is there: no LL/SC, no double-precision FPU.

decode(word, va) returns an Insn: `op` is the canonical mnemonic (never an
alias such as `nop`, `move` or `b`; those are only for display), `form` names
the operand layout, and the raw fields are kept (rs, rt, rd, sa, imm,
target). The recompiler works on `op` and the fields, never on the text.

Encodings come from the R5900 manuals' opcode maps: primary opcode in bits
31-26; SPECIAL (0x00) and MMI (0x1C) by function in bits 5-0; REGIMM (0x01)
by rt; MMI0-3 (MMI functions 0x08, 0x28, 0x09, 0x29) by bits 10-6; COP0,
COP1 and COP2 by rs (bits 25-21), then by function; COP2 with bit 25 set is
a VU0 macro instruction, decoded with the VU upper/lower tables of
ps2kit.vu (functions 0x3C-0x3F take their real opcode from bits 10-6 and
1-0, as in VU micro mode).
"""
import argparse
import collections
import struct

from . import vu

GPR = ("zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
       "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
       "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
       "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra")

PRIMARY = {
    0x02: ("j", "jump"), 0x03: ("jal", "jump"),
    0x04: ("beq", "rs_rt_br"), 0x05: ("bne", "rs_rt_br"), 0x06: ("blez", "rs_br"), 0x07: ("bgtz", "rs_br"),
    0x08: ("addi", "rt_rs_simm"), 0x09: ("addiu", "rt_rs_simm"), 0x0A: ("slti", "rt_rs_simm"),
    0x0B: ("sltiu", "rt_rs_simm"), 0x0C: ("andi", "rt_rs_uimm"), 0x0D: ("ori", "rt_rs_uimm"),
    0x0E: ("xori", "rt_rs_uimm"), 0x0F: ("lui", "rt_uimm"),
    0x14: ("beql", "rs_rt_br"), 0x15: ("bnel", "rs_rt_br"), 0x16: ("blezl", "rs_br"), 0x17: ("bgtzl", "rs_br"),
    0x18: ("daddi", "rt_rs_simm"), 0x19: ("daddiu", "rt_rs_simm"),
    0x1A: ("ldl", "rt_mem"), 0x1B: ("ldr", "rt_mem"), 0x1E: ("lq", "rt_mem"), 0x1F: ("sq", "rt_mem"),
    0x20: ("lb", "rt_mem"), 0x21: ("lh", "rt_mem"), 0x22: ("lwl", "rt_mem"), 0x23: ("lw", "rt_mem"),
    0x24: ("lbu", "rt_mem"), 0x25: ("lhu", "rt_mem"), 0x26: ("lwr", "rt_mem"), 0x27: ("lwu", "rt_mem"),
    0x28: ("sb", "rt_mem"), 0x29: ("sh", "rt_mem"), 0x2A: ("swl", "rt_mem"), 0x2B: ("sw", "rt_mem"),
    0x2C: ("sdl", "rt_mem"), 0x2D: ("sdr", "rt_mem"), 0x2E: ("swr", "rt_mem"), 0x2F: ("cache", "op_mem"),
    0x31: ("lwc1", "ft_mem"), 0x33: ("pref", "op_mem"), 0x36: ("lqc2", "vt_mem"), 0x37: ("ld", "rt_mem"),
    0x39: ("swc1", "ft_mem"), 0x3E: ("sqc2", "vt_mem"), 0x3F: ("sd", "rt_mem"),
}

SPECIAL = {
    0x00: ("sll", "rd_rt_sa"), 0x02: ("srl", "rd_rt_sa"), 0x03: ("sra", "rd_rt_sa"),
    0x04: ("sllv", "rd_rt_rs"), 0x06: ("srlv", "rd_rt_rs"), 0x07: ("srav", "rd_rt_rs"),
    0x08: ("jr", "rs"), 0x09: ("jalr", "rd_rs"), 0x0A: ("movz", "rd_rs_rt"), 0x0B: ("movn", "rd_rs_rt"),
    0x0C: ("syscall", "code"), 0x0D: ("break", "code"), 0x0F: ("sync", "sync"),
    0x10: ("mfhi", "rd"), 0x11: ("mthi", "rs"), 0x12: ("mflo", "rd"), 0x13: ("mtlo", "rs"),
    0x14: ("dsllv", "rd_rt_rs"), 0x16: ("dsrlv", "rd_rt_rs"), 0x17: ("dsrav", "rd_rt_rs"),
    0x18: ("mult", "rd_rs_rt"), 0x19: ("multu", "rd_rs_rt"), 0x1A: ("div", "rs_rt"), 0x1B: ("divu", "rs_rt"),
    0x20: ("add", "rd_rs_rt"), 0x21: ("addu", "rd_rs_rt"), 0x22: ("sub", "rd_rs_rt"), 0x23: ("subu", "rd_rs_rt"),
    0x24: ("and", "rd_rs_rt"), 0x25: ("or", "rd_rs_rt"), 0x26: ("xor", "rd_rs_rt"), 0x27: ("nor", "rd_rs_rt"),
    0x28: ("mfsa", "rd"), 0x29: ("mtsa", "rs"), 0x2A: ("slt", "rd_rs_rt"), 0x2B: ("sltu", "rd_rs_rt"),
    0x2C: ("dadd", "rd_rs_rt"), 0x2D: ("daddu", "rd_rs_rt"), 0x2E: ("dsub", "rd_rs_rt"), 0x2F: ("dsubu", "rd_rs_rt"),
    0x30: ("tge", "rs_rt"), 0x31: ("tgeu", "rs_rt"), 0x32: ("tlt", "rs_rt"), 0x33: ("tltu", "rs_rt"),
    0x34: ("teq", "rs_rt"), 0x36: ("tne", "rs_rt"),
    0x38: ("dsll", "rd_rt_sa"), 0x3A: ("dsrl", "rd_rt_sa"), 0x3B: ("dsra", "rd_rt_sa"),
    0x3C: ("dsll32", "rd_rt_sa"), 0x3E: ("dsrl32", "rd_rt_sa"), 0x3F: ("dsra32", "rd_rt_sa"),
}

REGIMM = {
    0x00: ("bltz", "rs_br"), 0x01: ("bgez", "rs_br"), 0x02: ("bltzl", "rs_br"), 0x03: ("bgezl", "rs_br"),
    0x08: ("tgei", "rs_simm"), 0x09: ("tgeiu", "rs_simm"), 0x0A: ("tlti", "rs_simm"), 0x0B: ("tltiu", "rs_simm"),
    0x0C: ("teqi", "rs_simm"), 0x0E: ("tnei", "rs_simm"),
    0x10: ("bltzal", "rs_br"), 0x11: ("bgezal", "rs_br"), 0x12: ("bltzall", "rs_br"), 0x13: ("bgezall", "rs_br"),
    0x18: ("mtsab", "rs_simm"), 0x19: ("mtsah", "rs_simm"),
}

MMI = {
    0x00: ("madd", "rd_rs_rt"), 0x01: ("maddu", "rd_rs_rt"), 0x04: ("plzcw", "rd_rs"),
    0x10: ("mfhi1", "rd"), 0x11: ("mthi1", "rs"), 0x12: ("mflo1", "rd"), 0x13: ("mtlo1", "rs"),
    0x18: ("mult1", "rd_rs_rt"), 0x19: ("multu1", "rd_rs_rt"), 0x1A: ("div1", "rs_rt"), 0x1B: ("divu1", "rs_rt"),
    0x20: ("madd1", "rd_rs_rt"), 0x21: ("maddu1", "rd_rs_rt"),
    0x34: ("psllh", "rd_rt_sa"), 0x36: ("psrlh", "rd_rt_sa"), 0x37: ("psrah", "rd_rt_sa"),
    0x3C: ("psllw", "rd_rt_sa"), 0x3E: ("psrlw", "rd_rt_sa"), 0x3F: ("psraw", "rd_rt_sa"),
}
PMFHL = {0: "pmfhl.lw", 1: "pmfhl.uw", 2: "pmfhl.slw", 3: "pmfhl.lh", 4: "pmfhl.sh"}

_R = "rd_rs_rt"
MMI0 = {0x00: "paddw", 0x01: "psubw", 0x02: "pcgtw", 0x03: "pmaxw", 0x04: "paddh", 0x05: "psubh",
        0x06: "pcgth", 0x07: "pmaxh", 0x08: "paddb", 0x09: "psubb", 0x0A: "pcgtb", 0x10: "paddsw",
        0x11: "psubsw", 0x12: "pextlw", 0x13: "ppacw", 0x14: "paddsh", 0x15: "psubsh", 0x16: "pextlh",
        0x17: "ppach", 0x18: "paddsb", 0x19: "psubsb", 0x1A: "pextlb", 0x1B: "ppacb",
        0x1E: ("pext5", "rd_rt"), 0x1F: ("ppac5", "rd_rt")}
MMI1 = {0x01: ("pabsw", "rd_rt"), 0x02: "pceqw", 0x03: "pminw", 0x04: "padsbh", 0x05: ("pabsh", "rd_rt"),
        0x06: "pceqh", 0x07: "pminh", 0x0A: "pceqb", 0x10: "padduw", 0x11: "psubuw", 0x12: "pextuw",
        0x14: "padduh", 0x15: "psubuh", 0x16: "pextuh", 0x18: "paddub", 0x19: "psubub", 0x1A: "pextub",
        0x1B: "qfsrv"}
MMI2 = {0x00: "pmaddw", 0x02: ("psllvw", "rd_rt_rs"), 0x03: ("psrlvw", "rd_rt_rs"), 0x04: "pmsubw",
        0x08: ("pmfhi", "rd"), 0x09: ("pmflo", "rd"), 0x0A: "pinth", 0x0C: "pmultw", 0x0D: ("pdivw", "rs_rt"),
        0x0E: "pcpyld", 0x10: "pmaddh", 0x11: "phmadh", 0x12: "pand", 0x13: "pxor", 0x14: "pmsubh",
        0x15: "phmsbh", 0x1A: ("pexeh", "rd_rt"), 0x1B: ("prevh", "rd_rt"), 0x1C: "pmulth",
        0x1D: ("pdivbw", "rs_rt"), 0x1E: ("pexew", "rd_rt"), 0x1F: ("prot3w", "rd_rt")}
MMI3 = {0x00: "pmadduw", 0x03: ("psravw", "rd_rt_rs"), 0x08: ("pmthi", "rs"), 0x09: ("pmtlo", "rs"),
        0x0A: "pinteh", 0x0C: "pmultuw", 0x0D: ("pdivuw", "rs_rt"), 0x0E: "pcpyud", 0x12: "por",
        0x13: "pnor", 0x1A: ("pexch", "rd_rt"), 0x1B: ("pcpyh", "rd_rt"), 0x1E: ("pexcw", "rd_rt")}
MMI_SUB = {0x08: MMI0, 0x28: MMI1, 0x09: MMI2, 0x29: MMI3}

COP0_C0 = {0x01: "tlbr", 0x02: "tlbwi", 0x06: "tlbwr", 0x08: "tlbp", 0x18: "eret", 0x38: "ei", 0x39: "di"}
BCX = {0: "f", 1: "t", 2: "fl", 3: "tl"}

COP1_S = {0x00: ("add.s", "fd_fs_ft"), 0x01: ("sub.s", "fd_fs_ft"), 0x02: ("mul.s", "fd_fs_ft"),
          0x03: ("div.s", "fd_fs_ft"), 0x04: ("sqrt.s", "fd_ft"), 0x05: ("abs.s", "fd_fs"),
          0x06: ("mov.s", "fd_fs"), 0x07: ("neg.s", "fd_fs"), 0x16: ("rsqrt.s", "fd_fs_ft"),
          0x18: ("adda.s", "fs_ft"), 0x19: ("suba.s", "fs_ft"), 0x1A: ("mula.s", "fs_ft"),
          0x1C: ("madd.s", "fd_fs_ft"), 0x1D: ("msub.s", "fd_fs_ft"), 0x1E: ("madda.s", "fs_ft"),
          0x1F: ("msuba.s", "fs_ft"), 0x24: ("cvt.w.s", "fd_fs"), 0x28: ("max.s", "fd_fs_ft"),
          0x29: ("min.s", "fd_fs_ft"), 0x30: ("c.f.s", "fs_ft"), 0x32: ("c.eq.s", "fs_ft"),
          0x34: ("c.lt.s", "fs_ft"), 0x36: ("c.le.s", "fs_ft")}

BRANCHES = {"beq", "bne", "blez", "bgtz", "beql", "bnel", "blezl", "bgtzl", "bltz", "bgez",
            "bltzl", "bgezl", "bltzal", "bgezal", "bltzall", "bgezall",
            "bc0f", "bc0t", "bc0fl", "bc0tl", "bc1f", "bc1t", "bc1fl", "bc1tl",
            "bc2f", "bc2t", "bc2fl", "bc2tl"}
JUMPS = {"j", "jal", "jr", "jalr"}


class Insn:
    __slots__ = ("va", "word", "op", "form", "rs", "rt", "rd", "sa", "imm", "target", "extra")

    def __init__(self, va, word, op, form, imm=0, target=None, extra=None):
        self.va, self.word, self.op, self.form = va, word, op, form
        self.rs, self.rt, self.rd, self.sa = word >> 21 & 31, word >> 16 & 31, word >> 11 & 31, word >> 6 & 31
        self.imm, self.target, self.extra = imm, target, extra

    @property
    def is_branch(self):
        return self.op in BRANCHES

    @property
    def is_likely(self):
        """Branch-likely: the delay slot runs only when the branch is taken."""
        return self.op in BRANCHES and self.op.endswith("l") and self.op not in ("bltzal", "bgezal")

    @property
    def has_delay_slot(self):
        return self.op in BRANCHES or self.op in JUMPS

    @property
    def valid(self):
        return not self.op.startswith("?")

    def text(self):
        return disasm(self)

    def __repr__(self):
        return "<%08X %s>" % (self.va, disasm(self))


def _s16(x):
    return x - 0x10000 if x & 0x8000 else x


def decode(w, va=0):
    op = w >> 26
    if op == 0x00:
        f = w & 0x3F
        if f in SPECIAL:
            name, form = SPECIAL[f]
            if name == "sync":
                name = "sync.p" if w >> 6 & 31 == 0x10 else "sync.l"
            return Insn(va, w, name, form, imm=w >> 6 & 0xFFFFF)
        return Insn(va, w, "?special", "")
    if op == 0x01:
        rt = w >> 16 & 31
        if rt in REGIMM:
            name, form = REGIMM[rt]
            return _imm_insn(va, w, name, form)
        return Insn(va, w, "?regimm", "")
    if op == 0x1C:
        return _mmi(va, w)
    if op == 0x10:
        return _cop0(va, w)
    if op == 0x11:
        return _cop1(va, w)
    if op == 0x12:
        return _cop2(va, w)
    if op in PRIMARY:
        name, form = PRIMARY[op]
        if form == "jump":
            return Insn(va, w, name, form, target=((va + 4) & 0xF0000000) | (w & 0x3FFFFFF) << 2)
        return _imm_insn(va, w, name, form)
    return Insn(va, w, "?op%02X" % op, "")


def _imm_insn(va, w, name, form):
    imm = w & 0xFFFF
    if form.endswith("_br"):
        return Insn(va, w, name, form, imm=_s16(imm), target=(va + 4 + (_s16(imm) << 2)) & 0xFFFFFFFF)
    if form.endswith("uimm"):
        return Insn(va, w, name, form, imm=imm)
    return Insn(va, w, name, form, imm=_s16(imm))


def _mmi(va, w):
    f = w & 0x3F
    if f in MMI_SUB:
        e = MMI_SUB[f].get(w >> 6 & 31)
        if e is None:
            return Insn(va, w, "?mmi%X" % f, "")
        name, form = (e, _R) if isinstance(e, str) else e
        return Insn(va, w, name, form)
    if f == 0x30:
        n = PMFHL.get(w >> 6 & 31)
        return Insn(va, w, n, "rd") if n else Insn(va, w, "?pmfhl", "")
    if f == 0x31:
        return Insn(va, w, "pmthl.lw", "rs") if w >> 6 & 31 == 0 else Insn(va, w, "?pmthl", "")
    if f in MMI:
        name, form = MMI[f]
        return Insn(va, w, name, form)
    return Insn(va, w, "?mmi", "")


def _cop0(va, w):
    rs = w >> 21 & 31
    if rs == 0x00:
        return Insn(va, w, "mfc0", "rt_c0")
    if rs == 0x04:
        return Insn(va, w, "mtc0", "rt_c0")
    if rs == 0x08 and (w >> 16 & 31) in BCX:
        return _imm_insn(va, w, "bc0" + BCX[w >> 16 & 31], "_br")
    if rs == 0x10 and (w & 0x3F) in COP0_C0:
        return Insn(va, w, COP0_C0[w & 0x3F], "")
    return Insn(va, w, "?cop0", "")


def _cop1(va, w):
    rs = w >> 21 & 31
    simple = {0x00: "mfc1", 0x02: "cfc1", 0x04: "mtc1", 0x06: "ctc1"}
    if rs in simple:
        return Insn(va, w, simple[rs], "rt_fs")
    if rs == 0x08 and (w >> 16 & 31) in BCX:
        return _imm_insn(va, w, "bc1" + BCX[w >> 16 & 31], "_br")
    if rs == 0x10 and (w & 0x3F) in COP1_S:
        name, form = COP1_S[w & 0x3F]
        return Insn(va, w, name, form)
    if rs == 0x14 and w & 0x3F == 0x20:
        return Insn(va, w, "cvt.s.w", "fd_fs")
    return Insn(va, w, "?cop1", "")


def _cop2(va, w):
    rs = w >> 21 & 31
    if rs < 0x10:
        simple = {0x01: "qmfc2", 0x02: "cfc2", 0x05: "qmtc2", 0x06: "ctc2"}
        if rs in simple:
            return Insn(va, w, simple[rs] + (".i" if w & 1 else ".ni"), "rt_c2")
        if rs == 0x08 and (w >> 16 & 31) in BCX:
            return _imm_insn(va, w, "bc2" + BCX[w >> 16 & 31], "_br")
        return Insn(va, w, "?cop2", "")
    # VU0 macro mode: bit 25 set; the VU tables, with a "v" prefix
    f = w & 0x3F
    if f < 0x30:
        e = vu.UPPER.get(f)
        return Insn(va, w, "v" + e[0], "vu_upper", extra=e[1]) if e else Insn(va, w, "?vu0", "")
    if f >= 0x3C:
        x = (w >> 6 & 31) << 2 | f & 3
        e = vu.UPPER_X.get(x) if x < 0x30 else vu.LOWER_X.get(x)
        if e is None or (x >= 0x30 and x > 0x43):
            return Insn(va, w, "?vu0x", "")
        return Insn(va, w, "v" + e[0], "vu_upper" if x < 0x30 else "vu_lower", extra=e[1])
    if f in vu.LOWER_R:
        e = vu.LOWER_R[f]
        return Insn(va, w, "v" + e[0], "vu_int", extra=e[1])
    if f == 0x38:
        return Insn(va, w, "vcallms", "vcallms", imm=(w >> 6 & 0x7FFF) << 3)
    if f == 0x39:
        return Insn(va, w, "vcallmsr", "")
    return Insn(va, w, "?vu0", "")


# ---------------------------------------------------------------- text

def _hex(v):
    return "-0x%x" % -v if v < 0 else "0x%x" % v


def disasm(i):
    g = GPR
    f = i.form
    if i.form == "vu_upper":
        return "v" + vu.upper(i.word & 0x01FFFFFF)
    if i.form == "vu_lower":
        return "v" + vu.lower(0x80000000 | i.word & 0x01FFFFFF)
    if i.form == "vu_int":
        return "v" + vu.lower(0x80000000 | i.word & 0x01FFFFFF)
    ops = {
        "": "", "jump": "0x%08x" % (i.target or 0), "code": "0x%x" % i.imm, "sync": "",
        "rs_rt_br": "%s, %s, 0x%08x" % (g[i.rs], g[i.rt], i.target or 0),
        "rs_br": "%s, 0x%08x" % (g[i.rs], i.target or 0), "_br": "0x%08x" % (i.target or 0),
        "rt_rs_simm": "%s, %s, %s" % (g[i.rt], g[i.rs], _hex(i.imm)),
        "rt_rs_uimm": "%s, %s, 0x%x" % (g[i.rt], g[i.rs], i.imm),
        "rt_uimm": "%s, 0x%x" % (g[i.rt], i.imm), "rs_simm": "%s, %s" % (g[i.rs], _hex(i.imm)),
        "rt_mem": "%s, %s(%s)" % (g[i.rt], _hex(i.imm), g[i.rs]),
        "ft_mem": "$f%d, %s(%s)" % (i.rt, _hex(i.imm), g[i.rs]),
        "vt_mem": "vf%d, %s(%s)" % (i.rt, _hex(i.imm), g[i.rs]),
        "op_mem": "0x%x, %s(%s)" % (i.rt, _hex(i.imm), g[i.rs]),
        "rd_rt_sa": "%s, %s, %d" % (g[i.rd], g[i.rt], i.sa),
        "rd_rt_rs": "%s, %s, %s" % (g[i.rd], g[i.rt], g[i.rs]),
        "rd_rs_rt": "%s, %s, %s" % (g[i.rd], g[i.rs], g[i.rt]),
        "rd_rs": "%s, %s" % (g[i.rd], g[i.rs]), "rd_rt": "%s, %s" % (g[i.rd], g[i.rt]),
        "rs_rt": "%s, %s" % (g[i.rs], g[i.rt]), "rd": g[i.rd], "rs": g[i.rs],
        "rt_c0": "%s, $%d" % (g[i.rt], i.rd), "rt_fs": "%s, $f%d" % (g[i.rt], i.rd),
        "rt_c2": "%s, vf%d" % (g[i.rt], i.rd) if i.op.startswith("q") else "%s, vi%d" % (g[i.rt], i.rd),
        "fd_fs_ft": "$f%d, $f%d, $f%d" % (i.sa, i.rd, i.rt), "fd_fs": "$f%d, $f%d" % (i.sa, i.rd),
        "fd_ft": "$f%d, $f%d" % (i.sa, i.rt), "fs_ft": "$f%d, $f%d" % (i.rd, i.rt),
        "vcallms": "0x%x" % i.imm,
    }.get(f, "?")
    return ("%-10s %s" % (i.op, ops)).rstrip()


# ---------------------------------------------------------------- CLI

def _ranges(elf, tsv):
    if tsv:
        for line in open(tsv, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            yield int(p[0], 16), int(p[1]), p[2] if len(p) > 2 else ""
    else:
        for va, (name, size, _src) in sorted(elf.functions().items()):
            yield va, size, name


def census(elf, tsv=None):
    """{op: count} over every function range, plus the undecoded words."""
    c, bad = collections.Counter(), []
    for va, size, name in _ranges(elf, tsv):
        for a, w in elf.words(va, va + size):
            i = decode(w, a)
            c[i.op] += 1
            if not i.valid:
                bad.append((a, w, name))
    return c, bad


# Ghidra (ghidra-emotionengine-reloaded) shows aliases where we keep the
# canonical operation; these pairs are the same instruction.
_ALIASES = {("sll", "nop"), ("beq", "b"), ("beq", "beqz"), ("bne", "bnez"), ("addu", "move"),
            ("daddu", "move"), ("or", "move"), ("bgezal", "bal"), ("addiu", "li"), ("ori", "li"),
            ("subu", "negu"), ("dsubu", "dnegu"), ("nor", "not"), ("sync.l", "sync"),
            ("beql", "beqzl"), ("bnel", "bnezl"), ("beql", "bl"), ("jalr", "jalr.hb"), ("por", "pmove"),
            ("daddiu", "dli"), ("sub", "neg"), ("dsub", "dneg")}


def _normalise(i, theirs, operands):
    """Ghidra's mnemonic brought to our naming, or None when it does not
    describe this instruction. Its VU0 macro mnemonics carry the dest field
    (vadd.xyzw), which is checked against the word's bits 24-21, not dropped;
    it leaves out the COP2 transfers' interlock bit (.i/.ni), so that is the
    one thing this comparison cannot see."""
    ours = i.op
    if i.form.startswith("vu_") and "." in theirs:
        base, dest = theirs.split(".", 1)
        if dest != vu._dest(i.word):
            return None
        theirs = base.replace("minibc", "mini")
    if ours.endswith((".i", ".ni")) and ours.startswith(("qmfc2", "qmtc2", "cfc2", "ctc2")):
        ours = ours.rsplit(".", 1)[0]
    if theirs == "clear":          # mtc1 zero, $fN / por rd, zero, zero
        theirs = ours if (ours == "mtc1" and i.rt == 0) or (ours == "por" and i.rs == i.rt == 0) else theirs
    if theirs == "moveq" and ours == "por" and i.rt == 0:
        theirs = ours
    if ours == "sync.p" and theirs == "sync" and operands == "0x10":
        theirs = ours
    return theirs if theirs == ours or (ours, theirs) in _ALIASES else None


def against(elf, listing):
    """Compare our decoding with a Ghidra listing (ps2kit/ghidra/ExportListing.java):
    the operation (aliases and Ghidra's VU0 dest suffix normalised) and every
    branch and jump target."""
    same, alias, diff, n = 0, collections.Counter(), collections.Counter(), 0
    examples, targets, target_diff = {}, 0, []
    for line in open(listing, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        va, theirs, operands = int(p[0], 16), p[1].lower().lstrip("_"), p[2] if len(p) > 2 else ""
        i = decode(elf.u32(va), va)
        n += 1
        if i.op == theirs:
            same += 1
        elif _normalise(i, theirs, operands) is not None:
            alias[(i.op, theirs.split(".")[0] if i.form.startswith("vu_") else theirs)] += 1
        else:
            diff[(i.op, theirs)] += 1
            examples.setdefault((i.op, theirs), (va, operands))
        if i.target is not None and i.op not in ("jr", "jalr"):
            targets += 1
            last = operands.split(",")[-1]
            if int(last, 16) != i.target:
                target_diff.append((va, i.target, last))
    return n, same, alias, diff, examples, targets, target_diff


def main():
    from .elf import Elf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--at", metavar="VA")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--functions", metavar="TSV", help="function ranges (va, size, name); default: the ELF's own")
    ap.add_argument("--against", metavar="LISTING", help="compare with a Ghidra listing TSV")
    a = ap.parse_args()
    e = Elf(a.elf)
    if a.at:
        va = int(a.at, 0)
        for k in range(a.count):
            i = decode(e.u32(va + 4 * k), va + 4 * k)
            print("%08X  %08X  %s" % (i.va, i.word, i.text()))
    if a.census:
        c, bad = census(e, a.functions)
        total = sum(c.values())
        print("%d instructions, %d distinct operations, %d undecoded" % (total, len(c), len(bad)))
        for op, n in c.most_common():
            print("  %-12s %7d" % (op, n))
        for va, w, name in bad[:50]:
            print("  undecoded %08X %08X in %s" % (va, w, name))
    if a.against:
        n, same, alias, diff, ex, nt, td = against(e, a.against)
        print("%d instructions: %d same mnemonic, %d aliases or notation, %d different; "
              "%d branch/jump targets, %d different" % (
                  n, same, sum(alias.values()), sum(diff.values()), nt, len(td)))
        for va, ours, theirs in td[:20]:
            print("  TARGET %08X ours %08X ghidra %s" % (va, ours, theirs))
        for k, v in alias.most_common():
            print("  alias %-10s = %-10s %7d" % (k[0], k[1], v))
        for k, v in diff.most_common():
            print("  DIFF  ours %-12s ghidra %-12s %6d  e.g. %08X %s" % (k[0], k[1], v, ex[k][0], ex[k][1]))


if __name__ == "__main__":
    main()

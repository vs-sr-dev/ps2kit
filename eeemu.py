"""A reference interpreter for EE code, the oracle of the recompiler's tests.

    python -m ps2kit.eeemu GAME.ELF --call 0x2006A0 --a0 1 --a1 0x3E6A84 [--trace]

It runs the executable's own functions one instruction at a time, delay
slots in their natural order, from the ELF's loaded image, until the
function returns to the sentinel return address. It is written from the
instruction semantics (ps2kit/runtime/ee.h documents the same choices), not
from the recompiler's emitter, so that a translation mistake shows up as a
difference between the two (see ps2kit.recomp.difftest).

Floating point follows the runtime's model, not the hardware's: denormals
flushed, infinities and NaNs clamped to the largest finite value, every
result rounded toward zero (computed exactly, then truncated). The tests check the translation; how close the
model is to the PS2 is a separate question, for PCSX2 and the console.

Not modelled: system calls, traps, hardware registers, VU0 micro mode (they
raise Stop), timing.
"""
import argparse
import collections
import math
import struct
from fractions import Fraction

from . import ee, vu

STOP = 0xFFFFFFF0            # sentinel return address
M64, M32, M128 = (1 << 64) - 1, (1 << 32) - 1, (1 << 128) - 1
RAM = 0x02000000


class Stop(Exception):
    pass


def s32(v):
    """Sign-extend the low 32 bits to 64 (as an unsigned 64-bit value)."""
    v &= M32
    return (v | 0xFFFFFFFF00000000) if v & 0x80000000 else v


def sx(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


# ---- floating point, the runtime's model -------------------------------------------

def bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 | (0x80000000 if x < 0 else 0)


def fl(u):
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


def ps2f_bits(u):
    e = u & 0x7F800000
    if e == 0:
        return u & 0x80000000
    if e == 0x7F800000:
        return (u & 0x80000000) | 0x7F7FFFFF
    return u & M32


def ps2f(x):
    return fl(ps2f_bits(bits(x)))


def rtz(q):
    """An exact rational rounded toward zero to single precision, as the
    PS2's FPU and VUs round (and the runtime sets the host to): no
    denormals (a result below 2^-126 is a signed zero), overflow saturates
    to the largest finite value."""
    if q == 0:
        return 0
    sign = 0x80000000 if q < 0 else 0
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if (Fraction(2) ** e) > a:
        e -= 1
    if e > 127:
        return sign | 0x7F7FFFFF
    if e < -126:
        return sign
    mant = math.floor(a / (Fraction(2) ** (e - 23)))
    return sign | ((e + 127) << 23) | (mant - (1 << 23))


def rtz_sqrt(q):
    """sqrt of a non-negative exact rational, rounded toward zero."""
    if q == 0:
        return 0
    e = (q.numerator.bit_length() - q.denominator.bit_length()) // 2
    k = 23 - e + 2                      # scale so that the root has 25+ significant bits
    scaled = q * (Fraction(4) ** k) if k >= 0 else q / (Fraction(4) ** -k)
    root = math.isqrt(math.floor(scaled))
    return rtz(Fraction(root) / (Fraction(2) ** k) if k >= 0 else Fraction(root) * (Fraction(2) ** -k))


def F(u):
    """The exact value of a single after the PS2 model's input rule."""
    return Fraction(ps2f(fl(u)))


def f32(x):
    """A host number (int or float) converted to single precision, toward zero."""
    if isinstance(x, float) and math.isnan(x):
        return ps2f_bits(0x7FC00000)
    if isinstance(x, float) and math.isinf(x):
        return 0x7F7FFFFF | (0x80000000 if x < 0 else 0)
    return rtz(Fraction(x))


# An exact zero has no sign as a rational: the IEEE rules give it one. A
# product or quotient takes the sign of a ^ b; a sum is +0 (round toward
# zero) unless both addends are zeros of the same sign.
def _sign(u):
    return ps2f_bits(u) & 0x80000000


def _is_zero(u):
    return ps2f_bits(u) & 0x7FFFFFFF == 0


def fadd(a, b):
    s = F(a) + F(b)
    if s == 0:
        return _sign(a) if _is_zero(a) and _is_zero(b) and _sign(a) == _sign(b) else 0
    return rtz(s)


def fsub(a, b):
    return fadd(a, (b & M32) ^ 0x80000000)


def fmul(a, b):
    p = F(a) * F(b)
    if p == 0:
        return _sign(a) ^ _sign(b)
    return rtz(p)


def fmax(a, b):
    x, y = ps2f(fl(a)), ps2f(fl(b))
    return bits(x if x > y else y)


def fmin(a, b):
    x, y = ps2f(fl(a)), ps2f(fl(b))
    return bits(x if x < y else y)


def fdiv(a, b):
    x, y = ps2f(fl(a)), ps2f(fl(b))
    if y == 0.0:
        return ((bits(x) ^ bits(y)) & 0x80000000) | 0x7F7FFFFF
    if x == 0.0:
        return (bits(x) ^ bits(y)) & 0x80000000
    return rtz(Fraction(x) / Fraction(y))


def fsqrt(a):
    return rtz_sqrt(abs(F(a)))


def frsqrt(a, b):
    return fdiv(a, fsqrt(b))


def trunc32(x):
    if x >= 2147483648.0:
        return 0x7FFFFFFF
    if x <= -2147483648.0:
        return 0x80000000
    return int(x) & M32


def fcvtw(a):
    return trunc32(ps2f(fl(a)))


def vu_ftoi(a, scale):
    return trunc32(fl(fmul(a, bits(scale))))


# ---- the machine ---------------------------------------------------------------------

class EE:
    def __init__(self, elf):
        self.elf = elf
        self.mem = bytearray(RAM)
        self.spr = bytearray(0x4000)
        self.vu0 = bytearray(0x1000)
        for s in elf.segments:
            if s.filesz:
                self.mem[s.vaddr:s.vaddr + s.filesz] = elf.data[s.offset:s.offset + s.filesz]
        self.r = [0] * 32            # 128-bit
        self.hi = self.lo = 0        # 128-bit: low 64 the first pipeline, high 64 the second
        self.sa = 0
        self.f = [0] * 32            # bits
        self.acc = 0
        self.fcr31 = 0
        self.vf = [[0, 0, 0, 0] for _ in range(32)]
        self.vf[0] = [0, 0, 0, 0x3F800000]
        self.vacc = [0, 0, 0, 0]
        self.vi = [0] * 32
        self.cop0 = [0] * 32
        self.steps = 0
        self._dec = {}
        self.trace = False
        self.ops = collections.Counter()     # operations executed (coverage)

    # -- memory --
    def _where(self, a, n):
        seg = a >> 28
        if seg == 0x7:
            o = a & 0x3FFF
            return self.spr, o
        p = a & 0x1FFFFFFF
        if seg in (0x0, 0x2, 0x3, 0x8, 0xA) and p + n <= RAM:
            return self.mem, p
        raise Stop("hardware access %08X" % a)

    def rd(self, a, n):
        m, o = self._where(a & M32, n)
        return int.from_bytes(m[o:o + n], "little")

    def wr(self, a, n, v):
        m, o = self._where(a & M32, n)
        m[o:o + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

    # -- registers --
    def g(self, n):
        return self.r[n] & M64

    def gw(self, n):
        return self.r[n] & M32

    def gs(self, n):
        return sx(self.r[n], 64)

    def set64(self, n, v):
        if n:
            self.r[n] = (self.r[n] & ~M64) | (v & M64)

    def set128(self, n, v):
        if n:
            self.r[n] = v & M128

    # -- running --
    def call(self, addr, args=(), max_steps=50_000_000):
        for k, v in enumerate(args):
            self.set64(4 + k, v)
        self.set64(31, s32(STOP))
        self.run(addr, max_steps)

    def decode(self, pc):
        i = self._dec.get(pc)
        if i is None:
            i = self._dec[pc] = ee.decode(self.rd(pc, 4), pc)
        return i

    def run(self, pc, max_steps):
        while pc != STOP:
            self.steps += 1
            if self.steps > max_steps:
                raise Stop("step limit at %08X" % pc)
            i = self.decode(pc)
            if self.trace:
                print("%08X  %s" % (pc, ee.disasm(i)))
            if i.has_delay_slot:
                self.ops[i.op] += 1
                taken, target = self.branch(i)
                if i.is_likely and not taken:
                    pc += 8
                    continue
                self.execute(self.decode(pc + 4))
                pc = (target & M32) if taken else pc + 8
            else:
                self.execute(i)
                pc += 4

    def branch(self, i):
        """(taken, target), applying the link register before the delay slot."""
        op = i.op
        if op in ("j", "jal"):
            if op == "jal":
                self.set64(31, s32(i.va + 8))
            return True, i.target
        if op == "jr":
            return True, self.gw(i.rs)
        if op == "jalr":
            t = self.gw(i.rs)
            self.set64(i.rd, s32(i.va + 8))
            return True, t
        base = op[:-1] if i.is_likely else op
        link = base.endswith("al")
        if link:
            base = base[:-2]
        s, t = self.gs(i.rs), self.gs(i.rt)
        cond = {"beq": s == t, "bne": s != t, "blez": s <= 0, "bgtz": s > 0, "bltz": s < 0,
                "bgez": s >= 0, "bc1f": not self.fcr31 & (1 << 23),
                "bc1t": bool(self.fcr31 & (1 << 23))}.get(base)
        if cond is None:
            raise Stop("branch %s" % op)
        if link:
            self.set64(31, s32(i.va + 8))
        return cond, i.target

    # -- the instructions --
    def execute(self, i):
        op = i.op
        self.ops[op] += 1
        h = _OPS.get(op)
        if h is None:
            if op.startswith("v") and i.form.startswith("vu_"):
                return self.vu(i)
            raise Stop("not modelled: %s at %08X" % (op, i.va))
        h(self, i)

    def ea(self, i):
        return (self.gw(i.rs) + i.imm) & M32

    # VU0 macro mode
    def vu(self, i):
        name, form = i.op[1:], i.extra
        dest = i.word >> 21 & 15
        lanes = [k for k in range(4) if dest & (8 >> k)]
        ft, fs, fd, bc = i.rt, i.rd, i.sa, i.word & 3
        S, T, A = list(self.vf[fs]), list(self.vf[ft]), list(self.vacc)
        if name == "nop" or name == "waitq":
            return
        if form == "fs_ftw":
            w = abs(ps2f(fl(T[3])))
            f = 0
            for k, (hi_, lo_) in enumerate(((1, 2), (4, 8), (16, 32))):
                x = ps2f(fl(S[k]))
                if x > w:
                    f |= hi_
                if x < -w:
                    f |= lo_
            self.vi[18] = ((self.vi[18] << 6) | f) & 0xFFFFFF
            return
        if form == "ft_fs" and i.form == "vu_upper":
            res = list(self.vf[ft])
            for k in lanes:
                if name == "abs":
                    res[k] = S[k] & 0x7FFFFFFF
                elif name.startswith("itof"):
                    res[k] = f32(float(sx(S[k], 32)) / (1 << int(name[4:])))
                elif name.startswith("ftoi"):
                    res[k] = vu_ftoi(S[k], float(1 << int(name[4:])))
            if ft:
                self.vf[ft] = res
            return
        if form in ("fd_fs_ftbc", "fd_fs_q", "fd_fs_i", "fd_fs_ft",
                    "acc_fs_ftbc", "acc_fs_q", "acc_fs_i", "acc_fs_ft"):
            to_acc = form.startswith("acc")
            if form.endswith("ftbc"):
                base, B = name[:-1], [T[bc]] * 4
            elif form.endswith("_q"):
                base, B = name[:-1], [self.vi[22]] * 4
            elif form.endswith("_i"):
                base, B = name[:-1], [self.vi[21]] * 4
            else:
                base, B = name, T
            if to_acc and base.endswith("a") and base != "opmula":
                base = base[:-1]
            res = list(self.vacc) if to_acc else list(self.vf[fd])
            for k in lanes:
                if base in ("opmula", "opmsub"):
                    if k == 3:
                        continue
                    p, q = ((1, 2), (2, 0), (0, 1))[k]
                    prod = fmul(S[p], T[q])
                    res[k] = prod if base == "opmula" else fsub(A[k], prod)
                    continue
                res[k] = {"add": lambda: fadd(S[k], B[k]), "sub": lambda: fsub(S[k], B[k]),
                          "mul": lambda: fmul(S[k], B[k]), "max": lambda: fmax(S[k], B[k]),
                          "mini": lambda: fmin(S[k], B[k]),
                          "madd": lambda: fadd(A[k], fmul(S[k], B[k])),
                          "msub": lambda: fsub(A[k], fmul(S[k], B[k]))}[base]()
            if to_acc:
                self.vacc = res
            elif fd:
                self.vf[fd] = res
            mac = 0
            for k in lanes:
                b = 8 >> k
                if res[k] & 0x7FFFFFFF == 0:
                    mac |= b
                if res[k] & 0x80000000:
                    mac |= b << 4
            self.vi[17] = mac
            st = self.vi[16] & ~0xF
            if mac & 0x0F:
                st |= 0x041
            if mac & 0xF0:
                st |= 0x082
            self.vi[16] = st
            return
        it, is_, id_ = i.rt, i.rd, i.sa
        fsf, ftf = i.word >> 21 & 3, i.word >> 23 & 3

        def setvi(n, v):
            if n:
                self.vi[n] = v & 0xFFFF
        if name in ("move", "mr32"):
            src = (0, 1, 2, 3) if name == "move" else (1, 2, 3, 0)
            if ft:
                for k in lanes:
                    self.vf[ft][k] = S[src[k]]
        elif name in ("lqi", "lqd"):
            if name == "lqd":
                setvi(is_, self.vi[is_] - 1)
            o = (self.vi[is_] * 16) & 0xFFF
            if ft:
                for k in lanes:
                    self.vf[ft][k] = int.from_bytes(self.vu0[o + 4 * k:o + 4 * k + 4], "little")
            if name == "lqi":
                setvi(is_, self.vi[is_] + 1)
        elif name in ("sqi", "sqd"):
            if name == "sqd":
                setvi(it, self.vi[it] - 1)
            o = (self.vi[it] * 16) & 0xFFF
            for k in lanes:
                self.vu0[o + 4 * k:o + 4 * k + 4] = S[k].to_bytes(4, "little")
            if name == "sqi":
                setvi(it, self.vi[it] + 1)
        elif name == "div":
            self.vi[22] = fdiv(S[fsf], T[ftf])
        elif name == "sqrt":
            self.vi[22] = fsqrt(T[ftf])
        elif name == "rsqrt":
            self.vi[22] = frsqrt(S[fsf], T[ftf])
        elif name == "mtir":
            setvi(it, S[fsf])
        elif name == "mfir":
            if ft:
                for k in lanes:
                    self.vf[ft][k] = sx(self.vi[is_], 16) & M32
        elif name in ("iadd", "isub", "iand", "ior"):
            a, b = self.vi[is_], self.vi[it]
            setvi(id_, {"iadd": a + b, "isub": a - b, "iand": a & b, "ior": a | b}[name])
        elif name == "iaddi":
            setvi(it, self.vi[is_] + sx(id_, 5))
        else:
            raise Stop("not modelled: %s at %08X" % (i.op, i.va))


# ---- the operation table ----------------------------------------------------------------

def _op(*names):
    def deco(fn):
        for n in names:
            _OPS[n] = fn
        return fn
    return deco


_OPS = {}

for _n, _f in {
    "addiu": lambda e, i: s32(e.gw(i.rs) + i.imm), "addi": lambda e, i: s32(e.gw(i.rs) + i.imm),
    "daddiu": lambda e, i: e.g(i.rs) + i.imm, "daddi": lambda e, i: e.g(i.rs) + i.imm,
    "slti": lambda e, i: int(e.gs(i.rs) < i.imm), "sltiu": lambda e, i: int(e.g(i.rs) < (i.imm & M64)),
    "andi": lambda e, i: e.g(i.rs) & i.imm, "ori": lambda e, i: e.g(i.rs) | i.imm,
    "xori": lambda e, i: e.g(i.rs) ^ i.imm, "lui": lambda e, i: s32(i.imm << 16),
}.items():
    _OPS[_n] = (lambda f: lambda e, i: e.set64(i.rt, f(e, i)))(_f)

for _n, _f in {
    "addu": lambda e, i: s32(e.gw(i.rs) + e.gw(i.rt)), "add": lambda e, i: s32(e.gw(i.rs) + e.gw(i.rt)),
    "subu": lambda e, i: s32(e.gw(i.rs) - e.gw(i.rt)), "sub": lambda e, i: s32(e.gw(i.rs) - e.gw(i.rt)),
    "daddu": lambda e, i: e.g(i.rs) + e.g(i.rt), "dadd": lambda e, i: e.g(i.rs) + e.g(i.rt),
    "dsubu": lambda e, i: e.g(i.rs) - e.g(i.rt), "dsub": lambda e, i: e.g(i.rs) - e.g(i.rt),
    "and": lambda e, i: e.g(i.rs) & e.g(i.rt), "or": lambda e, i: e.g(i.rs) | e.g(i.rt),
    "xor": lambda e, i: e.g(i.rs) ^ e.g(i.rt), "nor": lambda e, i: ~(e.g(i.rs) | e.g(i.rt)),
    "slt": lambda e, i: int(e.gs(i.rs) < e.gs(i.rt)), "sltu": lambda e, i: int(e.g(i.rs) < e.g(i.rt)),
    "sllv": lambda e, i: s32(e.gw(i.rt) << (e.gw(i.rs) & 31)),
    "srlv": lambda e, i: s32(e.gw(i.rt) >> (e.gw(i.rs) & 31)),
    "srav": lambda e, i: s32(sx(e.gw(i.rt), 32) >> (e.gw(i.rs) & 31)),
    "dsllv": lambda e, i: e.g(i.rt) << (e.gw(i.rs) & 63),
    "dsrlv": lambda e, i: e.g(i.rt) >> (e.gw(i.rs) & 63),
    "dsrav": lambda e, i: e.gs(i.rt) >> (e.gw(i.rs) & 63),
    "sll": lambda e, i: s32(e.gw(i.rt) << i.sa), "srl": lambda e, i: s32(e.gw(i.rt) >> i.sa),
    "sra": lambda e, i: s32(sx(e.gw(i.rt), 32) >> i.sa),
    "dsll": lambda e, i: e.g(i.rt) << i.sa, "dsrl": lambda e, i: e.g(i.rt) >> i.sa,
    "dsra": lambda e, i: e.gs(i.rt) >> i.sa, "dsll32": lambda e, i: e.g(i.rt) << (i.sa + 32),
    "dsrl32": lambda e, i: e.g(i.rt) >> (i.sa + 32), "dsra32": lambda e, i: e.gs(i.rt) >> (i.sa + 32),
    "mfhi": lambda e, i: e.hi & M64, "mflo": lambda e, i: e.lo & M64,
    "mfhi1": lambda e, i: e.hi >> 64, "mflo1": lambda e, i: e.lo >> 64,
    "mfsa": lambda e, i: e.sa,
}.items():
    _OPS[_n] = (lambda f: lambda e, i: e.set64(i.rd, f(e, i)))(_f)


@_op("movz", "movn")
def _mov(e, i):
    if (e.g(i.rt) == 0) == (i.op == "movz"):
        e.set64(i.rd, e.g(i.rs))


def _pipe_set(e, pipe, hi, lo):
    sh = 64 * pipe
    e.hi = (e.hi & ~(M64 << sh)) | ((hi & M64) << sh)
    e.lo = (e.lo & ~(M64 << sh)) | ((lo & M64) << sh)


@_op("mult", "multu", "mult1", "multu1", "madd", "maddu", "madd1", "maddu1")
def _mult(e, i):
    op = i.op
    pipe = 1 if op.endswith("1") else 0
    signed = "u" not in op.replace("mult", "").replace("madd", "")
    a, b = (sx(e.gw(i.rs), 32), sx(e.gw(i.rt), 32)) if signed else (e.gw(i.rs), e.gw(i.rt))
    p = a * b
    if op.startswith("madd"):
        sh = 64 * pipe
        acc = (((e.hi >> sh) & M32) << 32) | ((e.lo >> sh) & M32)
        p = acc + p
    _pipe_set(e, pipe, s32(p >> 32), s32(p))
    e.set64(i.rd, s32(p))


@_op("div", "div1", "divu", "divu1")
def _div(e, i):
    pipe = 1 if i.op.endswith("1") else 0
    if i.op.startswith("divu"):
        a, b = e.gw(i.rs), e.gw(i.rt)
        if b == 0:
            return _pipe_set(e, pipe, s32(a), s32(M32))
        return _pipe_set(e, pipe, s32(a % b), s32(a // b))
    a, b = sx(e.gw(i.rs), 32), sx(e.gw(i.rt), 32)
    if b == 0:
        return _pipe_set(e, pipe, s32(a), s32(1 if a < 0 else -1))
    if a == -0x80000000 and b == -1:
        return _pipe_set(e, pipe, 0, s32(a))
    q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)       # truncation toward zero
    return _pipe_set(e, pipe, s32(a - q * b), s32(q))


@_op("mthi", "mtlo", "mthi1", "mtlo1")
def _mthl(e, i):
    sh = 64 if i.op.endswith("1") else 0
    name = i.op[2:4]
    cur = getattr(e, name)
    setattr(e, name, (cur & ~(M64 << sh)) | (e.g(i.rs) << sh))


@_op("mtsa")
def _mtsa(e, i):
    e.sa = e.gw(i.rs)


@_op("mtsab")
def _mtsab(e, i):
    e.sa = ((e.gw(i.rs) ^ i.imm) & 15) * 8


@_op("mtsah")
def _mtsah(e, i):
    e.sa = ((e.gw(i.rs) ^ i.imm) & 7) * 16


_LOADS = {"lb": (1, True), "lbu": (1, False), "lh": (2, True), "lhu": (2, False),
          "lw": (4, True), "lwu": (4, False), "ld": (8, False)}


@_op(*_LOADS)
def _load(e, i):
    n, signed = _LOADS[i.op]
    v = e.rd(e.ea(i), n)
    e.set64(i.rt, (sx(v, 8 * n) & M64) if signed else v)


_STORES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}


@_op(*_STORES)
def _store(e, i):
    e.wr(e.ea(i), _STORES[i.op], e.g(i.rt))


@_op("lq")
def _lq(e, i):
    v = e.rd(e.ea(i) & ~15, 16)
    e.set128(i.rt, v)


@_op("sq")
def _sq(e, i):
    e.wr(e.ea(i) & ~15, 16, e.r[i.rt])


@_op("lwl", "lwr", "ldl", "ldr")
def _lmerge(e, i):
    a = e.ea(i)
    if i.op in ("lwl", "lwr"):
        sh = (a & 3) * 8
        w = e.rd(a & ~3, 4)
        rt = e.gw(i.rt)
        if i.op == "lwl":
            keep = 0 if sh == 24 else (0x00FFFFFF >> sh)
            e.set64(i.rt, s32((rt & keep) | (w << (24 - sh))))
        elif sh == 0:
            e.set64(i.rt, s32(w))
        else:
            keep = (0xFFFFFF00 << (24 - sh)) & M32
            e.set64(i.rt, (e.g(i.rt) & 0xFFFFFFFF00000000) | ((rt & keep) | (w >> sh)))
    else:
        sh = (a & 7) * 8
        w = e.rd(a & ~7, 8)
        rt = e.g(i.rt)
        if i.op == "ldl":
            keep = 0 if sh == 56 else (0x00FFFFFFFFFFFFFF >> sh)
            e.set64(i.rt, (rt & keep) | ((w << (56 - sh)) & M64))
        else:
            keep = 0 if sh == 0 else ((0xFFFFFFFFFFFFFF00 << (56 - sh)) & M64)
            e.set64(i.rt, (rt & keep) | (w >> sh))


@_op("swl", "swr", "sdl", "sdr")
def _smerge(e, i):
    a = e.ea(i)
    if i.op in ("swl", "swr"):
        sh = (a & 3) * 8
        w = e.rd(a & ~3, 4)
        rt = e.gw(i.rt)
        if i.op == "swl":
            keep = 0 if sh == 24 else ((0xFFFFFF00 << sh) & M32)
            e.wr(a & ~3, 4, (w & keep) | (rt >> (24 - sh)))
        else:
            keep = 0 if sh == 0 else (0x00FFFFFF >> (24 - sh))
            e.wr(a & ~3, 4, (w & keep) | ((rt << sh) & M32))
    else:
        sh = (a & 7) * 8
        w = e.rd(a & ~7, 8)
        rt = e.g(i.rt)
        if i.op == "sdl":
            keep = 0 if sh == 56 else ((0xFFFFFFFFFFFFFF00 << sh) & M64)
            e.wr(a & ~7, 8, (w & keep) | (rt >> (56 - sh)))
        else:
            keep = 0 if sh == 0 else (0x00FFFFFFFFFFFFFF >> (56 - sh))
            e.wr(a & ~7, 8, (w & keep) | ((rt << sh) & M64))


@_op("sync.l", "sync.p", "cache", "pref", "vnop", "vwaitq")
def _nop(e, i):
    pass


@_op("syscall", "break", "ei", "di", "mtc0", "mfc0", "vcallms", "vcallmsr")
def _system(e, i):
    raise Stop("%s at %08X" % (i.op, i.va))


# FPU
@_op("lwc1")
def _lwc1(e, i):
    e.f[i.rt] = e.rd(e.ea(i), 4)


@_op("swc1")
def _swc1(e, i):
    e.wr(e.ea(i), 4, e.f[i.rt])


@_op("mfc1")
def _mfc1(e, i):
    e.set64(i.rt, s32(e.f[i.rd]))


@_op("mtc1")
def _mtc1(e, i):
    e.f[i.rd] = e.gw(i.rt)


@_op("cfc1")
def _cfc1(e, i):
    e.set64(i.rt, s32(e.fcr31) if i.rd == 31 else 0x2E00)


@_op("ctc1")
def _ctc1(e, i):
    if i.rd == 31:
        e.fcr31 = e.gw(i.rt)


_F2 = {"add.s": fadd, "sub.s": fsub, "mul.s": fmul, "div.s": fdiv, "max.s": fmax, "min.s": fmin,
       "rsqrt.s": frsqrt}


@_op(*_F2)
def _f2(e, i):
    e.f[i.sa] = _F2[i.op](e.f[i.rd], e.f[i.rt])


@_op("adda.s", "suba.s", "mula.s", "madda.s", "msuba.s", "madd.s", "msub.s")
def _facc(e, i):
    fs, ft = e.f[i.rd], e.f[i.rt]
    v = {"adda.s": lambda: fadd(fs, ft), "suba.s": lambda: fsub(fs, ft), "mula.s": lambda: fmul(fs, ft),
         "madda.s": lambda: fadd(e.acc, fmul(fs, ft)), "msuba.s": lambda: fsub(e.acc, fmul(fs, ft)),
         "madd.s": lambda: fadd(e.acc, fmul(fs, ft)), "msub.s": lambda: fsub(e.acc, fmul(fs, ft))}[i.op]()
    if i.op in ("madd.s", "msub.s"):
        e.f[i.sa] = v
    else:
        e.acc = v


@_op("abs.s", "neg.s", "mov.s", "sqrt.s", "cvt.w.s", "cvt.s.w")
def _f1(e, i):
    fs = e.f[i.rd]
    e.f[i.sa] = {"abs.s": lambda: fs & 0x7FFFFFFF, "neg.s": lambda: fs ^ 0x80000000, "mov.s": lambda: fs,
                 "sqrt.s": lambda: fsqrt(e.f[i.rt]), "cvt.w.s": lambda: fcvtw(fs),
                 "cvt.s.w": lambda: f32(float(sx(fs, 32)))}[i.op]()


@_op("c.f.s", "c.eq.s", "c.lt.s", "c.le.s")
def _fcmp(e, i):
    a, b = ps2f(fl(e.f[i.rd])), ps2f(fl(e.f[i.rt]))
    v = {"c.f.s": False, "c.eq.s": a == b, "c.lt.s": a < b, "c.le.s": a <= b}[i.op]
    e.fcr31 = (e.fcr31 | (1 << 23)) if v else (e.fcr31 & ~(1 << 23))


# COP2 transfers
@_op("lqc2")
def _lqc2(e, i):
    v = e.rd(e.ea(i) & ~15, 16)
    if i.rt:
        e.vf[i.rt] = [(v >> (32 * k)) & M32 for k in range(4)]


@_op("sqc2")
def _sqc2(e, i):
    e.wr(e.ea(i) & ~15, 16, sum(x << (32 * k) for k, x in enumerate(e.vf[i.rt])))


@_op("qmfc2.i", "qmfc2.ni")
def _qmfc2(e, i):
    e.set128(i.rt, sum(x << (32 * k) for k, x in enumerate(e.vf[i.rd])))


@_op("qmtc2.i", "qmtc2.ni")
def _qmtc2(e, i):
    if i.rd:
        e.vf[i.rd] = [(e.r[i.rt] >> (32 * k)) & M32 for k in range(4)]


@_op("cfc2.i", "cfc2.ni")
def _cfc2(e, i):
    if i.rd not in set(range(16)) | {16, 17, 18, 20, 21, 22}:
        raise Stop("cfc2 vi%d at %08X" % (i.rd, i.va))
    e.set64(i.rt, s32(e.vi[i.rd]))


@_op("ctc2.i", "ctc2.ni")
def _ctc2(e, i):
    if i.rd == 0:
        return
    if i.rd < 16:
        e.vi[i.rd] = e.gw(i.rt) & 0xFFFF
    elif i.rd in (16, 17, 18, 20, 21, 22):
        e.vi[i.rd] = e.gw(i.rt)
    else:
        raise Stop("ctc2 vi%d at %08X" % (i.rd, i.va))


# MMI
def _lanes(v, n):
    w = 128 // n
    return [(v >> (w * k)) & ((1 << w) - 1) for k in range(n)]


def _join(vals, n):
    w = 128 // n
    return sum((x & ((1 << w) - 1)) << (w * k) for k, x in enumerate(vals))


def _mmi(fn):
    def h(e, i):
        e.set128(i.rd, fn(e.r[i.rs], e.r[i.rt], i, e))
    return h


def _pw(n, f):
    return _mmi(lambda s, t, i, e: _join([f(a, b) for a, b in zip(_lanes(s, n), _lanes(t, n))], n))


def _sgn(v, bitsn):
    return sx(v, bitsn)


_MMIOPS = {
    "por": _mmi(lambda s, t, i, e: s | t), "pand": _mmi(lambda s, t, i, e: s & t),
    "pxor": _mmi(lambda s, t, i, e: s ^ t), "pnor": _mmi(lambda s, t, i, e: ~(s | t)),
    "paddw": _pw(4, lambda a, b: a + b), "psubw": _pw(4, lambda a, b: a - b),
    "paddh": _pw(8, lambda a, b: a + b), "psubh": _pw(8, lambda a, b: a - b),
    "paddb": _pw(16, lambda a, b: a + b), "psubb": _pw(16, lambda a, b: a - b),
    "pcgtw": _pw(4, lambda a, b: M32 if _sgn(a, 32) > _sgn(b, 32) else 0),
    "pcgth": _pw(8, lambda a, b: 0xFFFF if _sgn(a, 16) > _sgn(b, 16) else 0),
    "pcgtb": _pw(16, lambda a, b: 0xFF if _sgn(a, 8) > _sgn(b, 8) else 0),
    "pmaxh": _pw(8, lambda a, b: a if _sgn(a, 16) > _sgn(b, 16) else b),
    "pminh": _pw(8, lambda a, b: a if _sgn(a, 16) < _sgn(b, 16) else b),
    "pmaxw": _pw(4, lambda a, b: a if _sgn(a, 32) > _sgn(b, 32) else b),
    "pminw": _pw(4, lambda a, b: a if _sgn(a, 32) < _sgn(b, 32) else b),
    "pcpyld": _mmi(lambda s, t, i, e: (t & M64) | ((s & M64) << 64)),
    "pcpyud": _mmi(lambda s, t, i, e: (s >> 64) | ((t >> 64) << 64)),
    "pextlw": _mmi(lambda s, t, i, e: _join([_lanes(t, 4)[0], _lanes(s, 4)[0], _lanes(t, 4)[1], _lanes(s, 4)[1]], 4)),
    "pextuw": _mmi(lambda s, t, i, e: _join([_lanes(t, 4)[2], _lanes(s, 4)[2], _lanes(t, 4)[3], _lanes(s, 4)[3]], 4)),
    "pextlh": _mmi(lambda s, t, i, e: _join([x for k in range(4) for x in (_lanes(t, 8)[k], _lanes(s, 8)[k])], 8)),
    "pextuh": _mmi(lambda s, t, i, e: _join([x for k in range(4, 8) for x in (_lanes(t, 8)[k], _lanes(s, 8)[k])], 8)),
    "pextlb": _mmi(lambda s, t, i, e: _join([x for k in range(8) for x in (_lanes(t, 16)[k], _lanes(s, 16)[k])], 16)),
    "pextub": _mmi(lambda s, t, i, e: _join([x for k in range(8, 16) for x in (_lanes(t, 16)[k], _lanes(s, 16)[k])], 16)),
    "ppacb": _mmi(lambda s, t, i, e: _join(_lanes(t, 16)[0::2] + _lanes(s, 16)[0::2], 16)),
    "ppach": _mmi(lambda s, t, i, e: _join(_lanes(t, 8)[0::2] + _lanes(s, 8)[0::2], 8)),
    "ppacw": _mmi(lambda s, t, i, e: _join(_lanes(t, 4)[0::2] + _lanes(s, 4)[0::2], 4)),
    "pcpyh": _mmi(lambda s, t, i, e: _join([_lanes(t, 8)[0]] * 4 + [_lanes(t, 8)[4]] * 4, 8)),
    "prot3w": _mmi(lambda s, t, i, e: _join([_lanes(t, 4)[k] for k in (1, 2, 0, 3)], 4)),
    "pexcw": _mmi(lambda s, t, i, e: _join([_lanes(t, 4)[k] for k in (0, 2, 1, 3)], 4)),
    "psllh": _mmi(lambda s, t, i, e: _join([x << (i.sa & 15) for x in _lanes(t, 8)], 8)),
    "psrlh": _mmi(lambda s, t, i, e: _join([x >> (i.sa & 15) for x in _lanes(t, 8)], 8)),
    "psrah": _mmi(lambda s, t, i, e: _join([_sgn(x, 16) >> (i.sa & 15) for x in _lanes(t, 8)], 8)),
    "psllw": _mmi(lambda s, t, i, e: _join([x << i.sa for x in _lanes(t, 4)], 4)),
    "psrlw": _mmi(lambda s, t, i, e: _join([x >> i.sa for x in _lanes(t, 4)], 4)),
    "psraw": _mmi(lambda s, t, i, e: _join([_sgn(x, 32) >> i.sa for x in _lanes(t, 4)], 4)),
    "qfsrv": _mmi(lambda s, t, i, e: ((((s & M128) << 128) | (t & M128)) >> (((e.sa >> 3) & 15) * 8)) & M128),
}
_OPS.update(_MMIOPS)


def main():
    from .elf import Elf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--call", required=True)
    for k in range(4):
        ap.add_argument("--a%d" % k, default=None)
    ap.add_argument("--sp", default="0x01F00000")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    m = EE(Elf(a.elf))
    m.trace = a.trace
    m.set64(29, int(a.sp, 0))
    args = [int(getattr(a, "a%d" % k), 0) for k in range(4) if getattr(a, "a%d" % k) is not None]
    m.call(int(a.call, 0), args)
    print("v0 %016X v1 %016X  %d steps" % (m.g(2), m.g(3), m.steps))


if __name__ == "__main__":
    main()

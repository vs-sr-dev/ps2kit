"""Differential tests: recompiled C++ against ps2kit.eeemu, function by function.

    python -m ps2kit.recomp.difftest GAME.ELF --driver build/recomp-build/difftest.exe
                                     [--only memcpy,strlen] [--seeds 20]

Each case calls one of the executable's functions on a machine state both
sides start from: every register filled with pseudo-random 128-bit values
(so that a translation that loses the upper halves shows), sp and gp set,
the arguments and their buffers in place, RAM as the ELF loads it. The
reference interpreter runs the call in Python; the driver
(ps2kit/runtime/difftest.cpp, built with the recompiled code) runs the
recompiled function. Every register and every byte of RAM, scratchpad and
VU0 memory must then be equal.

The cases are the functions every PS2 game links: newlib's R5900 string
routines, libgcc's 64-bit arithmetic and the SDK's soft float (fp-bit/dp-bit,
whose functions it renames: dpadd, fptosi, ...), libc's sprintf, rand
and qsort, the SDK's libvu0. A game's own functions can be added by a port
(cases are plain data: name, arguments, buffers).
"""
import argparse
import os
import random
import struct
import subprocess
import tempfile

from .. import eeemu
from ..elf import Elf

# The MinGW runtime DLLs the driver needs: MSYS2's, unless PS2KIT_MINGW_BIN says otherwise.
MINGW_BIN = os.environ.get("PS2KIT_MINGW_BIN", r"C:\msys64\mingw64\bin")

STACK = 0x01F00000
BUF = 0x01800000                  # test buffers
M32 = 0xFFFFFFFF


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def d2u(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# --- the cases -------------------------------------------------------------------------
# A case is (function name, builder). The builder gets a random.Random and
# returns (args, {address: bytes}) or (args, {address: bytes}, fargs): args
# are 64-bit values for a0-a3 and t0-t3 (the EE ABI's eight integer argument
# registers), fargs float bits for $f12, $f13, ...

def _str(rng, n):
    return bytes(rng.randrange(1, 256) for _ in range(n)) + b"\0"


def c_memcpy(rng):
    n = rng.choice([0, 1, 3, 15, 16, 17, 64, 255, 1000])
    so, do = rng.randrange(16), rng.randrange(16)
    return [BUF + 0x1000 + do, BUF + so, n], {BUF: bytes(rng.randrange(256) for _ in range(1100))}


def c_memmove(rng):
    n = rng.choice([1, 7, 16, 33, 200])
    off = rng.choice([-9, -1, 1, 5, 16])
    return [BUF + 64 + off, BUF + 64, n], {BUF: bytes(rng.randrange(256) for _ in range(400))}


def c_memset(rng):
    n = rng.choice([0, 1, 5, 16, 31, 100, 513])
    return [BUF + rng.randrange(16), rng.randrange(256), n], {BUF: bytes(600)}


def c_memchr(rng):
    data = bytes(rng.randrange(256) for _ in range(200))
    return [BUF + rng.randrange(8), data[rng.randrange(150)], rng.choice([10, 100, 190])], {BUF: data}


def c_strlen(rng):
    return [BUF + rng.randrange(16)], {BUF: b"x" * 16 + _str(rng, rng.choice([0, 1, 7, 16, 40, 300]))}


def c_strcmp(rng):
    a = _str(rng, rng.choice([0, 3, 17, 40]))
    b = a if rng.random() < 0.5 else _str(rng, rng.choice([0, 3, 17, 40]))
    if rng.random() < 0.3 and len(a) > 3:
        b = a[:-3] + b"\x01\x00"
    return [BUF + rng.randrange(8), BUF + 0x200 + rng.randrange(8)], {BUF: b"\0" * 8 + a, BUF + 0x200: b"\0" * 8 + b}


def c_strcpy(rng):
    return [BUF + 0x400 + rng.randrange(16), BUF + rng.randrange(16)], {BUF: b"q" * 16 + _str(rng, rng.choice([0, 5, 16, 77]))}


def c_strncpy(rng):
    return [BUF + 0x400 + rng.randrange(8), BUF + rng.randrange(8), rng.choice([0, 3, 20, 100])], \
           {BUF: b"q" * 8 + _str(rng, rng.choice([0, 5, 16, 77]))}


def c_strcat(rng):
    return [BUF + 0x400, BUF], {BUF: _str(rng, rng.choice([0, 9, 30])), BUF + 0x400: _str(rng, rng.choice([0, 4, 21]))}


def _i64(rng):
    return rng.choice([0, 1, -1, 7, -7, 1 << 40, -(1 << 63), (1 << 63) - 1, rng.randrange(-(1 << 63), 1 << 63),
                       rng.randrange(-1000, 1000)]) & ((1 << 64) - 1)


def c_int64(rng):
    a, b = _i64(rng), _i64(rng)
    if b == 0:
        b = 3
    return [a, b], {}


def _f(rng):
    return rng.choice([0.0, -0.0, 1.0, -2.5, 1e-30, 3.4e38, 1e10, rng.uniform(-1e6, 1e6), rng.uniform(-1, 1)])


def c_sf2(rng):
    a, b = f2u(_f(rng)), f2u(_f(rng))
    return [a, b], {}, [a, b]


def c_sf1(rng):
    a = f2u(_f(rng))
    return [a], {}, [a]


def c_df2(rng):
    return [d2u(_f(rng) * 1e3), d2u(_f(rng))], {}


def c_df1(rng):
    return [d2u(_f(rng) * 7)], {}


def c_si(rng):
    return [rng.choice([0, 1, -1, 123456789, -(1 << 31), rng.randrange(-(1 << 31), 1 << 31)]) & M32], {}


def c_sprintf(rng):
    fmt, kinds = rng.choice([(b"%d|%x|%s|%c\0", "nnsn"), (b"%08X %-5d %u\0", "nnn"),
                             (b"[%5s][%.2s]%%\0", "ss")])
    s = _str(rng, 6).replace(b"%", b"p")
    args = [BUF + 0x800, BUF] + [BUF + 0x100 if k == "s" else rng.randrange(-99999, 99999) & M32
                                 for k in kinds]
    return args, {BUF: fmt, BUF + 0x100: s}


def c_vu0_camera(rng):    # (dst, position, z direction, y direction)
    return [BUF + 0x200, BUF, BUF + 0x40, BUF + 0x80], {BUF: _vec(rng), BUF + 0x40: _vec(rng), BUF + 0x80: _vec(rng)}


def c_rand(rng):
    return [], {}


def _vec(rng):
    return b"".join(struct.pack("<f", rng.uniform(-10, 10)) for _ in range(4))


def _mat(rng):
    return b"".join(_vec(rng) for _ in range(4))


def c_vu0_mm(rng):        # (dst, a, b) matrices or vectors
    return [BUF + 0x200, BUF, BUF + 0x100], {BUF: _mat(rng), BUF + 0x100: _mat(rng)}


def c_vu0_m(rng):         # (dst, src)
    return [BUF + 0x200, BUF], {BUF: _mat(rng)}


def c_vu0_rot(rng):       # (dst, src, float angle)
    return [BUF + 0x200, BUF], {BUF: _mat(rng)}, [f2u(rng.uniform(-3.2, 3.2))]


def c_vu0_sv(rng):        # (dst, src, float scale)
    return [BUF + 0x200, BUF], {BUF: _vec(rng)}, [f2u(rng.uniform(-5, 5))]


def c_vu0_inter(rng):     # (dst, v0, v1, float t)
    return [BUF + 0x200, BUF, BUF + 0x100], {BUF: _vec(rng), BUF + 0x100: _vec(rng)}, [f2u(rng.uniform(0, 1))]


def c_vu0_clamp(rng):     # (dst, src, float min, float max)
    return [BUF + 0x200, BUF], {BUF: _vec(rng)}, [f2u(rng.uniform(-5, 0)), f2u(rng.uniform(0, 5))]


CASES = [
    ("memcpy", c_memcpy), ("memmove", c_memmove), ("memset", c_memset), ("memchr", c_memchr),
    ("strlen", c_strlen), ("strcmp", c_strcmp), ("strcpy", c_strcpy), ("strncpy", c_strncpy),
    ("strcat", c_strcat),
    ("__divdi3", c_int64), ("__moddi3", c_int64), ("__udivdi3", c_int64), ("__umoddi3", c_int64),
    ("__muldi3", c_int64),
    ("fpadd", c_sf2), ("fpsub", c_sf2), ("fpmul", c_sf2), ("fpdiv", c_sf2), ("fpcmp", c_sf2),
    ("fptosi", c_sf1), ("fptoui", c_sf1), ("fptodp", c_sf1), ("sitofp", c_si),
    ("dpadd", c_df2), ("dpsub", c_df2), ("dpmul", c_df2), ("dpdiv", c_df2), ("dpcmp", c_df2),
    ("dptoli", c_df1), ("dptoul", c_df1), ("dptofp", c_df1), ("litodp", c_int64),
    ("sprintf", c_sprintf), ("rand", c_rand),
    ("sceVu0MulMatrix", c_vu0_mm), ("sceVu0ApplyMatrix", c_vu0_mm), ("sceVu0OuterProduct", c_vu0_mm),
    ("sceVu0InnerProduct", c_vu0_mm), ("sceVu0AddVector", c_vu0_mm), ("sceVu0SubVector", c_vu0_mm),
    ("sceVu0MulVector", c_vu0_mm), ("sceVu0InterVector", c_vu0_inter),
    ("sceVu0InversMatrix", c_vu0_m), ("sceVu0TransposeMatrix", c_vu0_m), ("sceVu0Normalize", c_vu0_m),
    ("sceVu0CopyMatrix", c_vu0_m), ("sceVu0UnitMatrix", c_vu0_m), ("sceVu0ClampVector", c_vu0_clamp),
    ("sceVu0RotMatrixX", c_vu0_rot), ("sceVu0RotMatrixY", c_vu0_rot), ("sceVu0RotMatrixZ", c_vu0_rot),
    ("sceVu0RotMatrix", c_vu0_mm), ("sceVu0TransMatrix", c_vu0_mm), ("sceVu0ScaleVector", c_vu0_sv),
    ("sceVu0DivVector", c_vu0_sv), ("sceVu0CameraMatrix", c_vu0_camera),
]


# --- running ----------------------------------------------------------------------------

def _initial(elf, base, rng, args, mem, gp, fargs=()):
    m = eeemu.EE.__new__(eeemu.EE)
    m.__dict__.update(base.__dict__)
    m.mem = bytearray(base.mem)
    m.spr = bytearray(base.spr)
    m.vu0 = bytearray(base.vu0)
    m._dec = base._dec
    m.r = [0] + [rng.getrandbits(128) for _ in range(31)]
    m.hi, m.lo, m.sa = rng.getrandbits(128), rng.getrandbits(128), 0
    m.f = [eeemu.f32(rng.uniform(-100, 100)) for _ in range(32)]
    m.acc = eeemu.f32(rng.uniform(-100, 100))
    m.fcr31 = 0
    m.vf = [[0, 0, 0, 0x3F800000]] + [[eeemu.f32(rng.uniform(-100, 100)) for _ in range(4)] for _ in range(31)]
    m.vacc = [0, 0, 0, 0]
    m.vi = [rng.getrandbits(16) for _ in range(16)] + [0] * 16
    m.vi[0] = 0
    m.cop0 = [0] * 32
    m.steps = 0
    m.set64(29, STACK)
    m.set64(28, gp)
    m.set64(31, eeemu.s32(eeemu.STOP))
    for k, v in enumerate(args):
        m.set64(4 + k, v)
    for k, v in enumerate(fargs):
        m.f[12 + k] = v
    for a, b in mem.items():
        m.mem[a:a + len(b)] = b
    return m


def _pack(m, addr):
    out = [struct.pack("<II", 0x54444545, addr)]
    out += [r.to_bytes(16, "little") for r in m.r]
    out += [m.hi.to_bytes(16, "little"), m.lo.to_bytes(16, "little"), struct.pack("<I", m.sa)]
    out += [struct.pack("<I", x) for x in m.f]
    out += [struct.pack("<II", m.acc, m.fcr31)]
    out += [struct.pack("<4I", *v) for v in m.vf]
    out += [struct.pack("<4I", *m.vacc)]
    out += [struct.pack("<I", x & M32) for x in m.vi]
    out += [bytes(m.mem), bytes(m.spr), bytes(m.vu0)]
    return b"".join(out)


def _unpack(data):
    o = 8
    s = {}
    s["r"] = [int.from_bytes(data[o + 16 * k:o + 16 * k + 16], "little") for k in range(32)]
    o += 512
    s["hi"], s["lo"] = int.from_bytes(data[o:o + 16], "little"), int.from_bytes(data[o + 16:o + 32], "little")
    o += 32
    s["sa"], = struct.unpack_from("<I", data, o)
    o += 4
    s["f"] = list(struct.unpack_from("<32I", data, o))
    o += 128
    s["acc"], s["fcr31"] = struct.unpack_from("<II", data, o)
    o += 8
    s["vf"] = [list(struct.unpack_from("<4I", data, o + 16 * k)) for k in range(32)]
    o += 512
    s["vacc"] = list(struct.unpack_from("<4I", data, o))
    o += 16
    s["vi"] = list(struct.unpack_from("<32I", data, o))
    o += 128
    s["mem"] = data[o:o + eeemu.RAM]
    o += eeemu.RAM
    s["spr"], s["vu0"] = data[o:o + 0x4000], data[o + 0x4000:o + 0x5000]
    return s


def _compare(m, s):
    diffs = []
    for k in range(32):
        if m.r[k] != s["r"][k]:
            diffs.append("r%d %032X != %032X" % (k, m.r[k], s["r"][k]))
    for n in ("hi", "lo", "sa", "acc", "fcr31"):
        if getattr(m, n) != s[n]:
            diffs.append("%s %X != %X" % (n, getattr(m, n), s[n]))
    for k in range(32):
        if m.f[k] != s["f"][k]:
            diffs.append("f%d %08X != %08X" % (k, m.f[k], s["f"][k]))
        if m.vf[k] != s["vf"][k]:
            diffs.append("vf%d %s != %s" % (k, m.vf[k], s["vf"][k]))
        if (m.vi[k] & M32) != s["vi"][k]:
            diffs.append("vi%d %X != %X" % (k, m.vi[k], s["vi"][k]))
    if m.vacc != s["vacc"]:
        diffs.append("vacc %s != %s" % (m.vacc, s["vacc"]))
    for name, a, b in (("ram", m.mem, s["mem"]), ("spr", m.spr, s["spr"]), ("vu0", m.vu0, s["vu0"])):
        if bytes(a) != b:
            first = next(k for k in range(len(a)) if a[k] != b[k])
            n = sum(1 for k in range(first, min(len(a), first + 4096)) if a[k] != b[k])
            diffs.append("%s differs from %08X (%d bytes in the next 4 KB): emu %s, recomp %s" % (
                name, first, n, bytes(a[first:first + 16]).hex(), b[first:first + 16].hex()))
    return diffs


def run(elf_path, driver, only=None, seeds=10, log=print):
    driver = os.path.abspath(driver)
    elf = Elf(elf_path)
    names = {}
    for name, v, sz, bind, typ, shn in elf.symbols():
        if typ == 2:
            names.setdefault(name, v)
        if name == "_gp":
            names["_gp"] = v
    gp = names.get("_gp", 0)
    base = eeemu.EE(elf)
    env = dict(os.environ)
    env["PATH"] = MINGW_BIN + os.pathsep + env.get("PATH", "")
    total = failed = skipped = 0
    tmp = tempfile.mkdtemp(prefix="difftest_")
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    for name, build in CASES:
        if only and name not in only:
            continue
        if name not in names:
            skipped += 1
            continue
        addr = names[name]
        ok = 0
        for seed in range(seeds):
            rng = random.Random("%s/%d" % (name, seed))
            case = build(rng)
            args, mem = case[0], case[1]
            m = _initial(elf, base, rng, args, mem, gp, case[2] if len(case) > 2 else ())
            with open(fin, "wb") as f:
                f.write(_pack(m, addr))
            try:
                m.call(addr, ())
            except eeemu.Stop as ex:
                log("  %-22s seed %d: reference stopped: %s" % (name, seed, ex))
                failed += 1
                total += 1
                break
            p = subprocess.run([driver, fin, fout], env=env, capture_output=True, text=True)
            total += 1
            if p.returncode:
                log("  %-22s seed %d: recompiled run failed: %s" % (name, seed, p.stderr.strip()))
                failed += 1
                break
            with open(fout, "rb") as f:
                diffs = _compare(m, _unpack(f.read()))
            if diffs:
                failed += 1
                log("  %-22s seed %d: %d differences (args %s)" % (name, seed, len(diffs), [hex(a) for a in args]))
                for d in diffs[:8]:
                    log("      " + d)
                break
            ok += 1
        else:
            log("  %-22s %d/%d equal (%d instructions in the last)" % (name, ok, seeds, m.steps))
    log("%d runs, %d failed, %d functions absent" % (total, failed, skipped))
    from .. import ee as _ee
    used = set()
    lo, _o, size = elf.sections[".text"]
    for _a, w in elf.words(lo, lo + size):
        used.add(_ee.decode(w).op)
    done = set(base.ops)
    log("operations exercised: %d of the %d in .text; not exercised: %s" % (
        len(done & used), len(used), " ".join(sorted(used - done))))
    return failed


# --- leaf functions, fuzzed ---------------------------------------------------------------
# Every function that calls nothing and reaches neither the kernel nor the
# hardware can be run on its own. Its pointer arguments point into a buffer
# of mixed words (pointers back into the buffer, small integers, floats), so
# that structures holding pointers stay in RAM. A run the reference cannot
# finish (a pointer leaving RAM, a loop that does not end, a `break`: GCC
# guards every division with one) is skipped.

_NOT_LEAF = {"jal", "jalr", "syscall", "mfc0", "mtc0", "ei", "di", "vcallms", "vcallmsr",
             "bltzal", "bgezal", "bltzall", "bgezall"}


def leaves(elf):
    from .. import ee as _ee
    out = []
    for va, (name, size, src) in sorted(elf.functions().items()):
        ops = set()
        ok = True
        for a, w in elf.words(va, va + size):
            i = _ee.decode(w, a)
            if i.op in _NOT_LEAF or (i.op == "jr" and i.rs != 31) or (i.target is not None and not va <= i.target < va + size):
                ok = False
                break
            ops.add(i.op)
        if ok:
            out.append((va, name, ops))
    return out


def _fuzz_buffer(rng):
    words = []
    for _ in range(0x4000 // 4):
        k = rng.random()
        if k < 0.35:
            words.append(BUF + 4 * rng.randrange(0x3F00 // 4))
        elif k < 0.6:
            words.append(rng.randrange(-8, 64) & M32)
        elif k < 0.85:
            words.append(f2u(rng.uniform(-100, 100)))
        else:
            words.append(rng.getrandbits(32))
    return struct.pack("<%dI" % len(words), *words)


def fuzz_leaves(elf_path, driver, seeds=3, wanted=None, log=print):
    driver = os.path.abspath(driver)
    elf = Elf(elf_path)
    gp = next((v for n, v, *_ in elf.symbols() if n == "_gp"), 0)
    base = eeemu.EE(elf)
    env = dict(os.environ)
    env["PATH"] = MINGW_BIN + os.pathsep + env.get("PATH", "")
    tmp = tempfile.mkdtemp(prefix="difftest_")
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    cand = leaves(elf)
    if wanted:
        cand = [c for c in cand if c[2] & wanted]
    runs = equal = skipped = 0
    failed = []
    for va, name, ops in cand:
        for seed in range(seeds):
            rng = random.Random("leaf/%s/%d" % (name, seed))
            buf = _fuzz_buffer(rng)
            args = [BUF + 16 * rng.randrange(0x3000 // 16) for _ in range(8)]
            fargs = [f2u(rng.uniform(-100, 100)) for _ in range(8)]
            m = _initial(elf, base, rng, args, {BUF: buf}, gp, fargs)
            with open(fin, "wb") as f:
                f.write(_pack(m, va))
            try:
                m.call(va, (), max_steps=200_000)
            except eeemu.Stop:
                skipped += 1
                continue
            runs += 1
            try:
                p = subprocess.run([driver, fin, fout], env=env, capture_output=True, text=True, timeout=20)
            except subprocess.TimeoutExpired:
                failed.append((name, seed, ["recompiled run did not finish"]))
                break
            if p.returncode:
                failed.append((name, seed, [p.stderr.strip()]))
                break
            with open(fout, "rb") as f:
                diffs = _compare(m, _unpack(f.read()))
            if diffs:
                failed.append((name, seed, diffs))
                break
            equal += 1
    for name, seed, diffs in failed:
        log("  %-30s seed %d: %d differences" % (name, seed, len(diffs)))
        for d in diffs[:6]:
            log("      " + d)
    log("%d leaf functions, %d runs: %d equal, %d failed; %d runs skipped (the reference left RAM or looped)"
        % (len(cand), runs, equal, len(failed), skipped))
    return base.ops, len(failed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--driver", required=True)
    ap.add_argument("--only")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--leaves", action="store_true", help="fuzz every leaf function instead of the named cases")
    a = ap.parse_args()
    if a.leaves:
        ops, failed = fuzz_leaves(a.elf, a.driver, a.seeds)
        from .. import ee as _ee
        elf = Elf(a.elf)
        lo, _o, size = elf.sections[".text"]
        used = {_ee.decode(w).op for _a, w in elf.words(lo, lo + size)}
        print("operations exercised: %d of the %d in .text; not exercised: %s" % (
            len(set(ops) & used), len(used), " ".join(sorted(used - set(ops)))))
        raise SystemExit(1 if failed else 0)
    raise SystemExit(1 if run(a.elf, a.driver, set(a.only.split(",")) if a.only else None, a.seeds) else 0)


if __name__ == "__main__":
    main()

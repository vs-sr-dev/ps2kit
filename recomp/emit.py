"""EE instruction -> C++ statement(s), against ps2kit/runtime/ee.h.

emit(i) returns the C++ lines of one instruction that does not change the
flow of control; control_flow(i, slot, fn) those of a branch or jump
together with its delay slot (`slot`, the lines of the next instruction),
since the slot runs between the decision and the transfer:

    beq/bne/...     the condition is read first, then the slot runs, then the
                    branch is taken or not (the slot may change the operands)
    beql/bnel/...   likely: the slot runs only when the branch is taken
    jal             ra = pc + 8, the slot, then a direct C++ call; a callee
                    returns to pc + 8 (jr ra is a C++ return)
    j / b           inside the function a goto, outside it a tail call
    jr rX           a switch over the table when program.py found one,
                    otherwise an indirect tail call through the runtime
    jalr            an indirect call through the runtime

Every statement that declares a local is wrapped in its own block, so that a
`goto` never crosses an initialisation.
"""
from .. import vu


class Unsupported(Exception):
    pass


def fname(addr):
    return "f_%08X" % addr


def R(n):
    return "c.r[%d].ud[0]" % n


def W(n):
    return "c.r[%d].uw[0]" % n


def SD(n):
    return "c.r[%d].sd[0]" % n


def SW(n):
    return "c.r[%d].sw[0]" % n


def setr(n, e):
    return [] if n == 0 else ["c.r[%d].ud[0] = %s;" % (n, e)]


def u64(v):
    return "0x%Xull" % (v & 0xFFFFFFFFFFFFFFFF)


def u32(v):
    return "0x%Xu" % (v & 0xFFFFFFFF)


def ea(i):
    return W(i.rs) if i.imm == 0 else "(%s + %s)" % (W(i.rs), u32(i.imm))


def s32const(v):
    v &= 0xFFFFFFFF
    return u64(v - (1 << 32) if v & 0x80000000 else v)


# --- integer core ------------------------------------------------------------------

_IMM = {
    "addiu": lambda i: "S32(%s + %s)" % (W(i.rs), u32(i.imm)),
    "addi": lambda i: "S32(%s + %s)" % (W(i.rs), u32(i.imm)),
    "daddiu": lambda i: "%s + %s" % (R(i.rs), u64(i.imm)),
    "daddi": lambda i: "%s + %s" % (R(i.rs), u64(i.imm)),
    "slti": lambda i: "(uint64_t)(%s < (int64_t)%dll)" % (SD(i.rs), i.imm),
    "sltiu": lambda i: "(uint64_t)(%s < %s)" % (R(i.rs), u64(i.imm)),
    "andi": lambda i: "%s & %s" % (R(i.rs), u64(i.imm)),
    "ori": lambda i: "%s | %s" % (R(i.rs), u64(i.imm)),
    "xori": lambda i: "%s ^ %s" % (R(i.rs), u64(i.imm)),
    "lui": lambda i: s32const(i.imm << 16),
}

_R3 = {
    "addu": "S32({s32} + {t32})", "add": "S32({s32} + {t32})",
    "subu": "S32({s32} - {t32})", "sub": "S32({s32} - {t32})",
    "daddu": "{s} + {t}", "dadd": "{s} + {t}", "dsubu": "{s} - {t}", "dsub": "{s} - {t}",
    "and": "{s} & {t}", "or": "{s} | {t}", "xor": "{s} ^ {t}", "nor": "~({s} | {t})",
    "slt": "(uint64_t)({ss} < {ts})", "sltu": "(uint64_t)({s} < {t})",
    "sllv": "S32({t32} << ({s32} & 31))", "srlv": "S32({t32} >> ({s32} & 31))",
    "srav": "S32((int32_t){t32} >> ({s32} & 31))",
    "dsllv": "{t} << ({s32} & 63)", "dsrlv": "{t} >> ({s32} & 63)",
    "dsrav": "(uint64_t)({ts} >> ({s32} & 63))",
}

_SHIFT = {
    "sll": "S32({t32} << {sa})", "srl": "S32({t32} >> {sa})", "sra": "S32((int32_t){t32} >> {sa})",
    "dsll": "{t} << {sa}", "dsrl": "{t} >> {sa}", "dsra": "(uint64_t)({ts} >> {sa})",
    "dsll32": "{t} << {sa32}", "dsrl32": "{t} >> {sa32}", "dsra32": "(uint64_t)({ts} >> {sa32})",
}

_LOAD = {
    "lb": "(uint64_t)(int64_t)(int8_t)ee_ld8({ea})", "lbu": "(uint64_t)ee_ld8({ea})",
    "lh": "(uint64_t)(int64_t)(int16_t)ee_ld16({ea})", "lhu": "(uint64_t)ee_ld16({ea})",
    "lw": "S32(ee_ld32({ea}))", "lwu": "(uint64_t)ee_ld32({ea})", "ld": "ee_ld64({ea})",
}
_STORE = {
    "sb": "ee_st8({ea}, (uint8_t){t32})", "sh": "ee_st16({ea}, (uint16_t){t32})",
    "sw": "ee_st32({ea}, {t32})", "sd": "ee_st64({ea}, {t})",
    "swl": "ee_swl({ea}, {t32})", "swr": "ee_swr({ea}, {t32})",
    "sdl": "ee_sdl({ea}, {t})", "sdr": "ee_sdr({ea}, {t})",
}
_MERGE = {"lwl", "lwr", "ldl", "ldr"}

_NOPS = {"sync.l", "sync.p", "cache", "pref"}


def _fields(i):
    return dict(s=R(i.rs), t=R(i.rt), s32=W(i.rs), t32=W(i.rt), ss=SD(i.rs), ts=SD(i.rt),
                sa=i.sa, sa32=i.sa + 32, ea=ea(i))


def _integer(i):
    op = i.op
    if op in _IMM:
        return setr(i.rt, _IMM[op](i))
    if op in _R3:
        return setr(i.rd, _R3[op].format(**_fields(i)))
    if op in _SHIFT:
        if i.word == 0:
            return []                                     # nop
        return setr(i.rd, _SHIFT[op].format(**_fields(i)))
    if op in ("movz", "movn"):
        if i.rd == 0:
            return []
        return ["if (%s %s 0) %s = %s;" % (R(i.rt), "==" if op == "movz" else "!=", R(i.rd), R(i.rs))]
    if op in ("mult", "multu", "mult1", "multu1", "madd", "maddu", "madd1", "maddu1"):
        pipe = 1 if op.endswith("1") else 0
        signed = "u" not in op.replace("mult", "").replace("madd", "")
        prod = "(int64_t)%s * %s" % (SW(i.rs), SW(i.rt)) if signed else \
               "(int64_t)((uint64_t)%s * %s)" % (W(i.rs), W(i.rt))
        fn = "ee_madd" if op.startswith("madd") else "ee_mult"
        return ["%s(c, %d, %d, %s);" % (fn, i.rd, pipe, prod)]
    if op in ("div", "div1"):
        return ["ee_div(c, %d, %s, %s);" % (op.endswith("1"), SW(i.rs), SW(i.rt))]
    if op in ("divu", "divu1"):
        return ["ee_divu(c, %d, %s, %s);" % (op.endswith("1"), W(i.rs), W(i.rt))]
    hl = {"mfhi": "c.hi.ud[0]", "mflo": "c.lo.ud[0]", "mfhi1": "c.hi.ud[1]", "mflo1": "c.lo.ud[1]"}
    if op in hl:
        return setr(i.rd, hl[op])
    if op in ("mthi", "mtlo", "mthi1", "mtlo1"):
        return ["c.%s.ud[%d] = %s;" % (op[2:4], op.endswith("1"), R(i.rs))]
    if op == "mfsa":
        return setr(i.rd, "(uint64_t)c.sa")
    if op == "mtsa":
        return ["c.sa = %s;" % W(i.rs)]
    if op == "mtsab":
        return ["c.sa = ((%s ^ %s) & 15) * 8;" % (W(i.rs), u32(i.imm))]
    if op == "mtsah":
        return ["c.sa = ((%s ^ %s) & 7) * 16;" % (W(i.rs), u32(i.imm))]
    return None


def _memory(i):
    op = i.op
    f = _fields(i)
    if op in _LOAD:
        e = _LOAD[op].format(**f)
        return setr(i.rt, e) if i.rt else ["(void)%s;" % e]
    if op in _MERGE:
        return setr(i.rt, "ee_%s(%s, %s)" % (op, f["ea"], R(i.rt))) if i.rt else []
    if op in _STORE:
        return [_STORE[op].format(**f) + ";"]
    if op == "lq":
        return ["ee_ld128(%s, &c.r[%d]);" % (f["ea"], i.rt)] if i.rt else \
               ["{ EEReg t_; ee_ld128(%s, &t_); }" % f["ea"]]
    if op == "sq":
        return ["ee_st128(%s, &c.r[%d]);" % (f["ea"], i.rt)]
    if op == "lwc1":
        return ["c.f[%d].u = ee_ld32(%s);" % (i.rt, f["ea"])]
    if op == "swc1":
        return ["ee_st32(%s, c.f[%d].u);" % (f["ea"], i.rt)]
    if op == "lqc2":
        return ["ee_ld128(%s, &c.vf[%d]);" % (f["ea"], i.rt)] if i.rt else \
               ["{ VUReg t_; ee_ld128(%s, &t_); }" % f["ea"]]
    if op == "sqc2":
        return ["ee_st128(%s, &c.vf[%d]);" % (f["ea"], i.rt)]
    return None


def _system(i):
    op = i.op
    if op in _NOPS:
        return []
    if op == "syscall":
        return ["ee_syscall(c, 0x%08Xu);" % i.va]
    if op == "break":
        return ["ee_break(c, 0x%08Xu, 0x%Xu);" % (i.va, i.word >> 6 & 0xFFFFF)]
    if op == "ei":
        return ["ee_ei(c);"]
    if op == "di":
        return ["ee_di(c);"]
    if op == "mfc0":
        return setr(i.rt, "S32(ee_mfc0(c, %d))" % i.rd)
    if op == "mtc0":
        return ["ee_mtc0(c, %d, %s);" % (i.rd, W(i.rt))]
    return None


# --- MMI ----------------------------------------------------------------------------

def _lanes(n, expr):
    """C++ that fills d.<n lanes> from a per-lane expression in k."""
    return " ".join(expr.format(k=k) for k in range(n))


_MMI = {
    "por": "d.ud[0] = s.ud[0] | t.ud[0]; d.ud[1] = s.ud[1] | t.ud[1];",
    "pand": "d.ud[0] = s.ud[0] & t.ud[0]; d.ud[1] = s.ud[1] & t.ud[1];",
    "pxor": "d.ud[0] = s.ud[0] ^ t.ud[0]; d.ud[1] = s.ud[1] ^ t.ud[1];",
    "pnor": "d.ud[0] = ~(s.ud[0] | t.ud[0]); d.ud[1] = ~(s.ud[1] | t.ud[1]);",
    "paddw": _lanes(4, "d.uw[{k}] = s.uw[{k}] + t.uw[{k}];"),
    "psubw": _lanes(4, "d.uw[{k}] = s.uw[{k}] - t.uw[{k}];"),
    "paddh": _lanes(8, "d.uh[{k}] = (uint16_t)(s.uh[{k}] + t.uh[{k}]);"),
    "psubh": _lanes(8, "d.uh[{k}] = (uint16_t)(s.uh[{k}] - t.uh[{k}]);"),
    "paddb": _lanes(16, "d.ub[{k}] = (uint8_t)(s.ub[{k}] + t.ub[{k}]);"),
    "psubb": _lanes(16, "d.ub[{k}] = (uint8_t)(s.ub[{k}] - t.ub[{k}]);"),
    "pcgth": _lanes(8, "d.uh[{k}] = s.sh[{k}] > t.sh[{k}] ? 0xFFFF : 0;"),
    "pcgtw": _lanes(4, "d.uw[{k}] = s.sw[{k}] > t.sw[{k}] ? 0xFFFFFFFFu : 0;"),
    "pcgtb": _lanes(16, "d.ub[{k}] = s.sb[{k}] > t.sb[{k}] ? 0xFF : 0;"),
    "pmaxh": _lanes(8, "d.sh[{k}] = s.sh[{k}] > t.sh[{k}] ? s.sh[{k}] : t.sh[{k}];"),
    "pminh": _lanes(8, "d.sh[{k}] = s.sh[{k}] < t.sh[{k}] ? s.sh[{k}] : t.sh[{k}];"),
    "pmaxw": _lanes(4, "d.sw[{k}] = s.sw[{k}] > t.sw[{k}] ? s.sw[{k}] : t.sw[{k}];"),
    "pminw": _lanes(4, "d.sw[{k}] = s.sw[{k}] < t.sw[{k}] ? s.sw[{k}] : t.sw[{k}];"),
    "pcpyld": "d.ud[0] = t.ud[0]; d.ud[1] = s.ud[0];",
    "pcpyud": "d.ud[0] = s.ud[1]; d.ud[1] = t.ud[1];",
    "pextlw": "d.uw[0] = t.uw[0]; d.uw[1] = s.uw[0]; d.uw[2] = t.uw[1]; d.uw[3] = s.uw[1];",
    "pextuw": "d.uw[0] = t.uw[2]; d.uw[1] = s.uw[2]; d.uw[2] = t.uw[3]; d.uw[3] = s.uw[3];",
    "pextlh": _lanes(4, "d.uh[2*{k}] = t.uh[{k}]; d.uh[2*{k}+1] = s.uh[{k}];"),
    "pextuh": _lanes(4, "d.uh[2*{k}] = t.uh[{k}+4]; d.uh[2*{k}+1] = s.uh[{k}+4];"),
    "pextlb": _lanes(8, "d.ub[2*{k}] = t.ub[{k}]; d.ub[2*{k}+1] = s.ub[{k}];"),
    "pextub": _lanes(8, "d.ub[2*{k}] = t.ub[{k}+8]; d.ub[2*{k}+1] = s.ub[{k}+8];"),
    "ppacb": _lanes(8, "d.ub[{k}] = t.ub[2*{k}]; d.ub[{k}+8] = s.ub[2*{k}];"),
    "ppach": _lanes(4, "d.uh[{k}] = t.uh[2*{k}]; d.uh[{k}+4] = s.uh[2*{k}];"),
    "ppacw": "d.uw[0] = t.uw[0]; d.uw[1] = t.uw[2]; d.uw[2] = s.uw[0]; d.uw[3] = s.uw[2];",
    "pcpyh": "for (int k = 0; k < 4; ++k) { d.uh[k] = t.uh[0]; d.uh[k + 4] = t.uh[4]; }",
    "prot3w": "d.uw[0] = t.uw[1]; d.uw[1] = t.uw[2]; d.uw[2] = t.uw[0]; d.uw[3] = t.uw[3];",
    "pexcw": "d.uw[0] = t.uw[0]; d.uw[1] = t.uw[2]; d.uw[2] = t.uw[1]; d.uw[3] = t.uw[3];",
    "psllh": _lanes(8, "d.uh[{k}] = (uint16_t)(t.uh[{k}] << (SA & 15));"),
    "psrlh": _lanes(8, "d.uh[{k}] = (uint16_t)(t.uh[{k}] >> (SA & 15));"),
    "psrah": _lanes(8, "d.sh[{k}] = (int16_t)(t.sh[{k}] >> (SA & 15));"),
    "psllw": _lanes(4, "d.uw[{k}] = t.uw[{k}] << SA;"),
    "psrlw": _lanes(4, "d.uw[{k}] = t.uw[{k}] >> SA;"),
    "psraw": _lanes(4, "d.sw[{k}] = t.sw[{k}] >> SA;"),
    "qfsrv": "ee_qfsrv(d, s, t, c.sa);",
}


def _mmi(i):
    body = _MMI.get(i.op)
    if body is None:
        return None
    if i.rd == 0:
        return []
    body = body.replace("SA", str(i.sa))
    return ["{ const EEReg s = c.r[%d], t = c.r[%d]; EEReg d; %s c.r[%d] = d; }" % (i.rs, i.rt, body, i.rd)]


# --- FPU ----------------------------------------------------------------------------

def _fpu(i):
    op, fs, ft, fd = i.op, i.rd, i.rt, i.sa
    F = lambda n: "c.f[%d].f" % n
    two = {"add.s": "fadd", "sub.s": "fsub", "mul.s": "fmul", "div.s": "fdiv",
           "max.s": "fmax_", "min.s": "fmin_", "rsqrt.s": "frsqrt"}
    if op in two:
        return ["%s = %s(%s, %s);" % (F(fd), two[op], F(fs), F(ft))]
    acc = {"adda.s": "fadd(%s, %s)", "suba.s": "fsub(%s, %s)", "mula.s": "fmul(%s, %s)",
           "madda.s": "fadd(c.acc.f, fmul(%s, %s))", "msuba.s": "fsub(c.acc.f, fmul(%s, %s))"}
    if op in acc:
        return ["c.acc.f = %s;" % (acc[op] % (F(fs), F(ft)))]
    if op == "madd.s":
        return ["%s = fadd(c.acc.f, fmul(%s, %s));" % (F(fd), F(fs), F(ft))]
    if op == "msub.s":
        return ["%s = fsub(c.acc.f, fmul(%s, %s));" % (F(fd), F(fs), F(ft))]
    one = {"abs.s": "fabs_", "neg.s": "fneg"}
    if op in one:
        return ["%s = %s(%s);" % (F(fd), one[op], F(fs))]
    if op == "sqrt.s":
        return ["%s = fsqrt(%s);" % (F(fd), F(ft))]
    if op == "mov.s":
        return ["c.f[%d].u = c.f[%d].u;" % (fd, fs)]
    if op == "cvt.w.s":
        return ["c.f[%d].i = fcvtw(%s);" % (fd, F(fs))]
    if op == "cvt.s.w":
        return ["%s = ps2f((float)c.f[%d].i);" % (F(fd), fs)]
    cmp = {"c.f.s": None, "c.eq.s": "==", "c.lt.s": "<", "c.le.s": "<="}
    if op in cmp:
        if cmp[op] is None:
            return ["fcmp(c, false);"]
        return ["fcmp(c, ps2f(%s) %s ps2f(%s));" % (F(fs), cmp[op], F(ft))]
    if op == "mfc1":
        return setr(i.rt, "S32(c.f[%d].u)" % fs)
    if op == "mtc1":
        return ["c.f[%d].u = %s;" % (fs, W(i.rt))]
    if op == "cfc1":
        return setr(i.rt, "S32(c.fcr31)" if fs == 31 else "0x2E00ull")
    if op == "ctc1":
        return ["c.fcr31 = %s;" % W(i.rt)] if fs == 31 else []
    return None


# --- COP2: transfers and VU0 macro mode ------------------------------------------------

_VI_DIRECT = set(range(16)) | {16, 17, 18, 20, 21, 22}


def _cop2(i):
    op = i.op
    base = op.rsplit(".", 1)[0] if op.startswith(("qmfc2", "qmtc2", "cfc2", "ctc2")) else op
    if base == "qmfc2":
        return ["std::memcpy(&c.r[%d], &c.vf[%d], 16);" % (i.rt, i.rd)] if i.rt else []
    if base == "qmtc2":
        return ["std::memcpy(&c.vf[%d], &c.r[%d], 16);" % (i.rd, i.rt)] if i.rd else []
    if base == "cfc2":
        if i.rd in _VI_DIRECT:
            return setr(i.rt, "S32(c.vi[%d])" % i.rd)
        return setr(i.rt, "S32(ee_cfc2(c, %d))" % i.rd)
    if base == "ctc2":
        if i.rd == 0:
            return []
        if i.rd < 16:
            return ["c.vi[%d] = %s & 0xFFFFu;" % (i.rd, W(i.rt))]
        if i.rd in _VI_DIRECT:
            return ["c.vi[%d] = %s;" % (i.rd, W(i.rt))]
        return ["ee_ctc2(c, %d, %s);" % (i.rd, W(i.rt))]
    if op == "vcallms":
        return ["ee_vu0_callms(c, 0x%Xu);" % i.imm]
    if op == "vcallmsr":
        return ["ee_vu0_callms(c, c.vi[VI_CMSAR0] * 8);"]
    if i.form == "vu_upper":
        return _vu_upper(i)
    if i.form in ("vu_lower", "vu_int"):
        return _vu_lower(i)
    return None


_ARITH = {"add": "fadd({s}, {b})", "sub": "fsub({s}, {b})", "mul": "fmul({s}, {b})",
          "madd": "fadd(a.f[{k}], fmul({s}, {b}))", "msub": "fsub(a.f[{k}], fmul({s}, {b}))"}
# MAX and MINI compare bit patterns and set no flags (ee.h's vu_max, vu_min).
_MINMAX_BITS = {"t.f[{k}]": "t.u[{k}]", "q": "c.vi[VI_Q]", "ii": "c.vi[VI_I]"}
_OPCROSS = ((1, 2), (2, 0), (0, 1))     # opmula/opmsub: x = s.y*t.z, y = s.z*t.x, z = s.x*t.y


def _dest(i):
    return i.word >> 21 & 15


def _lanes_of(dest):
    return [k for k in range(4) if dest & (8 >> k)]


def _vu_upper(i):
    name, form = i.op[1:], i.extra
    dest, ft, fs, fd, bc = _dest(i), i.rt, i.rd, i.sa, i.word & 3
    lanes = _lanes_of(dest)
    if name == "nop":
        return []
    if form == "fs_ftw":                                  # clipw
        return ["vu_clip(c, c.vf[%d], c.vf[%d].f[3]);" % (fs, ft)]
    if form == "ft_fs":                                   # abs, itofN, ftoiN: no flags
        if ft == 0:
            return []
        if name == "abs":
            e = "r.f[{k}] = fabs_(s.f[{k}]);"
        elif name.startswith("itof"):
            e = "r.f[{k}] = ps2f((float)s.i[{k}] / %d.0f);" % (1 << int(name[4:]))
        elif name.startswith("ftoi"):
            e = "r.i[{k}] = vu_ftoi(s.f[{k}], %d.0f);" % (1 << int(name[4:]))
        else:
            raise Unsupported("vu " + name)
        body = " ".join(e.format(k=k) for k in lanes)
        return ["{ const VUReg s = c.vf[%d]; VUReg r = c.vf[%d]; %s c.vf[%d] = r; }" % (fs, ft, body, ft)]
    to_acc = form.startswith("acc")
    if form.endswith("ftbc"):
        base, b = name[:-1], "t.f[%d]" % bc
    elif form.endswith("_q"):
        base, b = name[:-1], "q"
    elif form.endswith("_i"):
        base, b = name[:-1], "ii"
    else:
        base, b = name, "t.f[{k}]"
    if base.endswith("a") and base not in ("opmula",) and to_acc:
        base = base[:-1]                                  # adda -> add, madda -> madd, mula -> mul
    target = "c.vacc" if to_acc else "c.vf[%d]" % fd
    if base in ("max", "mini"):
        if fd == 0:
            return []
        bu = _MINMAX_BITS.get(b, b.replace("t.f[", "t.u["))
        body = " ".join("r.u[%d] = vu_%s(s.u[%d], %s);" % (k, "max" if base == "max" else "min", k, bu.format(k=k))
                        for k in lanes)
        return ["{ const VUReg s = c.vf[%d], t = c.vf[%d]; VUReg r = c.vf[%d]; %s c.vf[%d] = r; }"
                % (fs, ft, fd, body, fd)]
    parts = []
    for k in lanes:
        if base in ("opmula", "opmsub"):
            if k == 3:
                continue
            p, q = _OPCROSS[k]
            prod = "fmul(s.f[%d], t.f[%d])" % (p, q)
            parts.append("r.f[%d] = %s;" % (k, prod if base == "opmula" else "fsub(a.f[%d], %s)" % (k, prod)))
            continue
        if base not in _ARITH:
            raise Unsupported("vu " + name)
        parts.append("r.f[%d] = %s;" % (k, _ARITH[base].format(s="s.f[%d]" % k, b=b.format(k=k), k=k)))
    pre = "const VUReg s = c.vf[%d], t = c.vf[%d], a = c.vacc; VUReg r = %s;" % (fs, ft, target)
    if b == "q":
        pre += " const float q = vu_q(c);"
    if b == "ii":
        pre += " const float ii = vu_i(c);"
    store = "" if (not to_acc and fd == 0) else "%s = r; " % target
    return ["{ %s %s %svu_flags(c, %d, r); }" % (pre, " ".join(parts), store, dest)]


def _vu_lower(i):
    name, form = i.op[1:], i.extra
    dest, ft, fs = _dest(i), i.rt, i.rd
    it, is_, id_ = i.rt, i.rd, i.sa
    fsf, ftf = i.word >> 21 & 3, i.word >> 23 & 3
    lanes = _lanes_of(dest)
    vi = lambda n, e: [] if n == 0 else ["c.vi[%d] = (%s) & 0xFFFFu;" % (n, e)]
    if name == "waitq":
        return []
    if name in ("move", "mr32"):
        if ft == 0:
            return []
        src = [0, 1, 2, 3] if name == "move" else [1, 2, 3, 0]
        body = " ".join("r.u[%d] = s.u[%d];" % (k, src[k]) for k in lanes)
        return ["{ const VUReg s = c.vf[%d]; VUReg r = c.vf[%d]; %s c.vf[%d] = r; }" % (fs, ft, body, ft)]
    if name in ("lqi", "lqd"):
        pre = "" if name == "lqi" else "if (%d) c.vi[%d] = (c.vi[%d] - 1) & 0xFFFFu; " % (is_, is_, is_)
        post = "" if name == "lqd" else "if (%d) c.vi[%d] = (c.vi[%d] + 1) & 0xFFFFu; " % (is_, is_, is_)
        body = " ".join("c.vf[%d].u[%d] = m->u[%d];" % (ft, k, k) for k in lanes) if ft else ""
        return ["{ %sconst VUReg* m = vu0_q(c.vi[%d]); %s %s}" % (pre, is_, body, post)]
    if name in ("sqi", "sqd"):
        pre = "" if name == "sqi" else "if (%d) c.vi[%d] = (c.vi[%d] - 1) & 0xFFFFu; " % (it, it, it)
        post = "" if name == "sqd" else "if (%d) c.vi[%d] = (c.vi[%d] + 1) & 0xFFFFu; " % (it, it, it)
        body = " ".join("m->u[%d] = c.vf[%d].u[%d];" % (k, fs, k) for k in lanes)
        return ["{ %sVUReg* m = vu0_q(c.vi[%d]); %s %s}" % (pre, it, body, post)]
    if name == "div":
        return ["vu_div(c, c.vf[%d].f[%d], c.vf[%d].f[%d]);" % (fs, fsf, ft, ftf)]
    if name == "sqrt":
        return ["vu_sqrt(c, c.vf[%d].f[%d]);" % (ft, ftf)]
    if name == "rsqrt":
        return ["vu_rsqrt(c, c.vf[%d].f[%d], c.vf[%d].f[%d]);" % (fs, fsf, ft, ftf)]
    if name == "mtir":
        return vi(it, "c.vf[%d].u[%d]" % (fs, fsf))
    if name == "mfir":
        if ft == 0:
            return []
        return [" ".join("c.vf[%d].i[%d] = (int32_t)(int16_t)c.vi[%d];" % (ft, k, is_) for k in lanes)]
    if name == "iadd":
        return vi(id_, "c.vi[%d] + c.vi[%d]" % (is_, it))
    if name == "isub":
        return vi(id_, "c.vi[%d] - c.vi[%d]" % (is_, it))
    if name == "iand":
        return vi(id_, "c.vi[%d] & c.vi[%d]" % (is_, it))
    if name == "ior":
        return vi(id_, "c.vi[%d] | c.vi[%d]" % (is_, it))
    if name == "iaddi":
        imm = id_ - 32 if id_ & 16 else id_
        return vi(it, "c.vi[%d] + (%d)" % (is_, imm))
    raise Unsupported("vu " + name)


# --- dispatch -------------------------------------------------------------------------

def emit(i):
    """C++ lines for an instruction that does not transfer control."""
    for f in (_integer, _memory, _system, _mmi, _fpu, _cop2):
        out = f(i)
        if out is not None:
            return out
    raise Unsupported(i.op)


_COND = {
    "beq": "{s} == {t}", "bne": "{s} != {t}", "blez": "{ss} <= 0", "bgtz": "{ss} > 0",
    "bltz": "{ss} < 0", "bgez": "{ss} >= 0",
    "bc1f": "!(c.fcr31 & FPU_C)", "bc1t": "(c.fcr31 & FPU_C)",
}


def condition(i):
    op = i.op
    base = op[:-1] if i.is_likely else op
    if base.endswith("al"):
        base = base[:-2]
    if base not in _COND:
        raise Unsupported(op)
    if base == "beq" and i.rs == i.rt:
        return None                                       # always taken
    return _COND[base].format(**_fields(i))


def terminates(i):
    """Control never falls through past this instruction's delay slot."""
    if i.op in ("j", "jr"):
        return True
    return i.op in ("beq", "beql") and i.rs == i.rt


def control_flow(i, slot, fn):
    """C++ lines for a branch or jump with its delay slot's lines `slot`."""
    op = i.op
    s = " ".join(slot)
    if op == "jal":
        return ["c.r[31].ud[0] = %s; %s %s" % (s32const(i.va + 8), s, fn.call(i.target))]
    if op == "j":
        return ["%s %s" % (s, fn.jump(i.target, i.va))]
    if op == "jr":
        if i.rs == 31:
            return ["%s return;" % s]
        if i.va in fn.tables:
            cases = " ".join("case 0x%08Xu: goto L_%08X;" % (t, t) for t in dict.fromkeys(fn.tables[i.va]))
            return ["{ const uint32_t t_ = %s; %s switch (t_) { %s default: "
                    "ee_unimplemented(c, 0x%08Xu, \"switch target\"); return; } }" % (W(i.rs), s, cases, i.va)]
        return ["{ const uint32_t t_ = %s; %s ee_call(c, t_); return; }" % (W(i.rs), s)]
    if op == "jalr":
        link = "c.r[%d].ud[0] = %s;" % (i.rd, s32const(i.va + 8)) if i.rd else ""
        return ["{ const uint32_t t_ = %s; %s %s ee_call(c, t_); }" % (W(i.rs), link, s)]
    # conditional branches, plain and likely, with or without link
    cond = condition(i)
    link = op.endswith(("al", "all")) and op.startswith(("bltz", "bgez"))
    if link:
        transfer = "c.r[31].ud[0] = %s; %s" % (s32const(i.va + 8), fn.call(i.target))
    else:
        transfer = fn.jump(i.target, i.va)
    if cond is None:
        return ["%s %s" % (s, transfer)]
    if i.is_likely:
        return ["if (%s) { %s %s }" % (cond, s, transfer)]
    return ["{ const bool t_ = %s; %s if (t_) { %s } }" % (cond, s, transfer)]

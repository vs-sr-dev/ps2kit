// ps2kit runtime — the vector units' micro mode, for VU0 and VU1 alike.
//
// A micro program runs at once, to its end (VU0: vcallms, vcallmsr; VU1:
// VIF1's MSCAL, MSCALF, MSCNT): the units are infinitely fast here, so
// whoever starts one, or waits for one, always finds it finished.
//
// Each step is one instruction pair (upper word above the lower one). The
// pair's two halves read the registers as they were before the pair; when
// both write the same VF register (or both the clip flags), the lower
// instruction is dropped. With the I bit the lower word is loaded into I
// after the upper instruction ran. Branches have one delay slot; the E bit
// ends the program after the next pair. A conditional branch right after the
// integer instruction that wrote its register reads the register's older
// value. (These pair rules are those of PCSX2's interpreter.)
//
// The pipelines, in cycles (one pair issues per cycle, unless it stalls):
// * FMAC: a VF register an instruction writes is ready 4 cycles after it
//   issues; an instruction that reads one of its lanes before that waits
//   (stalls) until it is. Register values therefore come out as if every
//   instruction completed before the next began. ACC is forwarded: MULA,
//   MADDA, MADD back to back do not wait.
// * The flags (MAC, status, clip) an FMAC instruction (or CLIP) sets are
//   written 4 cycles after it issues; FMAND, FSAND, FCAND and the others
//   read the flags as written by then, without waiting.
// * FDIV: DIV and SQRT write Q 7 cycles after they issue, RSQRT 13 (with the
//   status flags D and I); an instruction that reads Q earlier sees the old
//   Q. A second DIV, SQRT or RSQRT, and WAITQ, wait for the first to finish.
// * EFU (VU1): ESADD 11, ERSADD 18, ELENG 18, ERLENG 24, ESUM 12, ERCPR 12,
//   ESQRT 12, ERSQRT 18, ESIN 29, EATAN, EATANxy, EATANxz 54, EEXP 44 cycles
//   to P; MFP reads the P written by then; WAITP and another EFU
//   instruction wait for the one in flight.
// * ILW and ILWR take 4 cycles; a branch on their register waits for them.
// At the end of a program everything in flight is written.
// The latencies are the VU manual's (and PCSX2's); the rule for which reads
// wait is PCSX2's interpreter's. Reads that see a value still in flight are
// counted (the heartbeat) and, with PS2KIT_TRACE, reported once per address.
//
// The FMAC's arithmetic is the FPU model's (ee.h: no denormals, no
// infinities, rounding toward zero), computed with the host's DAZ and FTZ set
// while a program runs: a denormal operand counts as a signed zero and a
// denormal result becomes one, as ps2f makes them; with rounding toward zero a
// finite result never overflows to an infinity. Only an operand with exponent
// 255 (from memory) needs ps2f's clamp first. The same bits, fewer steps.
//
// The EFU's functions are computed in host single precision (ESIN, EATAN
// and EEXP with the host's library, as the VU manual defines them: sin x,
// atan x, exp(-x)); the console uses its own series and may differ in the
// last bits.
#include "vu_ops.h"

#include <algorithm>
#include <cmath>
#include <xmmintrin.h>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

const int LAT_FMAC = 4, LAT_DIV = 7, LAT_SQRT = 7, LAT_RSQRT = 13, LAT_ILW = 4;

// The FMAC's arithmetic, the flags as they land and the EFU are vu_ops.h's,
// shared with the translated programs.

int lanes(uint32_t w) { return (w >> 21) & 15; }
bool lane(int dest, int k) { return dest & (8 >> k); }
// dest (x is bit 3) as a lane mask with x in bit 0
int lane_mask(int dest) { return ((dest >> 3) & 1) | ((dest >> 1) & 2) | ((dest << 1) & 4) | ((dest << 3) & 8); }

// ---- the pipelines ----------------------------------------------------------------------------
struct FlagWrite { uint64_t ready; bool clip; uint32_t mac; uint32_t clip_bits; };

struct Pipe {
    uint64_t t = 0;                      // the cycle the next pair issues in
    uint64_t pair = 0;                   // pairs issued (the branch rule counts these)
    uint64_t vf_ready[32 * 4] = {};      // by register * 4 + lane
    // The VF registers written in the last cycles (a ring of the last four
    // writes): a pair that reads none of them cannot stall on a VF register.
    struct { uint64_t ready; uint32_t regs; } recent[4] = {};
    unsigned rpos = 0;
    uint32_t in_flight() const {
        uint32_t m = 0;
        for (const auto& r : recent) m |= r.ready > t ? r.regs : 0;
        return m;
    }
    void wrote(uint32_t regs, uint64_t ready) {
        if (recent[(rpos - 1) & 3].ready == ready) recent[(rpos - 1) & 3].regs |= regs;
        else { recent[rpos & 3] = {ready, regs}; ++rpos; }
    }
    uint64_t vi_ready[16] = {};
    FlagWrite flags[8];                  // in issue order (all take 4 cycles): at most 5 in flight
    unsigned fhead = 0, fn = 0;
    void push(const FlagWrite& f) { flags[(fhead + fn++) & 7] = f; }
    bool q_busy = false; uint64_t q_ready = 0; uint32_t q_val = 0, q_flags = 0;
    bool p_busy = false; uint64_t p_ready = 0; uint32_t p_val = 0;
};

void land_flags(EEContext& c, const FlagWrite& f) {
    if (f.clip) vu_land_clip(c, f.clip_bits);
    else vu_land_mac(c, f.mac);
}

// Everything due by cycle t (or all of it) is written.
void land_slow(VUUnit& u, Pipe& p, bool all);
inline void land(VUUnit& u, Pipe& p, bool all) {
    if (p.fn | p.q_busy | p.p_busy) land_slow(u, p, all);
}
void land_slow(VUUnit& u, Pipe& p, bool all) {
    EEContext& c = *u.c;
    while (p.fn && (all || p.flags[p.fhead].ready <= p.t)) {
        land_flags(c, p.flags[p.fhead]);
        p.fhead = (p.fhead + 1) & 7;
        --p.fn;
    }
    if (p.q_busy && (all || p.q_ready <= p.t)) {
        p.q_busy = false;
        c.vi[VI_Q] = p.q_val;
        vu_divflags(c, p.q_flags);
    }
    if (p.p_busy && (all || p.p_ready <= p.t)) {
        p.p_busy = false;
        u.p = p.p_val;
    }
}

void report(VUUnit& u, uint32_t pc, const char* what) {
    if (g_trace < 1 || !u.warned.insert(pc).second) return;
    std::fprintf(stderr, "[vu%d] %04X reads %s while a newer one is in flight\n", u.n, pc, what);
}

// The VF lanes a pair reads, for its stalls.
struct Reads {
    int n = 0;
    uint8_t reg[4], mask[4];
    void add(int r, int m) { if (r && m) { reg[n] = (uint8_t)r; mask[n] = (uint8_t)m; ++n; } }
};

void upper_reads(uint32_t w, Reads& r) {
    uint32_t op = w & 0x3F;
    int ft = (w >> 16) & 31, fs = (w >> 11) & 31, d = lane_mask(lanes(w)), bc = 1 << (w & 3);
    if (op < 0x1C) { r.add(fs, d); r.add(ft, bc); return; }
    if (op < 0x30) {
        bool q_or_i = op <= 0x27;                        // the Q and I forms read fs alone
        if (op == 0x2E) { r.add(fs, 7); r.add(ft, 7); return; }   // opmsub
        r.add(fs, d);
        if (!q_or_i) r.add(ft, d);
        return;
    }
    if (op < 0x3C) return;
    uint32_t x = (((w >> 6) & 31) << 2) | (op & 3);
    if (x < 0x10 || (x >= 0x18 && x < 0x1C)) { r.add(fs, d); r.add(ft, bc); return; }
    if (x < 0x18 || x == 0x1D) { r.add(fs, d); return; }          // itof, ftoi, abs
    if (x == 0x1F) { r.add(fs, 7); r.add(ft, 8); return; }        // clip
    if (x == 0x2E) { r.add(fs, 7); r.add(ft, 7); return; }        // opmula
    if (x == 0x2F) return;                                        // nop
    r.add(fs, d);
    if (x >= 0x28) r.add(ft, d);
}

enum { P_NONE, P_FDIV, P_EFU, P_BRANCH };

int lower_reads(uint32_t w, Reads& r, uint32_t& vi_read) {
    uint32_t op = w >> 25;
    int it = (w >> 16) & 31, is = (w >> 11) & 31, d = lane_mask(lanes(w));
    int fsf = 1 << ((w >> 21) & 3), ftf = 1 << ((w >> 23) & 3);
    vi_read = 0;
    if (w == 0x8000033Cu) return P_NONE;
    if (op == 0x01) { r.add(is, d); return P_NONE; }                  // sq
    if (op >= 0x24 && op <= 0x2F) {                                   // jr, jalr, conditional branches
        vi_read = (1u << is) | (op >= 0x28 && op <= 0x29 ? 1u << it : 0);
        return P_BRANCH;
    }
    if (op != 0x40 || (w & 0x3F) < 0x3C) return P_NONE;
    uint32_t x = (((w >> 6) & 31) << 2) | (w & 3);
    switch (x) {
    case 0x30: r.add(is, d); return P_NONE;                           // move
    case 0x31: r.add(is, 15); return P_NONE;                          // mr32
    case 0x35: case 0x37: r.add(is, d); return P_NONE;                // sqi, sqd
    case 0x38: case 0x3A: r.add(is, fsf); r.add(it, ftf); return P_FDIV;   // div, rsqrt
    case 0x39: r.add(it, ftf); return P_FDIV;                         // sqrt
    case 0x3B: return P_FDIV;                                         // waitq
    case 0x3C: case 0x42: case 0x43: r.add(is, fsf); return P_NONE;   // mtir, rinit, rxor
    case 0x70: case 0x71: case 0x72: case 0x73: r.add(is, 7); return P_EFU;   // esadd ersadd eleng erleng
    case 0x74: r.add(is, 3); return P_EFU;                            // eatanxy
    case 0x75: r.add(is, 5); return P_EFU;                            // eatanxz
    case 0x76: r.add(is, 15); return P_EFU;                           // esum
    case 0x78: case 0x79: case 0x7A: case 0x7C: case 0x7D: case 0x7E: r.add(is, fsf); return P_EFU;
    case 0x7B: return P_EFU;                                          // waitp
    default: return P_NONE;
    }
}

VUReg* mem_q(VUUnit& u, uint32_t qaddr, uint32_t pc) {
    uint32_t a = qaddr * 16;
    if (u.n == 0 && (a & 0x4000)) return ee_vu0_vu1reg(qaddr);
    return (VUReg*)(u.mem + (a & u.mem_mask));
}

// ---- the upper instruction --------------------------------------------------------------------
enum Kind { K_ADD, K_SUB, K_MUL, K_MADD, K_MSUB, K_MAX, K_MINI, K_OPMULA, K_OPMSUB };
enum { B_BC, B_Q, B_I, B_FT };
enum { UK_NOP, UK_ARITH, UK_ITOF, UK_FTOI, UK_ABS, UK_CLIP };

// A pair, decoded once (until its micro memory changes).
struct Dec {
    uint32_t lo, up;
    bool valid;
    // the upper instruction
    uint8_t uk, kind, bsrc, bc, dest, fs, ft;
    int8_t target;                       // -2 none, -1 ACC, else a VF register
    float scale;                         // ITOF, FTOI
    uint8_t lm;                          // dest as a lane mask, x in bit 0
    // the stalls: the VF lanes both halves read (register * 4 + lane), the lower one's unit, the VI
    // registers a branch reads
    uint8_t nrl, rl[16], pipe;
    uint16_t vi_read;
    uint32_t rregs;                      // the VF registers read, as a mask
    // the lower instruction: run or not (I bit, or dropped for writing what the upper one writes)
    bool lower;
    int8_t lw;                           // the VF register it writes
    uint8_t lwm;                         // and its lanes (x in bit 0)
};

void decode_upper(Dec& d, uint32_t w, int unit, uint32_t pc) {
    uint32_t op = w & 0x3F;
    int ft = (w >> 16) & 31, fs = (w >> 11) & 31, fd = (w >> 6) & 31;
    d.dest = (uint8_t)lanes(w);
    d.lm = (uint8_t)lane_mask(d.dest);
    d.bc = (uint8_t)(w & 3);
    d.fs = (uint8_t)fs;
    d.ft = (uint8_t)ft;
    d.uk = UK_ARITH;
    bool to_acc = false;
    auto special = [&](int uk) {
        d.uk = (uint8_t)(ft ? uk : UK_NOP);
        d.target = (int8_t)ft;
    };
    if (op < 0x1C) {
        static const Kind K[7] = {K_ADD, K_SUB, K_MADD, K_MSUB, K_MAX, K_MINI, K_MUL};
        d.kind = K[op >> 2];
        d.bsrc = B_BC;
    } else if (op < 0x30) {
        static const struct { Kind k; int b; } T[20] = {
            {K_MUL, B_Q}, {K_MAX, B_I}, {K_MUL, B_I}, {K_MINI, B_I}, {K_ADD, B_Q}, {K_MADD, B_Q}, {K_ADD, B_I},
            {K_MADD, B_I}, {K_SUB, B_Q}, {K_MSUB, B_Q}, {K_SUB, B_I}, {K_MSUB, B_I}, {K_ADD, B_FT}, {K_MADD, B_FT},
            {K_MUL, B_FT}, {K_MAX, B_FT}, {K_SUB, B_FT}, {K_MSUB, B_FT}, {K_OPMSUB, B_FT}, {K_MINI, B_FT}};
        d.kind = T[op - 0x1C].k;
        d.bsrc = (uint8_t)T[op - 0x1C].b;
    } else if (op >= 0x3C) {
        uint32_t x = (((w >> 6) & 31) << 2) | (op & 3);
        to_acc = true;
        if (x < 0x10) {
            static const Kind K[4] = {K_ADD, K_SUB, K_MADD, K_MSUB};
            d.kind = K[x >> 2];
            d.bsrc = B_BC;
        } else if (x < 0x18) {                                   // itof0-15, ftoi0-15: no flags
            static const float SCALE[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
            d.scale = SCALE[x & 3];
            special(x < 0x14 ? UK_ITOF : UK_FTOI);
            return;
        } else if (x < 0x1C) {
            d.kind = K_MUL;
            d.bsrc = B_BC;
        } else {
            switch (x) {
            case 0x1C: d.kind = K_MUL; d.bsrc = B_Q; break;
            case 0x1D: special(UK_ABS); return;
            case 0x1E: d.kind = K_MUL; d.bsrc = B_I; break;
            case 0x1F: d.uk = UK_CLIP; d.target = -2; return;
            case 0x20: d.kind = K_ADD; d.bsrc = B_Q; break;
            case 0x21: d.kind = K_MADD; d.bsrc = B_Q; break;
            case 0x22: d.kind = K_ADD; d.bsrc = B_I; break;
            case 0x23: d.kind = K_MADD; d.bsrc = B_I; break;
            case 0x24: d.kind = K_SUB; d.bsrc = B_Q; break;
            case 0x25: d.kind = K_MSUB; d.bsrc = B_Q; break;
            case 0x26: d.kind = K_SUB; d.bsrc = B_I; break;
            case 0x27: d.kind = K_MSUB; d.bsrc = B_I; break;
            case 0x28: d.kind = K_ADD; d.bsrc = B_FT; break;
            case 0x29: d.kind = K_MADD; d.bsrc = B_FT; break;
            case 0x2A: d.kind = K_MUL; d.bsrc = B_FT; break;
            case 0x2C: d.kind = K_SUB; d.bsrc = B_FT; break;
            case 0x2D: d.kind = K_MSUB; d.bsrc = B_FT; break;
            case 0x2E: d.kind = K_OPMULA; d.bsrc = B_FT; break;
            case 0x2F: d.uk = UK_NOP; d.target = -2; return;
            default: rt_die("VU%d micro program at %04X: upper instruction %08X is not one", unit, pc, w);
            }
        }
    } else {
        rt_die("VU%d micro program at %04X: upper instruction %08X is not one", unit, pc, w);
    }
    // vf0 is read-only: the result goes nowhere, the flags still change
    d.target = (int8_t)(to_acc ? -1 : fd ? fd : -2);
}

// The upper instruction's result, held until the lower one has run.
struct Upper {
    int target = -2;                     // -2 none, -1 ACC, else a VF register
    int dest = 0;
    VUReg r;
    bool flags = false;                  // MAC and status from r
    bool clip = false;
    uint32_t clip_bits = 0;
};

Upper do_upper(VUUnit& vu, Pipe& p, const Dec& d, uint32_t pc) {
    EEContext& c = *vu.c;
    Upper u;
    int dest = d.dest;
    switch (d.uk) {
    case UK_NOP: return u;
    case UK_ITOF: case UK_FTOI: case UK_ABS: {
        const VUReg s = c.vf[d.fs];
        u.target = d.target;
        u.dest = dest;
        u.r = c.vf[d.target];
        for (int k = 0; k < 4; ++k) {
            if (!lane(dest, k)) continue;
            if (d.uk == UK_ITOF) u.r.f[k] = ps2f((float)s.i[k] / d.scale);
            else if (d.uk == UK_FTOI) u.r.i[k] = vu_ftoi(s.f[k], d.scale);
            else u.r.u[k] = s.u[k] & 0x7FFFFFFFu;
        }
        return u;
    }
    case UK_CLIP: {
        const VUReg s = c.vf[d.fs];
        float w3 = std::fabs(ps2f(c.vf[d.ft].f[3]));
        uint32_t f = 0;
        for (int k = 0; k < 3; ++k) {
            if (ps2f(s.f[k]) > w3) f |= 1u << (2 * k);
            if (ps2f(s.f[k]) < -w3) f |= 2u << (2 * k);
        }
        u.clip = true;
        u.clip_bits = f;
        return u;
    }
    default: break;
    }
    if (d.bsrc == B_Q && p.q_busy) { ++vu.old_q; report(vu, pc, "Q"); }
    // The upper instruction writes only u.r: its operands can be read in place.
    const VUReg& s = c.vf[d.fs];
    const VUReg& tt = c.vf[d.ft];
    const VUReg& a = c.vacc;
    u.target = d.target;
    u.dest = dest;
    u.r = d.target == -1 ? c.vacc : d.target > 0 ? c.vf[d.target] : c.vf[0];
    Kind kind = (Kind)d.kind;
    int lm = d.lm;
    if (kind == K_OPMULA || kind == K_OPMSUB) {
        static const int P[3][2] = {{1, 2}, {2, 0}, {0, 1}};
        for (int k = 0; k < 3; ++k) {
            if (!(lm & (1 << k))) continue;
            float prod = vu_fmul(s.f[P[k][0]], tt.f[P[k][1]]);
            u.r.f[k] = kind == K_OPMULA ? prod : vu_fsub(a.f[k], prod);
        }
        u.flags = true;
        return u;
    }
    float bv[4];
    switch (d.bsrc) {
    case B_BC: bv[0] = bv[1] = bv[2] = bv[3] = tt.f[d.bc]; break;
    case B_Q: bv[0] = bv[1] = bv[2] = bv[3] = vu_q(c); break;
    case B_I: bv[0] = bv[1] = bv[2] = bv[3] = vu_i(c); break;
    default: std::memcpy(bv, tt.f, 16); break;
    }
#define LANES(expr) for (int k = 0; k < 4; ++k) if (lm & (1 << k)) { expr; }
    switch (kind) {
    case K_ADD: LANES(u.r.f[k] = vu_fadd(s.f[k], bv[k])) break;
    case K_SUB: LANES(u.r.f[k] = vu_fsub(s.f[k], bv[k])) break;
    case K_MUL: LANES(u.r.f[k] = vu_fmul(s.f[k], bv[k])) break;
    case K_MADD: LANES(float m = vu_fmul(s.f[k], bv[k]); u.r.f[k] = vu_fadd(a.f[k], m)) break;
    case K_MSUB: LANES(float m = vu_fmul(s.f[k], bv[k]); u.r.f[k] = vu_fsub(a.f[k], m)) break;
    case K_MAX: LANES(uint32_t bu; std::memcpy(&bu, &bv[k], 4); u.r.u[k] = vu_max(s.u[k], bu)) break;
    case K_MINI: LANES(uint32_t bu; std::memcpy(&bu, &bv[k], 4); u.r.u[k] = vu_min(s.u[k], bu)) break;
    default: break;
    }
#undef LANES
    u.flags = kind != K_MAX && kind != K_MINI;
    return u;
}

// The VF register a lower instruction writes (0 none), found before it runs
// so that a clash with the upper instruction can drop it.
int lower_vf_write(uint32_t w) {
    uint32_t op = w >> 25;
    int ft = (w >> 16) & 31;
    if (op == 0x00) return ft;                                   // lq
    if (op != 0x40 || (w & 0x3F) < 0x3C) return 0;
    uint32_t x = (((w >> 6) & 31) << 2) | (w & 3);
    switch (x) {
    case 0x30: case 0x31: case 0x34: case 0x36: case 0x3D: case 0x40: case 0x41: case 0x64: return ft;
    default: return 0;
    }
}
bool lower_writes_clip(uint32_t w) { return (w >> 25) == 0x11; }   // fcset

struct Backup { int reg = 0; uint32_t old = 0; uint64_t pair = ~0ull; };

int efu_latency(uint32_t x) {
    switch (x) {
    case 0x70: return 11; case 0x71: return 18; case 0x72: return 18; case 0x73: return 24;
    case 0x74: case 0x75: return 54; case 0x76: return 12; case 0x78: return 12; case 0x79: return 18;
    case 0x7A: return 12; case 0x7C: return 29; case 0x7D: return 54; case 0x7E: return 44;
    default: return 0;
    }
}

// The lower instruction. Returns true when it branches (target in `to`).
bool do_lower(VUUnit& vu, Pipe& p, uint32_t w, uint32_t pc, Backup& bk, Backup& nbk, uint32_t& to) {
    if (w == 0x8000033Cu) return false;                          // nop
    EEContext& c = *vu.c;
    uint32_t op = w >> 25;
    int it = (w >> 16) & 31, is = (w >> 11) & 31, id = (w >> 6) & 31, dest = lanes(w);
    int fsf = (w >> 21) & 3, ftf = (w >> 23) & 3;
    int32_t imm11 = (int32_t)(w & 0x7FF) - ((w & 0x400) ? 0x800 : 0);
    uint32_t mmask = vu.micro_mask;
    auto setvi = [&](int n, uint32_t v, bool ialu) {
        if (!n) return;
        if (ialu) nbk = Backup{n, c.vi[n], p.pair};
        c.vi[n] = v & 0xFFFFu;
    };
    // A branch reads a VI register as it was if the instruction just before wrote it.
    auto rd = [&](int n) -> int16_t {
        if (n && bk.reg == n && bk.pair + 1 == p.pair) {
            if (vu.warned.insert(pc | 0x10000).second)
                std::fprintf(stderr, "[vu%d] %04X: branch reads vi%d from before the previous instruction\n", vu.n, pc, n);
            return (int16_t)bk.old;
        }
        return (int16_t)c.vi[n];
    };
    auto flags_read = [&](const char* what) {
        if (p.fn) { ++vu.old_flags; report(vu, pc, what); }
    };
    uint32_t next = pc + 8;
    auto branch = [&](bool take) {
        if (take) to = (uint32_t)(next + imm11 * 8) & mmask;
        return take;
    };
    switch (op) {
    case 0x00: {                                                 // lq
        if (!it) return false;
        const VUReg* m = mem_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[it].u[k] = m->u[k];
        return false;
    }
    case 0x01: {                                                 // sq
        VUReg* m = mem_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[it] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vf[is].u[k];
        return false;
    }
    case 0x04: {                                                 // ilw: the lane named by dest
        const VUReg* m = mem_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) { setvi(it, m->u[k], false); break; }
        if (it && it < 16) p.vi_ready[it] = p.t + LAT_ILW;
        return false;
    }
    case 0x05: {                                                 // isw
        VUReg* m = mem_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vi[it] & 0xFFFFu;
        return false;
    }
    case 0x08: case 0x09: {                                      // iaddiu, isubiu
        uint32_t imm15 = ((w >> 10) & 0x7800) | (w & 0x7FF);
        setvi(it, op == 0x08 ? c.vi[is] + imm15 : c.vi[is] - imm15, true);
        return false;
    }
    case 0x10: case 0x12: case 0x13: case 0x1C: {                // fceq, fcand, fcor (to vi1); fcget
        flags_read("the clip flags");
        uint32_t cf = c.vi[VI_CLIP] & 0xFFFFFF, imm = w & 0xFFFFFF;
        if (op == 0x1C) setvi(it, cf & 0xFFF, false);
        else setvi(1, op == 0x10 ? cf == imm : op == 0x12 ? (cf & imm) != 0 : (cf | imm) == 0xFFFFFF, false);
        return false;
    }
    case 0x11: c.vi[VI_CLIP] = w & 0xFFFFFF; return false;       // fcset
    case 0x14: case 0x16: case 0x17: {                           // fseq, fsand, fsor
        flags_read("the status flags");
        uint32_t imm = (((w >> 21) & 1) << 11) | (w & 0x7FF);
        uint32_t st = c.vi[VI_STATUS] & 0xFFF;
        setvi(it, op == 0x14 ? st == imm : op == 0x16 ? st & imm : st | imm, false);
        return false;
    }
    case 0x15: {                                                 // fsset: the sticky bits
        uint32_t imm = (((w >> 21) & 1) << 11) | (w & 0x7FF);
        c.vi[VI_STATUS] = (imm & 0xFC0) | (c.vi[VI_STATUS] & 0x3F);
        return false;
    }
    case 0x18: case 0x1A: case 0x1B: {                           // fmeq, fmand, fmor
        flags_read("the MAC flags");
        uint32_t m = c.vi[VI_MAC] & 0xFFFF, s = c.vi[is] & 0xFFFF;
        setvi(it, op == 0x18 ? m == s : op == 0x1A ? m & s : m | s, false);
        return false;
    }
    case 0x20: return branch(true);                              // b
    case 0x21: setvi(it, (next + 8) / 8, false); return branch(true);   // bal
    case 0x24: to = (c.vi[is] * 8) & mmask; return true;         // jr
    case 0x25: {                                                 // jalr
        uint32_t target = (c.vi[is] * 8) & mmask;
        setvi(it, (next + 8) / 8, false);
        to = target;
        return true;
    }
    case 0x28: return branch(rd(it) == rd(is));                  // ibeq
    case 0x29: return branch(rd(it) != rd(is));                  // ibne
    case 0x2C: return branch(rd(is) < 0);                        // ibltz
    case 0x2D: return branch(rd(is) > 0);                        // ibgtz
    case 0x2E: return branch(rd(is) <= 0);                       // iblez
    case 0x2F: return branch(rd(is) >= 0);                       // ibgez
    case 0x40: break;
    default: rt_die("VU%d micro program at %04X: lower instruction %08X is not modelled", vu.n, pc, w);
    }
    uint32_t sub = w & 0x3F;
    if (sub < 0x3C) {
        switch (sub) {
        case 0x30: setvi(id, c.vi[is] + c.vi[it], true); return false;   // iadd
        case 0x31: setvi(id, c.vi[is] - c.vi[it], true); return false;   // isub
        case 0x32: setvi(it, c.vi[is] + (uint32_t)(id & 16 ? id - 32 : id), true); return false;   // iaddi
        case 0x34: setvi(id, c.vi[is] & c.vi[it], true); return false;   // iand
        case 0x35: setvi(id, c.vi[is] | c.vi[it], true); return false;   // ior
        default: rt_die("VU%d micro program at %04X: lower instruction %08X is not modelled", vu.n, pc, w);
        }
    }
    uint32_t x = (((w >> 6) & 31) << 2) | (sub & 3);
    int ft = it, fs = is;
    switch (x) {
    case 0x30: case 0x31: {                                      // move, mr32
        if (!ft) return false;
        const VUReg s = c.vf[fs];
        for (int k = 0; k < 4; ++k)
            if (lane(dest, k)) c.vf[ft].u[k] = s.u[x == 0x30 ? k : (k + 1) & 3];
        return false;
    }
    case 0x34: case 0x36: {                                      // lqi, lqd
        if (x == 0x36 && is) c.vi[is] = (c.vi[is] - 1) & 0xFFFFu;
        uint32_t at = c.vi[is];
        if (ft) {
            const VUReg* m = mem_q(vu, at, pc);
            for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].u[k] = m->u[k];
        }
        if (x == 0x34 && is) c.vi[is] = (c.vi[is] + 1) & 0xFFFFu;
        if (is) nbk = Backup{is, x == 0x34 ? (at & 0xFFFF) : ((at + 1) & 0xFFFF), p.pair};
        return false;
    }
    case 0x35: case 0x37: {                                      // sqi, sqd
        uint32_t old = c.vi[it];
        if (x == 0x37 && it) c.vi[it] = (c.vi[it] - 1) & 0xFFFFu;
        VUReg* m = mem_q(vu, c.vi[it], pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vf[fs].u[k];
        if (x == 0x35 && it) c.vi[it] = (c.vi[it] + 1) & 0xFFFFu;
        if (it) nbk = Backup{it, old, p.pair};
        return false;
    }
    case 0x38: case 0x39: case 0x3A: {                           // div, sqrt, rsqrt: into the FDIV pipeline
        vu_fdiv(c, x, c.vf[fs].f[fsf], c.vf[ft].f[ftf], p.q_val, p.q_flags);
        p.q_busy = true;
        p.q_ready = p.t + (x == 0x3A ? LAT_RSQRT : x == 0x39 ? LAT_SQRT : LAT_DIV);
        return false;
    }
    case 0x3B: return false;                                     // waitq: the stall did it
    case 0x3C: setvi(it, c.vf[fs].u[fsf], true); return false;   // mtir
    case 0x3D:                                                   // mfir
        if (!ft) return false;
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].i[k] = (int16_t)c.vi[is];
        return false;
    case 0x3E: {                                                 // ilwr
        const VUReg* m = mem_q(vu, c.vi[is], pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) { setvi(it, m->u[k], false); break; }
        if (it && it < 16) p.vi_ready[it] = p.t + LAT_ILW;
        return false;
    }
    case 0x3F: {                                                 // iswr
        VUReg* m = mem_q(vu, c.vi[is], pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vi[it] & 0xFFFFu;
        return false;
    }
    case 0x40: case 0x41: {                                      // rnext, rget: R as a float in [1, 2)
        if (x == 0x40) {
            uint32_t r = c.vi[VI_R];
            uint32_t bit = ((r >> 4) & 1) ^ ((r >> 22) & 1);
            c.vi[VI_R] = (((r << 1) ^ bit) & 0x7FFFFF) | 0x3F800000;
        }
        if (!ft) return false;
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].u[k] = c.vi[VI_R];
        return false;
    }
    case 0x42: c.vi[VI_R] = 0x3F800000 | (c.vf[fs].u[fsf] & 0x7FFFFF); return false;                  // rinit
    case 0x43: c.vi[VI_R] = 0x3F800000 | ((c.vi[VI_R] ^ c.vf[fs].u[fsf]) & 0x7FFFFF); return false;   // rxor
    default: break;
    }
    if (vu.n == 0) rt_die("VU0 micro program at %04X: lower instruction %08X is VU1's", pc, w);
    switch (x) {
    case 0x64:                                                   // mfp
        if (p.p_busy) { ++vu.old_p; report(vu, pc, "P"); }
        if (!ft) return false;
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].u[k] = vu.p;
        return false;
    case 0x68: setvi(it, vu.xtop() & 0x3FF, false); return false;      // xtop
    case 0x69: setvi(it, vu.xitop() & 0x3FF, false); return false;     // xitop
    case 0x6C: vu.xgkick(c.vi[is]); return false;                // xgkick
    case 0x7B: return false;                                     // waitp: the stall did it
    default:
        if (int lat = efu_latency(x)) {
            p.p_busy = true;
            p.p_val = vu_efu(x, c.vf[fs], fsf);
            p.p_ready = p.t + lat;
            return false;
        }
        rt_die("VU1 micro program at %04X: lower instruction %08X is not one", pc, w);
    }
}

}  // namespace

struct VUDecoded {
    uint64_t gen = ~0ull;
    Dec pair[2048];
};

namespace {


// Once per program: the decoded pairs, dropped if micro memory changed.
VUDecoded& decoded_table(VUUnit& vu) {
    if (!vu.dec) vu.dec = new VUDecoded;
    VUDecoded& t = *vu.dec;
    if (t.gen != vu.micro_gen) {
        for (Dec& d : t.pair) d.valid = false;
        t.gen = vu.micro_gen;
    }
    return t;
}

const Dec& decode(VUUnit& vu, Dec& d, uint32_t pc);

const Dec& decoded(VUUnit& vu, VUDecoded& t, uint32_t pc) {
    Dec& d = t.pair[pc / 8];
    if (EE_UNLIKELY(!d.valid)) return decode(vu, d, pc);
    return d;
}

const Dec& decode(VUUnit& vu, Dec& d, uint32_t pc) {
    std::memcpy(&d.lo, vu.micro + pc, 4);
    std::memcpy(&d.up, vu.micro + pc + 4, 4);
    if (d.up & 0x18000000u) rt_die("VU%d micro program at %04X: the D or T bit is not modelled yet", vu.n, pc);
    decode_upper(d, d.up, vu.n, pc);
    bool ibit = d.up & 0x80000000u;
    Reads r;
    upper_reads(d.up, r);
    uint32_t vi_read = 0;
    d.pipe = (uint8_t)(ibit ? P_NONE : lower_reads(d.lo, r, vi_read));
    d.vi_read = (uint16_t)vi_read;
    d.nrl = 0;
    d.rregs = 0;
    for (int k = 0; k < r.n; ++k) d.rregs |= 1u << r.reg[k];
    for (int k = 0; k < r.n; ++k)
        for (int l = 0; l < 4; ++l)
            if (r.mask[k] & (1 << l)) d.rl[d.nrl++] = (uint8_t)(r.reg[k] * 4 + l);
    d.lower = !ibit;
    d.lw = 0;
    if (!ibit) {
        int lw = lower_vf_write(d.lo);
        bool clash = (lw && lw == d.target) || (d.uk == UK_CLIP && lower_writes_clip(d.lo));
        if (clash) d.lower = false;
        else d.lw = (int8_t)lw;
        d.lwm = (uint8_t)lane_mask(lanes(d.lo));
    }
    d.valid = true;
    return d;
}

}  // namespace

void vu_exec(VUUnit& vu, uint32_t start) {
    EEContext& c = *vu.c;
    Pipe p;
    Backup bk, nbk;
    uint32_t mmask = vu.micro_mask;
    uint32_t pc = start & mmask, target = 0;
    bool delay = false, ending = false;
    ++vu.runs;
    VuDaz daz;
    VUDecoded& table = decoded_table(vu);
    for (uint64_t guard = 0;; ++guard) {
        if (guard > 10000000) rt_die("VU%d micro program from %04X: ten million pairs without an end", vu.n, start);
        const Dec& d = decoded(vu, table, pc);
        // Stalls: the VF lanes both halves read, then the lower one's own unit.
        if (d.rregs & p.in_flight())
            for (int k = 0; k < d.nrl; ++k) p.t = std::max(p.t, p.vf_ready[d.rl[k]]);
        if (d.pipe == P_FDIV && p.q_busy) p.t = std::max(p.t, p.q_ready);
        if (d.pipe == P_EFU && p.p_busy) p.t = std::max(p.t, p.p_ready);
        if (d.pipe == P_BRANCH)
            for (int n = 1; n < 16; ++n)
                if (d.vi_read & (1u << n)) p.t = std::max(p.t, p.vi_ready[n]);
        land(vu, p, false);

        bool in_delay = delay;
        delay = false;
        Upper u = do_upper(vu, p, d, pc);
        uint32_t to = 0;
        bool jumps = false;
        nbk = Backup{};
        if (d.lower) jumps = do_lower(vu, p, d.lo, pc, bk, nbk, to);
        if (u.target == -1) {
            for (int k = 0; k < 4; ++k) if (lane(u.dest, k)) c.vacc.u[k] = u.r.u[k];
        } else if (u.target > 0) {
            for (int k = 0; k < 4; ++k)
                if (d.lm & (1 << k)) { c.vf[u.target].u[k] = u.r.u[k]; p.vf_ready[u.target * 4 + k] = p.t + LAT_FMAC; }
            p.wrote(1u << u.target, p.t + LAT_FMAC);
        }
        if (d.lw) {
            for (int k = 0; k < 4; ++k) if (d.lwm & (1 << k)) p.vf_ready[d.lw * 4 + k] = p.t + LAT_FMAC;
            p.wrote(1u << d.lw, p.t + LAT_FMAC);
        }
        if (u.flags) {
            uint32_t mac = 0;
            for (int k = 0; k < 4; ++k) {
                if (!lane(u.dest, k)) continue;
                uint32_t bit = 8 >> k;                           // x is the high bit of each nibble
                if ((u.r.u[k] & 0x7FFFFFFFu) == 0) mac |= bit;
                if (u.r.u[k] & 0x80000000u) mac |= bit << 4;
            }
            p.push(FlagWrite{p.t + LAT_FMAC, false, mac, 0});
        }
        if (u.clip) p.push(FlagWrite{p.t + LAT_FMAC, true, 0, u.clip_bits});
        if (!d.lower && (d.up & 0x80000000u)) c.vi[VI_I] = d.lo;
        bk = nbk;
        ++p.t;
        ++p.pair;
        ++vu.pairs;
        if (ending) {
            if (jumps) rt_die("VU%d micro program at %04X: a branch in the E bit's delay slot", vu.n, pc);
            land(vu, p, true);
            vu.cycles += p.t;
            c.vi[VI_TPC] = ((in_delay ? target : pc + 8) & mmask) / 8;
            return;
        }
        if (d.up & 0x40000000u) ending = true;                   // E: one more pair
        if (jumps) {
            if (in_delay) rt_die("VU%d micro program at %04X: a branch in a branch's delay slot", vu.n, pc);
            delay = true;
            pc = (pc + 8) & mmask;
            target = to;
            continue;
        }
        pc = in_delay ? target : (pc + 8) & mmask;
    }
}

// ---- the translator: a VU1 program as C++ (vutrans) -------------------------------------------
// What the interpreter decides from the pipelines' timing (the stalls, when
// Q, P and the flags land, a branch's view of the VI register the pair before
// wrote) depends only on the pairs run and the way through the branches,
// never on the values. So the translator follows every way through a
// program with the pipelines' timing (TState: what is in flight, how many
// cycles from now it lands) instead of their values, the interpreter's rules
// applied to it pair by pair, and emits each pair for the timing it meets
// there: a pair reached with two timings is emitted twice. The code left
// does the arithmetic, the loads and stores, and moves the values in flight
// (Q, P, the flags) where the timing says, in the interpreter's order. The
// interpreter stays the reference: vubench runs both and compares.
namespace {

struct TState {                         // compared as bytes: no padding (pad)
    uint32_t pc;
    uint32_t target;                     // the delay slot's branch target (when not dyn)
    uint8_t vf[128];                     // cycles until each VF lane is written (0: written)
    uint8_t vi[16];                      // ILW, ILWR
    uint8_t nf, fhead;                   // flag writes in flight (in fq[0], fq[1]... oldest first); fhead: unused, 0
    uint8_t fready[8], fclip[8];         // oldest first: cycles until it lands (0: at the next pair); a clip judgement
    uint8_t q_busy, q_ready, p_busy, p_ready;
    uint8_t bk;                          // the VI register the pair before wrote through the integer unit (its old value: bk_old)
    uint8_t in_delay, dyn;               // in a branch's delay slot; its target in the variable `dyn_to`
    uint8_t ending;                      // the pair before had the E bit
    uint8_t pad[2];
};
static_assert(sizeof(TState) == 180, "TState has padding");

struct Emit {
    std::string s;
    void line(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char b[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(b, sizeof b, fmt, ap);
        va_end(ap);
        s += "    ";
        s += b;
        s += '\n';
    }
};

const char LN[4] = {'0', '1', '2', '3'};   // lanes by index (x 0 ... w 3)

// Whether the interpreter takes this pair without stopping (its decode and
// do_lower accept it).
bool pair_known(uint32_t lo, uint32_t up, std::string& why) {
    char b[96];
    std::snprintf(b, sizeof b, "pair %08X %08X", lo, up);
    why = b;
    if (up & 0x18000000u) return false;                          // D, T bits
    uint32_t op = up & 0x3F;
    if (op >= 0x30 && op < 0x3C) return false;
    if (op >= 0x3C) {
        uint32_t x = (((up >> 6) & 31) << 2) | (op & 3);
        if (x == 0x2B || x >= 0x30) return false;
    }
    if (up & 0x80000000u) return true;                           // the lower word is I's float
    if (lo == 0x8000033Cu) return true;
    uint32_t lop = lo >> 25;
    switch (lop) {
    case 0x00: case 0x01: case 0x04: case 0x05: case 0x08: case 0x09: case 0x10: case 0x11: case 0x12: case 0x13:
    case 0x14: case 0x15: case 0x16: case 0x17: case 0x18: case 0x1A: case 0x1B: case 0x1C: case 0x20: case 0x21:
    case 0x24: case 0x25: case 0x28: case 0x29: case 0x2C: case 0x2D: case 0x2E: case 0x2F:
        return true;
    case 0x40: break;
    default: return false;
    }
    uint32_t sub = lo & 0x3F;
    if (sub < 0x3C) return sub == 0x30 || sub == 0x31 || sub == 0x32 || sub == 0x34 || sub == 0x35;
    uint32_t x = (((lo >> 6) & 31) << 2) | (sub & 3);
    if ((x >= 0x30 && x <= 0x43 && x != 0x32 && x != 0x33) || x == 0x64 || x == 0x68 || x == 0x69 || x == 0x6C ||
        x == 0x7B)
        return true;
    return efu_latency(x) != 0;
}

bool is_cond_branch(uint32_t lo) { uint32_t op = lo >> 25; return op == 0x28 || op == 0x29 || (op >= 0x2C && op <= 0x2F); }

class Translator {
public:
    Translator(const uint8_t* micro) {
        std::memcpy(mic_, micro, sizeof mic_);
        vu_.n = 1;
        vu_.micro = mic_;
        vu_.micro_mask = 0x3FFF;
        vu_.mem_mask = 0x3FFF;
    }
    ~Translator() { delete vu_.dec; }

    bool run(uint32_t entry, const std::string& name, std::string& out, std::vector<uint32_t>& covered,
             std::string& why) {
        returns_.clear();
        if (!scan(entry, why)) return false;
        TState s0;
        std::memset(&s0, 0, sizeof s0);
        s0.pc = entry & 0x3FFF;
        std::vector<int> work{label(s0)};
        while (!work.empty()) {
            int n = work.back();
            work.pop_back();
            if (nodes_[n].done) continue;
            if (nodes_.size() > 60000) { why = "more than 60000 pairs once timed"; return false; }
            Emit e;
            std::vector<int> next;
            if (!pair(nodes_[n].st, e, next, why)) return false;
            nodes_[n].code = std::move(e.s);
            nodes_[n].done = true;
            for (auto it = next.rbegin(); it != next.rend(); ++it) work.push_back(*it);
        }
        // the function: labels in the order made, each falling into the next where it can.
        // The registers are its locals (indexed by constants: the compiler keeps
        // them in the host's registers), from VU1's and back at the end.
        std::string& o = out;
        o += "void " + name + "(VUUnit& vu) {\n";
        o += "    EEContext& c = *vu.c;\n    VuDaz daz;\n    ++vu.runs;\n";
        o += "    VUReg vf[32], acc = c.vacc;\n    uint32_t vi[32];\n";
        o += "    std::memcpy(vf, c.vf, sizeof vf);\n    std::memcpy(vi, c.vi, sizeof vi);\n";
        o += "    uint64_t np = 0, cyc = 0;\n    uint32_t fq[8] = {}, q_val = 0, q_fl = 0, p_val = 0, dyn_to = 0;\n";
        o += "    uint16_t bk_old = 0;\n    bool take = false;\n    VUReg ur;\n    uint32_t clipb = 0;\n";
        o += "    (void)fq; (void)q_val; (void)q_fl; (void)p_val; (void)dyn_to; (void)bk_old; (void)take; (void)ur; (void)clipb;\n";
        for (size_t n = 0; n < nodes_.size(); ++n) {
            char b[48];
            std::snprintf(b, sizeof b, "L%zu: // %04X\n", n, nodes_[n].st.pc);
            o += b;
            o += locals(nodes_[n].code);
        }
        o += "}\n";
        for (uint32_t pc : seen_pcs_) covered.push_back(pc);
        std::sort(covered.begin(), covered.end());
        return true;
    }

private:
    struct Node { TState st; std::string code; bool done = false; };
    uint8_t mic_[0x4000];
    VUUnit vu_{};
    std::vector<Node> nodes_;
    std::map<std::string, int> index_;
    std::set<uint32_t> returns_, seen_pcs_;

    // The emitted code names VU1's registers as do_upper and do_lower do;
    // here they become the function's locals.
    static std::string locals(std::string code) {
        static const char* const R[][2] = {
            {"vu_land_mac(c, ", "vu_land_mac_vi(vi, "}, {"vu_land_clip(c, ", "vu_land_clip_vi(vi, "},
            {"vu_divflags(c, ", "vu_divflags_vi(vi, "}, {"vu_q(c)", "vu_bits_f(vi[VI_Q])"},
            {"vu_i(c)", "vu_bits_f(vi[VI_I])"}, {"vu_fdiv(c, ", "vu_fdiv_vals("}, {"c.vf[", "vf["},
            {"c.vi[", "vi["}, {"c.vacc", "acc"},
            {"EPILOGUE;", "std::memcpy(c.vf, vf, sizeof vf); c.vacc = acc; std::memcpy(c.vi, vi, sizeof vi);"}};
        for (auto& r : R)
            for (size_t at = 0; (at = code.find(r[0], at)) != std::string::npos; at += std::strlen(r[1]))
                code.replace(at, std::strlen(r[0]), r[1]);
        return code;
    }

    int label(const TState& s) {
        std::string k((const char*)&s, sizeof s);
        auto it = index_.find(k);
        if (it != index_.end()) return it->second;
        nodes_.push_back(Node{s, {}, false});
        index_[k] = (int)nodes_.size() - 1;
        return (int)nodes_.size() - 1;
    }

    uint32_t lo_at(uint32_t pc) const { uint32_t w; std::memcpy(&w, mic_ + pc, 4); return w; }
    uint32_t up_at(uint32_t pc) const { uint32_t w; std::memcpy(&w, mic_ + pc + 4, 4); return w; }

    // Every pair reachable from entry, without timing: the return addresses
    // (after BAL's and JALR's delay slots) a JR or JALR may go back to.
    bool scan(uint32_t entry, std::string& why) {
        std::set<uint32_t> seen;
        std::vector<uint32_t> work{entry & 0x3FFF};
        bool has_jr = false;
        while (!work.empty()) {
            uint32_t pc = work.back();
            work.pop_back();
            if (!seen.insert(pc).second) continue;
            uint32_t lo = lo_at(pc), up = up_at(pc);
            if (!pair_known(lo, up, why)) continue;              // the interpreter stops there: so will the code
            bool ibit = up & 0x80000000u, ebit = up & 0x40000000u;
            uint32_t op = lo >> 25, next = (pc + 8) & 0x3FFF;
            int32_t imm11 = (int32_t)(lo & 0x7FF) - ((lo & 0x400) ? 0x800 : 0);
            uint32_t to = (uint32_t)(next + imm11 * 8) & 0x3FFF;
            if (!ibit && (op == 0x21 || op == 0x25)) returns_.insert((next + 8) & 0x3FFF);
            if (ebit) {                                          // the next pair ends it, after a branch its target's never
                work.push_back(next);
                continue;
            }
            if (!ibit && (op == 0x20 || op == 0x21)) { work.push_back(next); work.push_back(to); continue; }
            if (!ibit && (op == 0x24 || op == 0x25)) { has_jr = true; work.push_back(next); continue; }
            if (!ibit && is_cond_branch(lo)) { work.push_back(next); work.push_back(to); continue; }
            work.push_back(next);
        }
        if (has_jr) {
            if (returns_.empty()) { why = "JR without a BAL or JALR to return to"; return false; }
            // what follows a JR's delay slot: any return address, and what is reachable from those
            for (int round = 0; round < 8; ++round) {
                size_t before = seen.size();
                std::vector<uint32_t> more(returns_.begin(), returns_.end());
                for (uint32_t r : more) work.push_back(r);
                while (!work.empty()) {
                    uint32_t pc = work.back();
                    work.pop_back();
                    if (!seen.insert(pc).second) continue;
                    uint32_t lo = lo_at(pc), up = up_at(pc);
                    std::string w2;
                    if (!pair_known(lo, up, w2)) continue;
                    bool ibit = up & 0x80000000u;
                    uint32_t op = lo >> 25, next = (pc + 8) & 0x3FFF;
                    int32_t imm11 = (int32_t)(lo & 0x7FF) - ((lo & 0x400) ? 0x800 : 0);
                    if (!ibit && (op == 0x21 || op == 0x25)) returns_.insert((next + 8) & 0x3FFF);
                    if (up & 0x40000000u) { work.push_back(next); continue; }
                    if (!ibit && (op == 0x20 || op == 0x21 || is_cond_branch(lo)))
                        work.push_back((uint32_t)(next + imm11 * 8) & 0x3FFF);
                    work.push_back(next);
                }
                if (seen.size() == before) break;
            }
        }
        why.clear();
        return true;
    }

    // One pair at state s: its code into e, its successors' labels into next.
    bool pair(TState s, Emit& e, std::vector<int>& next, std::string& why) {
        uint32_t pc = s.pc;
        seen_pcs_.insert(pc);
        uint32_t lo = lo_at(pc), up = up_at(pc);
        if (!pair_known(lo, up, why)) {
            e.line("rt_die(\"VU1 micro program at %04X: pair %08X %08X stops the interpreter\");", pc, lo, up);
            return true;
        }
        VUDecoded& table = decoded_table(vu_);
        const Dec& d = decoded(vu_, table, pc);
        // the stall: this pair issues `st` cycles after it could
        int st = 0;
        for (int k = 0; k < d.nrl; ++k) st = std::max(st, (int)s.vf[d.rl[k]]);
        if (d.pipe == P_FDIV && s.q_busy) st = std::max(st, (int)s.q_ready);
        if (d.pipe == P_EFU && s.p_busy) st = std::max(st, (int)s.p_ready);
        if (d.pipe == P_BRANCH)
            for (int n = 1; n < 16; ++n)
                if (d.vi_read & (1u << n)) st = std::max(st, (int)s.vi[n]);
        // what lands by then
        // (the queue is positional: the oldest in fq[0], the others move up, so
        // that a loop does not meet a new state at every turn of a ring)
        while (s.nf && s.fready[0] <= st) {
            if (s.fclip[0]) e.line("vu_land_clip(c, fq[0]);");
            else e.line("vu_land_mac(c, fq[0]);");
            for (int k = 0; k + 1 < s.nf; ++k) {
                s.fready[k] = s.fready[k + 1];
                s.fclip[k] = s.fclip[k + 1];
                e.line("fq[%d] = fq[%d];", k, k + 1);
            }
            --s.nf;
            s.fready[s.nf] = s.fclip[s.nf] = 0;
        }
        if (s.q_busy && s.q_ready <= st) { e.line("c.vi[VI_Q] = q_val; vu_divflags(c, q_fl);"); s.q_busy = 0; s.q_ready = 0; }
        if (s.p_busy && s.p_ready <= st) { e.line("vu.p = p_val;"); s.p_busy = 0; s.p_ready = 0; }

        bool in_delay = s.in_delay, dyn = s.dyn;
        uint32_t target = s.target;
        // the upper instruction, into ur (clipb)
        bool flags = false, clip = false;
        upper(d, s, e, flags, clip);
        // the lower one
        int new_bk = 0;
        enum { J_NO, J_YES, J_COND, J_DYN } jump = J_NO;
        uint32_t to = 0;
        if (d.lower && !lower(d, s, st, e, new_bk, jump, to, why)) return false;
        // the upper one's result, its flags; I
        if (d.target == -1) {
            for (int k = 0; k < 4; ++k) if (lane(d.dest, k)) e.line("c.vacc.u[%c] = ur.u[%c];", LN[k], LN[k]);
        } else if (d.target > 0) {
            for (int k = 0; k < 4; ++k)
                if (d.lm & (1 << k)) e.line("c.vf[%d].u[%c] = ur.u[%c];", d.target, LN[k], LN[k]);
        }
        auto ready_vf = [&](int reg, int lm) {
            for (int k = 0; k < 4; ++k) if (lm & (1 << k)) s.vf[reg * 4 + k] = (uint8_t)(st + LAT_FMAC);
        };
        if (d.target > 0 && d.uk != UK_NOP) ready_vf(d.target, d.lm);
        if (d.lw) ready_vf(d.lw, d.lwm);
        if (flags || clip) {
            int slot = s.nf;
            if (flags) e.line("fq[%d] = vu_mac(ur, %d);", slot, d.dest);
            else e.line("fq[%d] = clipb;", slot);
            s.fready[s.nf] = (uint8_t)(st + LAT_FMAC);
            s.fclip[s.nf] = clip;
            ++s.nf;
        }
        if (!d.lower && (d.up & 0x80000000u)) e.line("c.vi[VI_I] = 0x%08Xu;", d.lo);
        e.line("++np; cyc += %d;", st + 1);
        // the next pair's timing: everything one cycle later than this pair issued
        int shift = st + 1;
        auto later = [&](uint8_t& v) { v = (uint8_t)std::max(0, (int)v - shift); };
        for (uint8_t& v : s.vf) later(v);
        for (uint8_t& v : s.vi) later(v);
        for (int k = 0; k < s.nf; ++k) later(s.fready[k]);
        if (s.q_busy) later(s.q_ready);
        if (s.p_busy) later(s.p_ready);
        s.bk = (uint8_t)new_bk;
        // where it goes
        if (s.ending) {
            if (jump != J_NO) { why = "a branch in the E bit's delay slot"; return false; }
            // everything in flight is written; TPC
            for (int k = 0; k < s.nf; ++k) {
                int slot = k;
                if (s.fclip[k]) e.line("vu_land_clip(c, fq[%d]);", slot);
                else e.line("vu_land_mac(c, fq[%d]);", slot);
            }
            if (s.q_busy) e.line("c.vi[VI_Q] = q_val; vu_divflags(c, q_fl);");
            if (s.p_busy) e.line("vu.p = p_val;");
            if (in_delay && dyn) e.line("c.vi[VI_TPC] = dyn_to / 8;");
            else e.line("c.vi[VI_TPC] = 0x%Xu;", ((in_delay ? target : pc + 8) & 0x3FFF) / 8);
            e.line("vu.pairs += np; vu.cycles += cyc;");
            e.line("EPILOGUE;");
            e.line("return;");
            return true;
        }
        TState n = s;
        n.ending = (d.up & 0x40000000u) != 0;
        n.in_delay = 0;
        n.dyn = 0;
        n.target = 0;
        if (jump != J_NO) {
            if (in_delay) { why = "a branch in a branch's delay slot"; return false; }
            TState t = n;
            t.pc = (pc + 8) & 0x3FFF;
            t.in_delay = 1;
            t.dyn = jump == J_DYN;
            t.target = jump == J_DYN ? 0 : to;
            int lt = label(t);
            if (jump == J_COND) {
                TState f = n;
                f.pc = (pc + 8) & 0x3FFF;
                int lf = label(f);
                e.line("if (take) goto L%d; else goto L%d;", lt, lf);
                next.push_back(lt);
                next.push_back(lf);
            } else {
                e.line("goto L%d;", lt);
                next.push_back(lt);
            }
            return true;
        }
        if (in_delay && dyn) {
            e.line("switch (dyn_to) {");
            for (uint32_t r : returns_) {
                TState t = n;
                t.pc = r;
                int l = label(t);
                e.line("case 0x%04X: goto L%d;", r, l);
                next.push_back(l);
            }
            e.line("default: rt_die(\"VU1 micro program: JR to %%04X, not an address after a BAL or JALR\", dyn_to);");
            e.line("}");
            return true;
        }
        n.pc = in_delay ? target : (pc + 8) & 0x3FFF;
        int l = label(n);
        e.line("goto L%d;", l);
        next.push_back(l);
        return true;
    }

    // do_upper, for one decoded pair, as code: the result into ur (or clipb).
    void upper(const Dec& d, const TState& s, Emit& e, bool& flags, bool& clip) {
        switch (d.uk) {
        case UK_NOP: return;
        case UK_ITOF: case UK_FTOI: case UK_ABS: {
            e.line("{ const VUReg sv = c.vf[%d]; ur = c.vf[%d];", d.fs, d.target);
            for (int k = 0; k < 4; ++k) {
                if (!lane(d.dest, k)) continue;
                if (d.uk == UK_ITOF) e.line("  ur.f[%c] = ps2f((float)sv.i[%c] / %.1ff);", LN[k], LN[k], d.scale);
                else if (d.uk == UK_FTOI) e.line("  ur.i[%c] = vu_ftoi(sv.f[%c], %.1ff);", LN[k], LN[k], d.scale);
                else e.line("  ur.u[%c] = sv.u[%c] & 0x7FFFFFFFu;", LN[k], LN[k]);
            }
            e.line("}");
            return;
        }
        case UK_CLIP:
            e.line("{ const VUReg sv = c.vf[%d]; float w3 = std::fabs(ps2f(c.vf[%d].f[3])); clipb = 0;", d.fs, d.ft);
            for (int k = 0; k < 3; ++k)
                e.line("  if (ps2f(sv.f[%d]) > w3) clipb |= %uu; if (ps2f(sv.f[%d]) < -w3) clipb |= %uu;", k, 1u << (2 * k),
                       k, 2u << (2 * k));
            e.line("}");
            clip = true;
            return;
        default: break;
        }
        // the old Q's readers are counted, as the interpreter counts them
        if (d.bsrc == B_Q && s.q_busy) e.line("++vu.old_q;");
        e.line("{ const VUReg& sv = c.vf[%d]; const VUReg& tv = c.vf[%d]; const VUReg& av = c.vacc;", d.fs, d.ft);
        e.line("  (void)sv; (void)tv; (void)av;");
        if (d.target == -1) e.line("  ur = c.vacc;");
        else e.line("  ur = c.vf[%d];", d.target > 0 ? d.target : 0);
        Kind kind = (Kind)d.kind;
        int lm = d.lm;
        if (kind == K_OPMULA || kind == K_OPMSUB) {
            static const int P[3][2] = {{1, 2}, {2, 0}, {0, 1}};
            for (int k = 0; k < 3; ++k) {
                if (!(lm & (1 << k))) continue;
                if (kind == K_OPMULA) e.line("  ur.f[%d] = vu_fmul(sv.f[%d], tv.f[%d]);", k, P[k][0], P[k][1]);
                else e.line("  ur.f[%d] = vu_fsub(av.f[%d], vu_fmul(sv.f[%d], tv.f[%d]));", k, k, P[k][0], P[k][1]);
            }
            e.line("}");
            flags = true;
            return;
        }
        // ADD, SUB, MUL, MADD, MSUB: the four lanes at once when no operand
        // has exponent 255 (the results are then finite: rounding toward zero),
        // each lane as the interpreter computes it otherwise
        bool vec = kind == K_ADD || kind == K_SUB || kind == K_MUL || kind == K_MADD || kind == K_MSUB;
        if (vec) {
            const char* B = d.bsrc == B_BC ? nullptr : d.bsrc == B_Q ? "_mm_set1_ps(vu_q(c))"
                          : d.bsrc == B_I ? "_mm_set1_ps(vu_i(c))" : "_mm_loadu_ps(tv.f)";
            if (B) e.line("  __m128 S = _mm_loadu_ps(sv.f), B = %s;", B);
            else e.line("  __m128 S = _mm_loadu_ps(sv.f), B = _mm_set1_ps(tv.f[%d]);", d.bc);
            bool acc_in = kind == K_MADD || kind == K_MSUB;
            if (acc_in) e.line("  __m128 A = _mm_loadu_ps(av.f);");
            e.line("  if (!EE_UNLIKELY(vu_any255(S) | vu_any255(B)%s)) {", acc_in ? " | vu_any255(A)" : "");
            const char* R = kind == K_ADD ? "_mm_add_ps(S, B)" : kind == K_SUB ? "_mm_sub_ps(S, B)"
                          : kind == K_MUL ? "_mm_mul_ps(S, B)" : kind == K_MADD ? "_mm_add_ps(A, _mm_mul_ps(S, B))"
                          : "_mm_sub_ps(A, _mm_mul_ps(S, B))";
            if (lm == 15) e.line("    _mm_storeu_ps(ur.f, %s);", R);
            else {
                e.line("    alignas(16) float rr[4]; _mm_store_ps(rr, %s);", R);
                for (int k = 0; k < 4; ++k) if (lm & (1 << k)) e.line("    ur.f[%d] = rr[%d];", k, k);
            }
            e.line("  } else {");
        }
        // the second operand, per lane
        char bf[4][32], bu[4][32];
        for (int k = 0; k < 4; ++k) {
            switch (d.bsrc) {
            case B_BC: std::snprintf(bf[k], 32, "tv.f[%d]", d.bc); std::snprintf(bu[k], 32, "tv.u[%d]", d.bc); break;
            case B_Q: std::snprintf(bf[k], 32, "vu_q(c)"); std::snprintf(bu[k], 32, "c.vi[VI_Q]"); break;
            case B_I: std::snprintf(bf[k], 32, "vu_i(c)"); std::snprintf(bu[k], 32, "c.vi[VI_I]"); break;
            default: std::snprintf(bf[k], 32, "tv.f[%d]", k); std::snprintf(bu[k], 32, "tv.u[%d]", k); break;
            }
        }
        for (int k = 0; k < 4; ++k) {
            if (!(lm & (1 << k))) continue;
            switch (kind) {
            case K_ADD: e.line("  ur.f[%d] = vu_fadd(sv.f[%d], %s);", k, k, bf[k]); break;
            case K_SUB: e.line("  ur.f[%d] = vu_fsub(sv.f[%d], %s);", k, k, bf[k]); break;
            case K_MUL: e.line("  ur.f[%d] = vu_fmul(sv.f[%d], %s);", k, k, bf[k]); break;
            case K_MADD: e.line("  ur.f[%d] = vu_fadd(av.f[%d], vu_fmul(sv.f[%d], %s));", k, k, k, bf[k]); break;
            case K_MSUB: e.line("  ur.f[%d] = vu_fsub(av.f[%d], vu_fmul(sv.f[%d], %s));", k, k, k, bf[k]); break;
            case K_MAX: e.line("  ur.u[%d] = vu_max(sv.u[%d], %s);", k, k, bu[k]); break;
            case K_MINI: e.line("  ur.u[%d] = vu_min(sv.u[%d], %s);", k, k, bu[k]); break;
            default: break;
            }
        }
        if (vec) e.line("  }");
        e.line("}");
        flags = kind != K_MAX && kind != K_MINI;
    }

    // do_lower, as code. new_bk: the VI register it writes through the
    // integer unit (its old value saved in bk_old). jump: whether it branches.
    template <class J>
    bool lower(const Dec& d, TState& s, int st, Emit& e, int& new_bk, J& jump, uint32_t& to, std::string& why) {
        uint32_t w = d.lo, pc = s.pc;
        if (w == 0x8000033Cu) return true;
        uint32_t op = w >> 25;
        int it = (w >> 16) & 31, is = (w >> 11) & 31, id = (w >> 6) & 31, dest = lanes(w);
        int fsf = (w >> 21) & 3, ftf = (w >> 23) & 3;
        int32_t imm11 = (int32_t)(w & 0x7FF) - ((w & 0x400) ? 0x800 : 0);
        uint32_t next = pc + 8;
        // setvi: through the integer unit (ialu) the old value is kept for a branch next
        auto setvi = [&](int n, const std::string& v, bool ialu) {
            if (!n) return;
            if (ialu) { e.line("bk_old = (uint16_t)c.vi[%d];", n); new_bk = n; }
            e.line("c.vi[%d] = (%s) & 0xFFFFu;", n, v.c_str());
        };
        auto vi = [&](int n) { return "c.vi[" + std::to_string(n) + "]"; };
        auto rd = [&](int n) -> std::string {
            if (n && s.bk == n) return "(int16_t)bk_old";
            return "(int16_t)c.vi[" + std::to_string(n) + "]";
        };
        auto flags_read = [&]() { if (s.nf) e.line("++vu.old_flags;"); };
        auto lanes_of = [&](auto fn) { for (int k = 0; k < 4; ++k) if (lane(dest, k)) fn(k); };
        char b[256];
        switch (op) {
        case 0x00:                                                   // lq
            if (!it) return true;
            e.line("{ const VUReg* m = vu1_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[%d] + %d));", is, imm11);
            lanes_of([&](int k) { e.line("  c.vf[%d].u[%d] = m->u[%d];", it, k, k); });
            e.line("}");
            return true;
        case 0x01:                                                   // sq
            e.line("{ VUReg* m = vu1_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[%d] + %d));", it, imm11);
            lanes_of([&](int k) { e.line("  m->u[%d] = c.vf[%d].u[%d];", k, is, k); });
            e.line("}");
            return true;
        case 0x04: {                                                 // ilw
            e.line("{ const VUReg* m = vu1_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[%d] + %d));", is, imm11);
            for (int k = 0; k < 4; ++k)
                if (lane(dest, k)) { if (it) e.line("  c.vi[%d] = m->u[%d] & 0xFFFFu;", it, k); break; }
            e.line("}");
            if (it && it < 16) s.vi[it] = (uint8_t)(st + LAT_ILW);
            return true;
        }
        case 0x05:                                                   // isw
            e.line("{ VUReg* m = vu1_q(vu, (uint32_t)((int32_t)(int16_t)c.vi[%d] + %d));", is, imm11);
            lanes_of([&](int k) { e.line("  m->u[%d] = c.vi[%d] & 0xFFFFu;", k, it); });
            e.line("}");
            return true;
        case 0x08: case 0x09: {                                      // iaddiu, isubiu
            uint32_t imm15 = ((w >> 10) & 0x7800) | (w & 0x7FF);
            std::snprintf(b, sizeof b, "c.vi[%d] %c %uu", is, op == 0x08 ? '+' : '-', imm15);
            setvi(it, b, true);
            return true;
        }
        case 0x10: case 0x12: case 0x13: case 0x1C: {                // fceq, fcand, fcor; fcget
            flags_read();
            uint32_t imm = w & 0xFFFFFF;
            if (op == 0x1C) setvi(it, "c.vi[VI_CLIP] & 0xFFF", false);
            else {
                if (op == 0x10) std::snprintf(b, sizeof b, "(c.vi[VI_CLIP] & 0xFFFFFF) == 0x%Xu", imm);
                else if (op == 0x12) std::snprintf(b, sizeof b, "(c.vi[VI_CLIP] & 0x%Xu) != 0", imm);
                else std::snprintf(b, sizeof b, "((c.vi[VI_CLIP] & 0xFFFFFF) | 0x%Xu) == 0xFFFFFF", imm);
                setvi(1, b, false);
            }
            return true;
        }
        case 0x11: e.line("c.vi[VI_CLIP] = 0x%Xu;", w & 0xFFFFFF); return true;   // fcset
        case 0x14: case 0x16: case 0x17: {                           // fseq, fsand, fsor
            flags_read();
            uint32_t imm = (((w >> 21) & 1) << 11) | (w & 0x7FF);
            if (op == 0x14) std::snprintf(b, sizeof b, "(c.vi[VI_STATUS] & 0xFFF) == 0x%Xu", imm);
            else if (op == 0x16) std::snprintf(b, sizeof b, "c.vi[VI_STATUS] & 0x%Xu", imm & 0xFFF);
            else std::snprintf(b, sizeof b, "(c.vi[VI_STATUS] & 0xFFF) | 0x%Xu", imm);
            setvi(it, b, false);
            return true;
        }
        case 0x15: {                                                 // fsset
            uint32_t imm = (((w >> 21) & 1) << 11) | (w & 0x7FF);
            e.line("c.vi[VI_STATUS] = 0x%Xu | (c.vi[VI_STATUS] & 0x3F);", imm & 0xFC0);
            return true;
        }
        case 0x18: case 0x1A: case 0x1B: {                           // fmeq, fmand, fmor
            flags_read();
            if (op == 0x18) std::snprintf(b, sizeof b, "(c.vi[VI_MAC] & 0xFFFF) == (c.vi[%d] & 0xFFFF)", is);
            else if (op == 0x1A) std::snprintf(b, sizeof b, "(c.vi[VI_MAC] & 0xFFFF) & (c.vi[%d] & 0xFFFF)", is);
            else std::snprintf(b, sizeof b, "(c.vi[VI_MAC] & 0xFFFF) | (c.vi[%d] & 0xFFFF)", is);
            setvi(it, b, false);
            return true;
        }
        case 0x20: jump = J::J_YES; to = (uint32_t)(next + imm11 * 8) & 0x3FFF; return true;   // b
        case 0x21:                                                   // bal
            std::snprintf(b, sizeof b, "%uu", (next + 8) / 8);
            setvi(it, b, false);
            jump = J::J_YES;
            to = (uint32_t)(next + imm11 * 8) & 0x3FFF;
            return true;
        case 0x24: e.line("dyn_to = (c.vi[%d] * 8) & 0x3FFF;", is); jump = J::J_DYN; return true;   // jr
        case 0x25:                                                   // jalr
            e.line("dyn_to = (c.vi[%d] * 8) & 0x3FFF;", is);
            std::snprintf(b, sizeof b, "%uu", (next + 8) / 8);
            setvi(it, b, false);
            jump = J::J_DYN;
            return true;
        case 0x28: case 0x29: case 0x2C: case 0x2D: case 0x2E: case 0x2F: {
            std::string cnd;
            if (op == 0x28) cnd = rd(it) + " == " + rd(is);
            else if (op == 0x29) cnd = rd(it) + " != " + rd(is);
            else if (op == 0x2C) cnd = rd(is) + " < 0";
            else if (op == 0x2D) cnd = rd(is) + " > 0";
            else if (op == 0x2E) cnd = rd(is) + " <= 0";
            else cnd = rd(is) + " >= 0";
            e.line("take = %s;", cnd.c_str());
            jump = J::J_COND;
            to = (uint32_t)(next + imm11 * 8) & 0x3FFF;
            return true;
        }
        case 0x40: break;
        default: why = "lower instruction"; return false;
        }
        uint32_t sub = w & 0x3F;
        if (sub < 0x3C) {
            switch (sub) {
            case 0x30: setvi(id, vi(is) + " + " + vi(it), true); return true;      // iadd
            case 0x31: setvi(id, vi(is) + " - " + vi(it), true); return true;      // isub
            case 0x32:                                                             // iaddi
                std::snprintf(b, sizeof b, "c.vi[%d] + %uu", is, (uint32_t)(id & 16 ? id - 32 : id));
                setvi(it, b, true);
                return true;
            case 0x34: setvi(id, vi(is) + " & " + vi(it), true); return true;      // iand
            case 0x35: setvi(id, vi(is) + " | " + vi(it), true); return true;      // ior
            default: why = "lower instruction"; return false;
            }
        }
        uint32_t x = (((w >> 6) & 31) << 2) | (sub & 3);
        int ft = it, fs = is;
        switch (x) {
        case 0x30: case 0x31:                                        // move, mr32
            if (!ft) return true;
            e.line("{ const VUReg sv = c.vf[%d];", fs);
            lanes_of([&](int k) { e.line("  c.vf[%d].u[%d] = sv.u[%d];", ft, k, x == 0x30 ? k : (k + 1) & 3); });
            e.line("}");
            return true;
        case 0x34: case 0x36:                                        // lqi, lqd
            e.line("{");
            if (x == 0x36 && is) e.line("  c.vi[%d] = (c.vi[%d] - 1) & 0xFFFFu;", is, is);
            e.line("  uint32_t at = c.vi[%d];", is);
            if (ft) {
                e.line("  const VUReg* m = vu1_q(vu, at);");
                lanes_of([&](int k) { e.line("  c.vf[%d].u[%d] = m->u[%d];", ft, k, k); });
            }
            if (x == 0x34 && is) e.line("  c.vi[%d] = (c.vi[%d] + 1) & 0xFFFFu;", is, is);
            if (is) {
                e.line("  bk_old = (uint16_t)%s;", x == 0x34 ? "(at & 0xFFFF)" : "((at + 1) & 0xFFFF)");
                new_bk = is;
            }
            e.line("  (void)at; }");
            return true;
        case 0x35: case 0x37:                                        // sqi, sqd
            e.line("{ uint32_t old = c.vi[%d];", it);
            if (x == 0x37 && it) e.line("  c.vi[%d] = (c.vi[%d] - 1) & 0xFFFFu;", it, it);
            e.line("  VUReg* m = vu1_q(vu, c.vi[%d]);", it);
            lanes_of([&](int k) { e.line("  m->u[%d] = c.vf[%d].u[%d];", k, fs, k); });
            if (x == 0x35 && it) e.line("  c.vi[%d] = (c.vi[%d] + 1) & 0xFFFFu;", it, it);
            if (it) { e.line("  bk_old = (uint16_t)old;"); new_bk = it; }
            e.line("  (void)old; }");
            return true;
        case 0x38: case 0x39: case 0x3A:                             // div, sqrt, rsqrt
            e.line("vu_fdiv(c, 0x%Xu, c.vf[%d].f[%d], c.vf[%d].f[%d], q_val, q_fl);", x, fs, fsf, ft, ftf);
            s.q_busy = 1;
            s.q_ready = (uint8_t)(st + (x == 0x3A ? LAT_RSQRT : x == 0x39 ? LAT_SQRT : LAT_DIV));
            return true;
        case 0x3B: return true;                                      // waitq
        case 0x3C:                                                   // mtir
            std::snprintf(b, sizeof b, "c.vf[%d].u[%d]", fs, fsf);
            setvi(it, b, true);
            return true;
        case 0x3D:                                                   // mfir
            if (!ft) return true;
            lanes_of([&](int k) { e.line("c.vf[%d].i[%d] = (int16_t)c.vi[%d];", ft, k, is); });
            return true;
        case 0x3E: {                                                 // ilwr
            e.line("{ const VUReg* m = vu1_q(vu, c.vi[%d]);", is);
            for (int k = 0; k < 4; ++k)
                if (lane(dest, k)) { if (it) e.line("  c.vi[%d] = m->u[%d] & 0xFFFFu;", it, k); break; }
            e.line("}");
            if (it && it < 16) s.vi[it] = (uint8_t)(st + LAT_ILW);
            return true;
        }
        case 0x3F:                                                   // iswr
            e.line("{ VUReg* m = vu1_q(vu, c.vi[%d]);", is);
            lanes_of([&](int k) { e.line("  m->u[%d] = c.vi[%d] & 0xFFFFu;", k, it); });
            e.line("}");
            return true;
        case 0x40: case 0x41:                                        // rnext, rget
            if (x == 0x40) {
                e.line("{ uint32_t r = c.vi[VI_R], bit = ((r >> 4) & 1) ^ ((r >> 22) & 1);");
                e.line("  c.vi[VI_R] = (((r << 1) ^ bit) & 0x7FFFFF) | 0x3F800000; }");
            }
            if (!ft) return true;
            lanes_of([&](int k) { e.line("c.vf[%d].u[%d] = c.vi[VI_R];", ft, k); });
            return true;
        case 0x42: e.line("c.vi[VI_R] = 0x3F800000 | (c.vf[%d].u[%d] & 0x7FFFFF);", fs, fsf); return true;   // rinit
        case 0x43: e.line("c.vi[VI_R] = 0x3F800000 | ((c.vi[VI_R] ^ c.vf[%d].u[%d]) & 0x7FFFFF);", fs, fsf); return true;
        case 0x64:                                                   // mfp
            if (s.p_busy) e.line("++vu.old_p;");
            if (!ft) return true;
            lanes_of([&](int k) { e.line("c.vf[%d].u[%d] = vu.p;", ft, k); });
            return true;
        case 0x68: setvi(it, "vu.xtop() & 0x3FF", false); return true;     // xtop
        case 0x69: setvi(it, "vu.xitop() & 0x3FF", false); return true;    // xitop
        case 0x6C: e.line("vu.xgkick(c.vi[%d]);", is); return true;        // xgkick
        case 0x7B: return true;                                      // waitp
        default:
            if (int lat = efu_latency(x)) {
                e.line("p_val = vu_efu(0x%Xu, c.vf[%d], %d);", x, fs, fsf);
                s.p_busy = 1;
                s.p_ready = (uint8_t)(st + lat);
                return true;
            }
            why = "lower instruction";
            return false;
        }
    }
};

}  // namespace

bool vu1_translate(const uint8_t* micro, uint32_t entry, const std::string& name, std::string& out,
                   std::vector<uint32_t>& covered, std::string& why) {
    Translator t(micro);
    return t.run(entry, name, out, covered, why);
}

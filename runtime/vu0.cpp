// ps2kit runtime — VU0 in micro mode, and VIF0 in front of it.
//
// VIF0 takes the codes DMA channel 0 (or the VIF0 FIFO) brings: NOP, STCYCL,
// STMASK, STMOD, STROW, STCOL, MARK, ITOP, FLUSHE are kept or ignored (VU0
// is idle whenever VIF0 runs, see below); MPG writes microcode into VU0's
// 4 KB micro memory. Anything else stops the run until a game needs it.
//
// A micro program (vcallms, vcallmsr) runs at once, to its end, on the
// calling thread's VU0 registers (the vf/vi/acc of EEContext): VU0 is
// infinitely fast here, so the EE's interlocked reads (qmfc2.i, cfc2.i) and
// the next vcallms always find it finished.
//
// Each step is one instruction pair (upper word above the lower one). The
// pair's two halves read the registers as they were before the pair; when
// both write the same VF register (or both the clip flags), the lower
// instruction is dropped, and the upper one's flags are written after the
// lower's. With the I bit the lower word is loaded into I after the upper
// instruction ran. Branches have one delay slot; the E bit ends the program
// after the next pair. A conditional branch right after the integer
// instruction that wrote its register reads the register's older value
// (said once per address when it happens).
// (These pair rules are those of PCSX2's VU0 interpreter.)
//
// Results are produced in program order, as if every instruction completed
// before the next began. On the console the FMAC pipeline takes 4 cycles,
// DIV and SQRT 7, RSQRT 13, and the VU stalls on a register that is not
// ready, so register values come out the same; what does not stall is a
// read of the flags (MAC, status, clip) or of Q, which sees only the
// instructions already completed. The runtime counts pairs and says once
// per address when a program reads flags or Q fewer pairs after their
// producer than the console's latency: there, the console may see an older
// value than this model. (Stalls only add cycles, so a read that passes this
// check sees the same value on the console.) DW2's trigonometry library
// waits exactly 4 pairs after CLIP and FMAC and 7 after DIV.
#include "os.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

uint8_t g_micro[0x1000];                 // VU0 micro memory: 512 instruction pairs
uint64_t g_runs, g_pairs;

// ---- VIF0 -----------------------------------------------------------------------------------
struct Vif0 {
    std::vector<uint8_t> fifo;           // bytes received, not yet a whole code with its data
    uint32_t cycle = 0, mask = 0, mode = 0, itop = 0, mark = 0;
    uint32_t row[4] = {}, col[4] = {};
    uint64_t mpg_bytes = 0;
} g_vif0;

uint32_t word_at(const std::vector<uint8_t>& b, size_t at) {
    uint32_t v;
    std::memcpy(&v, &b[at], 4);
    return v;
}

// Codes with their data, as many as the FIFO holds whole.
void vif0_run() {
    Vif0& v = g_vif0;
    size_t p = 0;
    while (p + 4 <= v.fifo.size()) {
        uint32_t code = word_at(v.fifo, p);
        uint32_t cmd = (code >> 24) & 0x7F, num = (code >> 16) & 0xFF, imm = code & 0xFFFF;
        size_t data = 0;
        if (cmd == 0x20) data = 4;
        else if (cmd == 0x30 || cmd == 0x31) data = 16;
        else if (cmd == 0x4A) data = (size_t)(num ? num : 256) * 8;
        if (p + 4 + data > v.fifo.size()) break;
        const uint8_t* d = v.fifo.data() + p + 4;
        if (code & 0x80000000u) rt_die("VIF0: code %08X with its interrupt bit is not modelled yet", code);
        switch (cmd) {
        case 0x00: break;                                        // NOP
        case 0x01: v.cycle = imm; break;                         // STCYCL
        case 0x04: v.itop = imm & 0x3FF; break;                  // ITOP
        case 0x05: v.mode = imm & 3; break;                      // STMOD
        case 0x07: v.mark = imm; break;                          // MARK
        case 0x10: break;                                        // FLUSHE: VU0 is never running here
        case 0x20: std::memcpy(&v.mask, d, 4); break;            // STMASK
        case 0x30: std::memcpy(v.row, d, 16); break;             // STROW
        case 0x31: std::memcpy(v.col, d, 16); break;             // STCOL
        case 0x4A: {                                             // MPG: num pairs at imm (in pairs)
            uint32_t at = (imm * 8) & 0xFFF;
            if (at + data > sizeof g_micro) rt_die("VIF0: MPG of %zu bytes at %04X beyond micro memory", data, at);
            std::memcpy(g_micro + at, d, data);
            v.mpg_bytes += data;
            TRACE(1, "[vif0] MPG %zu instructions at %04X\n", data / 8, at);
            break;
        }
        default:
            rt_die("VIF0: code %08X (command %02X) is not modelled yet", code, cmd);
        }
        p += 4 + data;
    }
    v.fifo.erase(v.fifo.begin(), v.fifo.begin() + (std::ptrdiff_t)p);
}

// ---- the micro program -------------------------------------------------------------------------
// When each flag group and Q become visible, in pairs (see the top).
struct Timing {
    uint64_t pair = 0;
    uint64_t mac = 0, st_fmac = 0, st_div = 0, clip = 0, q = 0;
};
std::set<uint32_t> g_warned;

void early(uint32_t pc, const char* what, uint64_t pair, uint64_t ready) {
    if (!g_warned.insert(pc).second) return;
    std::fprintf(stderr,
                 "ps2kit: VU0 micro program at %04X reads %s %llu pairs before the console would see it: "
                 "the console may read an older value\n",
                 pc, what, (unsigned long long)(ready - pair));
}

const int LAT_FMAC = 4, LAT_DIV = 7, LAT_SQRT = 7, LAT_RSQRT = 13;
// Status bits by producer: FMAC writes Z S U O and their sticky copies, DIV/SQRT/RSQRT I D and theirs.
const uint32_t ST_FMAC = 0x3CF, ST_DIV = 0xC30;

int lanes(uint32_t w) { return (w >> 21) & 15; }
bool lane(int dest, int k) { return dest & (8 >> k); }

VUReg* vu0_mem_q(uint32_t qaddr, uint32_t pc) {
    uint32_t a = qaddr * 16;
    if (a & 0x4000) rt_die("VU0 micro program at %04X: access to VU1's registers (%04X) is not modelled yet", pc, a);
    return (VUReg*)(g_vu0_mem + (a & 0xFFF));
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

enum Kind { K_ADD, K_SUB, K_MUL, K_MADD, K_MSUB, K_MAX, K_MINI, K_OPMULA, K_OPMSUB };

Upper do_upper(EEContext& c, uint32_t w, uint32_t pc, Timing& t) {
    Upper u;
    uint32_t op = w & 0x3F;
    int ft = (w >> 16) & 31, fs = (w >> 11) & 31, fd = (w >> 6) & 31, dest = lanes(w), bc = w & 3;
    Kind kind;
    enum { B_BC, B_Q, B_I, B_FT } bsrc;
    bool to_acc = false;
    if (op < 0x1C) {
        static const Kind K[7] = {K_ADD, K_SUB, K_MADD, K_MSUB, K_MAX, K_MINI, K_MUL};
        kind = K[op >> 2];
        bsrc = B_BC;
    } else if (op < 0x30) {
        static const struct { Kind k; int b; } T[20] = {
            {K_MUL, B_Q}, {K_MAX, B_I}, {K_MUL, B_I}, {K_MINI, B_I}, {K_ADD, B_Q}, {K_MADD, B_Q}, {K_ADD, B_I},
            {K_MADD, B_I}, {K_SUB, B_Q}, {K_MSUB, B_Q}, {K_SUB, B_I}, {K_MSUB, B_I}, {K_ADD, B_FT}, {K_MADD, B_FT},
            {K_MUL, B_FT}, {K_MAX, B_FT}, {K_SUB, B_FT}, {K_MSUB, B_FT}, {K_OPMSUB, B_FT}, {K_MINI, B_FT}};
        kind = T[op - 0x1C].k;
        bsrc = (decltype(bsrc))T[op - 0x1C].b;
    } else if (op >= 0x3C) {
        uint32_t x = (((w >> 6) & 31) << 2) | (op & 3);
        to_acc = true;
        if (x < 0x10) {
            static const Kind K[4] = {K_ADD, K_SUB, K_MADD, K_MSUB};
            kind = K[x >> 2];
            bsrc = B_BC;
        } else if (x < 0x18) {                                   // itof0-15, ftoi0-15: no flags
            if (!ft) return u;
            const VUReg s = c.vf[fs];
            u.target = ft;
            u.dest = dest;
            u.r = c.vf[ft];
            static const float SCALE[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
            float scale = SCALE[x & 3];
            for (int k = 0; k < 4; ++k)
                if (lane(dest, k)) {
                    if (x < 0x14) u.r.f[k] = ps2f((float)s.i[k] / scale);
                    else u.r.i[k] = vu_ftoi(s.f[k], scale);
                }
            return u;
        } else if (x < 0x1C) {
            kind = K_MUL;
            bsrc = B_BC;
        } else {
            switch (x) {
            case 0x1C: kind = K_MUL; bsrc = B_Q; break;
            case 0x1D: {                                         // abs
                if (!ft) return u;
                const VUReg s = c.vf[fs];
                u.target = ft;
                u.dest = dest;
                u.r = c.vf[ft];
                for (int k = 0; k < 4; ++k)
                    if (lane(dest, k)) u.r.u[k] = s.u[k] & 0x7FFFFFFFu;
                return u;
            }
            case 0x1E: kind = K_MUL; bsrc = B_I; break;
            case 0x1F: {                                         // clip
                const VUReg s = c.vf[fs];
                float w3 = std::fabs(ps2f(c.vf[ft].f[3]));
                uint32_t f = 0;
                for (int k = 0; k < 3; ++k) {
                    if (ps2f(s.f[k]) > w3) f |= 1u << (2 * k);
                    if (ps2f(s.f[k]) < -w3) f |= 2u << (2 * k);
                }
                u.clip = true;
                u.clip_bits = f;
                return u;
            }
            case 0x20: kind = K_ADD; bsrc = B_Q; break;
            case 0x21: kind = K_MADD; bsrc = B_Q; break;
            case 0x22: kind = K_ADD; bsrc = B_I; break;
            case 0x23: kind = K_MADD; bsrc = B_I; break;
            case 0x24: kind = K_SUB; bsrc = B_Q; break;
            case 0x25: kind = K_MSUB; bsrc = B_Q; break;
            case 0x26: kind = K_SUB; bsrc = B_I; break;
            case 0x27: kind = K_MSUB; bsrc = B_I; break;
            case 0x28: kind = K_ADD; bsrc = B_FT; break;
            case 0x29: kind = K_MADD; bsrc = B_FT; break;
            case 0x2A: kind = K_MUL; bsrc = B_FT; break;
            case 0x2C: kind = K_SUB; bsrc = B_FT; break;
            case 0x2D: kind = K_MSUB; bsrc = B_FT; break;
            case 0x2E: kind = K_OPMULA; bsrc = B_FT; break;
            case 0x2F: return u;                                 // nop
            default: rt_die("VU0 micro program at %04X: upper instruction %08X is not one", pc, w);
            }
        }
    } else {
        rt_die("VU0 micro program at %04X: upper instruction %08X is not one", pc, w);
    }
    if (bsrc == B_Q && t.pair < t.q) early(pc, "Q", t.pair, t.q);
    const VUReg s = c.vf[fs], tt = c.vf[ft], a = c.vacc;
    const float q = vu_q(c), ii = vu_i(c);
    u.target = to_acc ? -1 : fd;
    u.dest = dest;
    u.r = to_acc ? c.vacc : c.vf[fd];
    for (int k = 0; k < 4; ++k) {
        if (!lane(dest, k)) continue;
        if (kind == K_OPMULA || kind == K_OPMSUB) {
            static const int P[3][2] = {{1, 2}, {2, 0}, {0, 1}};
            if (k == 3) continue;
            float prod = fmul(s.f[P[k][0]], tt.f[P[k][1]]);
            u.r.f[k] = kind == K_OPMULA ? prod : fsub(a.f[k], prod);
            continue;
        }
        float b = bsrc == B_BC ? tt.f[bc] : bsrc == B_Q ? q : bsrc == B_I ? ii : tt.f[k];
        uint32_t bu;
        std::memcpy(&bu, &b, 4);
        switch (kind) {
        case K_ADD: u.r.f[k] = fadd(s.f[k], b); break;
        case K_SUB: u.r.f[k] = fsub(s.f[k], b); break;
        case K_MUL: u.r.f[k] = fmul(s.f[k], b); break;
        case K_MADD: u.r.f[k] = fadd(a.f[k], fmul(s.f[k], b)); break;
        case K_MSUB: u.r.f[k] = fsub(a.f[k], fmul(s.f[k], b)); break;
        case K_MAX: u.r.u[k] = vu_max(s.u[k], bu); break;
        case K_MINI: u.r.u[k] = vu_min(s.u[k], bu); break;
        default: break;
        }
    }
    if (kind == K_MAX || kind == K_MINI) {
        if (u.target == 0) u.target = -2;
        return u;
    }
    u.flags = true;
    if (u.target == 0) u.target = -2;                           // vf0 is read-only; the flags still change
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

// The lower instruction. Returns true when it branches (target in `to`).
bool do_lower(EEContext& c, uint32_t w, uint32_t pc, Timing& t, Backup& bk, Backup& nbk, uint32_t& to) {
    if (w == 0x8000033Cu) return false;                          // nop
    uint32_t op = w >> 25;
    int it = (w >> 16) & 31, is = (w >> 11) & 31, id = (w >> 6) & 31, dest = lanes(w);
    int fsf = (w >> 21) & 3, ftf = (w >> 23) & 3;
    int32_t imm11 = (int32_t)(w & 0x7FF) - ((w & 0x400) ? 0x800 : 0);
    auto setvi = [&](int n, uint32_t v, bool ialu) {
        if (!n) return;
        if (ialu) nbk = Backup{n, c.vi[n], t.pair};
        c.vi[n] = v & 0xFFFFu;
    };
    // A branch reads a VI register as it was if the instruction just before wrote it.
    auto rd = [&](int n) -> int16_t {
        if (n && bk.reg == n && bk.pair + 1 == t.pair) {
            if (g_warned.insert(pc | 0x10000).second)
                std::fprintf(stderr, "[vu0] %04X: branch reads vi%d from before the previous instruction\n", pc, n);
            return (int16_t)bk.old;
        }
        return (int16_t)c.vi[n];
    };
    uint32_t next = pc + 8;
    auto branch = [&](bool take) {
        if (take) to = (uint32_t)(next + imm11 * 8) & 0xFFF;
        return take;
    };
    switch (op) {
    case 0x00: {                                                 // lq
        if (!it) return false;
        const VUReg* m = vu0_mem_q((uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[it].u[k] = m->u[k];
        return false;
    }
    case 0x01: {                                                 // sq
        VUReg* m = vu0_mem_q((uint32_t)((int32_t)(int16_t)c.vi[it] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vf[is].u[k];
        return false;
    }
    case 0x04: {                                                 // ilw: the lane named by dest
        const VUReg* m = vu0_mem_q((uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) { setvi(it, m->u[k], false); break; }
        return false;
    }
    case 0x05: {                                                 // isw
        VUReg* m = vu0_mem_q((uint32_t)((int32_t)(int16_t)c.vi[is] + imm11), pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vi[it] & 0xFFFFu;
        return false;
    }
    case 0x08: case 0x09: {                                      // iaddiu, isubiu
        uint32_t imm15 = ((w >> 10) & 0x7800) | (w & 0x7FF);
        setvi(it, op == 0x08 ? c.vi[is] + imm15 : c.vi[is] - imm15, true);
        return false;
    }
    case 0x10: case 0x12: case 0x13: case 0x1C: {                // fceq, fcand, fcor (to vi1); fcget
        if (t.pair < t.clip) early(pc, "the clip flags", t.pair, t.clip);
        uint32_t cf = c.vi[VI_CLIP] & 0xFFFFFF, imm = w & 0xFFFFFF;
        if (op == 0x1C) setvi(it, cf & 0xFFF, false);
        else setvi(1, op == 0x10 ? cf == imm : op == 0x12 ? (cf & imm) != 0 : (cf | imm) == 0xFFFFFF, false);
        return false;
    }
    case 0x11: c.vi[VI_CLIP] = w & 0xFFFFFF; return false;       // fcset
    case 0x14: case 0x16: case 0x17: {                           // fseq, fsand, fsor
        uint32_t imm = (((w >> 21) & 1) << 11) | (w & 0x7FF);
        uint32_t need = op == 0x16 ? imm : 0xFFF;
        if ((need & ST_FMAC) && t.pair < t.st_fmac) early(pc, "the status flags", t.pair, t.st_fmac);
        if ((need & ST_DIV) && t.pair < t.st_div) early(pc, "the status flags", t.pair, t.st_div);
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
        if (t.pair < t.mac) early(pc, "the MAC flags", t.pair, t.mac);
        uint32_t m = c.vi[VI_MAC] & 0xFFFF, s = c.vi[is] & 0xFFFF;
        setvi(it, op == 0x18 ? m == s : op == 0x1A ? m & s : m | s, false);
        return false;
    }
    case 0x20: return branch(true);                              // b
    case 0x21: setvi(it, (next + 8) / 8, false); return branch(true);   // bal
    case 0x24: to = (c.vi[is] * 8) & 0xFFF; return true;         // jr
    case 0x25: {                                                 // jalr
        uint32_t target = (c.vi[is] * 8) & 0xFFF;
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
    default: rt_die("VU0 micro program at %04X: lower instruction %08X is not modelled", pc, w);
    }
    uint32_t sub = w & 0x3F;
    if (sub < 0x3C) {
        switch (sub) {
        case 0x30: setvi(id, c.vi[is] + c.vi[it], true); return false;   // iadd
        case 0x31: setvi(id, c.vi[is] - c.vi[it], true); return false;   // isub
        case 0x32: setvi(it, c.vi[is] + (uint32_t)(id & 16 ? id - 32 : id), true); return false;   // iaddi
        case 0x34: setvi(id, c.vi[is] & c.vi[it], true); return false;   // iand
        case 0x35: setvi(id, c.vi[is] | c.vi[it], true); return false;   // ior
        default: rt_die("VU0 micro program at %04X: lower instruction %08X is not modelled", pc, w);
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
            const VUReg* m = vu0_mem_q(at, pc);
            for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].u[k] = m->u[k];
        }
        if (x == 0x34 && is) c.vi[is] = (c.vi[is] + 1) & 0xFFFFu;
        if (is) nbk = Backup{is, x == 0x34 ? (at & 0xFFFF) : ((at + 1) & 0xFFFF), t.pair};
        return false;
    }
    case 0x35: case 0x37: {                                      // sqi, sqd
        uint32_t old = c.vi[it];
        if (x == 0x37 && it) c.vi[it] = (c.vi[it] - 1) & 0xFFFFu;
        VUReg* m = vu0_mem_q(c.vi[it], pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) m->u[k] = c.vf[fs].u[k];
        if (x == 0x35 && it) c.vi[it] = (c.vi[it] + 1) & 0xFFFFu;
        if (it) nbk = Backup{it, old, t.pair};
        return false;
    }
    case 0x38: case 0x39: case 0x3A:                             // div, sqrt, rsqrt
        if (t.pair < t.q) t.pair = t.q;                          // a second one waits for the first
        if (x == 0x38) vu_div(c, c.vf[fs].f[fsf], c.vf[ft].f[ftf]);
        else if (x == 0x39) vu_sqrt(c, c.vf[ft].f[ftf]);
        else vu_rsqrt(c, c.vf[fs].f[fsf], c.vf[ft].f[ftf]);
        t.q = t.st_div = t.pair + (x == 0x3A ? LAT_RSQRT : x == 0x39 ? LAT_SQRT : LAT_DIV);
        return false;
    case 0x3B: if (t.pair < t.q) t.pair = t.q; return false;     // waitq
    case 0x3C: setvi(it, c.vf[fs].u[fsf], true); return false;   // mtir
    case 0x3D:                                                   // mfir
        if (!ft) return false;
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) c.vf[ft].i[k] = (int16_t)c.vi[is];
        return false;
    case 0x3E: {                                                 // ilwr
        const VUReg* m = vu0_mem_q(c.vi[is], pc);
        for (int k = 0; k < 4; ++k) if (lane(dest, k)) { setvi(it, m->u[k], false); break; }
        return false;
    }
    case 0x3F: {                                                 // iswr
        VUReg* m = vu0_mem_q(c.vi[is], pc);
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
    default:
        rt_die("VU0 micro program at %04X: lower instruction %08X is not modelled (VU1 only, or not yet)", pc, w);
    }
}

}  // namespace

// ---- the interface ---------------------------------------------------------------------------
void vif0_data(const uint8_t* p, size_t n) {
    g_vif0.fifo.insert(g_vif0.fifo.end(), p, p + n);
    vif0_run();
}

// PS2KIT_VU0_LOG=N: the first N runs from each start address are printed,
// with vf1-vf2 (and vf16-vf19 when they change) before and after, as data
// for checking the programs against what they compute.
static void vu0_exec(EEContext& c, uint32_t start);
static void vu0_log(const char* when, uint32_t start, const EEContext& c) {
    std::fprintf(stderr, "[vu0] %04X %s", start, when);
    static const bool all = std::getenv("PS2KIT_VU0_LOG_ALL") != nullptr;
    for (int r = 1; r < 32; ++r) {
        if (!all && r != 1 && r != 2 && (r < 16 || r > 19)) continue;
        std::fprintf(stderr, " vf%d=%.9g,%.9g,%.9g,%.9g", r, c.vf[r].f[0], c.vf[r].f[1], c.vf[r].f[2], c.vf[r].f[3]);
    }
    std::fprintf(stderr, "\n");
}

void vu0_run(EEContext& c, uint32_t start) {
    ++g_runs;
    static const int log_n = std::getenv("PS2KIT_VU0_LOG") ? std::atoi(std::getenv("PS2KIT_VU0_LOG")) : 0;
    static std::map<uint32_t, int> logged;
    bool log = log_n > 0 && logged[start & 0xFFF]++ < log_n;
    if (log) vu0_log("in ", start & 0xFFF, c);
    vu0_exec(c, start);
    if (log) vu0_log("out", start & 0xFFF, c);
}

static void vu0_exec(EEContext& c, uint32_t start) {
    Timing t;
    Backup bk, nbk;
    uint32_t pc = start & 0xFFF, target = 0;
    bool delay = false, ending = false;
    for (int guard = 0;; ++guard) {
        if (guard > 10000000) rt_die("VU0 micro program from %04X: ten million pairs without an end", start);
        uint32_t lo, up;
        std::memcpy(&lo, g_micro + pc, 4);
        std::memcpy(&up, g_micro + pc + 4, 4);
        if (up & 0x18000000u) rt_die("VU0 micro program at %04X: the D or T bit is not modelled yet", pc);
        bool in_delay = delay;
        delay = false;
        Upper u = do_upper(c, up, pc, t);
        uint32_t to = 0;
        bool jumps = false;
        nbk = Backup{};
        if (up & 0x80000000u) {
            // I bit: the lower word is a float for I, loaded after the upper instruction.
        } else {
            int lw = lower_vf_write(lo);
            bool clash = (lw && lw == u.target) || (u.clip && lower_writes_clip(lo));
            if (!clash) jumps = do_lower(c, lo, pc, t, bk, nbk, to);
        }
        if (u.target == -1) {
            for (int k = 0; k < 4; ++k) if (lane(u.dest, k)) c.vacc.u[k] = u.r.u[k];
        } else if (u.target > 0) {
            for (int k = 0; k < 4; ++k) if (lane(u.dest, k)) c.vf[u.target].u[k] = u.r.u[k];
        }
        if (u.flags) {
            vu_flags(c, u.dest, u.r);
            t.mac = t.st_fmac = t.pair + LAT_FMAC;
        }
        if (u.clip) {
            c.vi[VI_CLIP] = ((c.vi[VI_CLIP] << 6) | u.clip_bits) & 0xFFFFFF;
            t.clip = t.pair + LAT_FMAC;
        }
        if (up & 0x80000000u) c.vi[VI_I] = lo;
        bk = nbk;
        ++t.pair;
        ++g_pairs;
        if (ending) {
            if (jumps) rt_die("VU0 micro program at %04X: a branch in the E bit's delay slot", pc);
            c.vi[VI_TPC] = ((in_delay ? target : pc + 8) & 0xFFF) / 8;
            return;
        }
        if (up & 0x40000000u) ending = true;                     // E: one more pair
        if (jumps) {
            if (in_delay) rt_die("VU0 micro program at %04X: a branch in a branch's delay slot", pc);
            delay = true;
            pc = (pc + 8) & 0xFFF;
            target = to;
            continue;
        }
        pc = in_delay ? target : (pc + 8) & 0xFFF;
    }
}

std::string vu0_status() {
    if (!g_runs && !g_vif0.mpg_bytes) return "";
    return std::to_string(g_vif0.mpg_bytes / 8) + " microcode pairs, " + std::to_string(g_runs) + " runs, " +
           std::to_string(g_pairs) + " pairs";
}

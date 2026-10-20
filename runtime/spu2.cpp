// ps2kit runtime — the SPU2, as the IOP's code drives it.
//
// Two cores of 24 voices, their registers at 0x000 and 0x400 (16 bits
// each; the master and effect volumes at 0x760 + core * 0x28), 2 MB of
// sound memory addressed in halfwords, 48000 stereo samples a second.
//
// A voice plays SPU ADPCM: 16-byte blocks of a header (shift and filter;
// flags: bit 0 loop end, bit 1 repeat, bit 2 loop start) and 28 4-bit
// samples, at a pitch (0x1000: one sample per output sample), through an
// ADSR envelope and a left and right volume. Its output goes to the core's
// dry and wet (effect) mixes by VMIXL/R and VMIXEL/R. A core mixes its
// voices, its input and (core 1) core 0's output by MMIX's gates; core 1's
// mix, through its master volume, is the SPU2's output. Key on (KON) starts
// a voice at its start address (SSA); a block whose loop-start flag is set
// sets the loop address (LSAX) unless the IOP wrote LSAX after the key on;
// a block with loop end jumps there, and stops the voice (ENDX) unless it
// repeats.
//
// The interrupt: with ATTR bit 6, a voice reading the block that holds
// IRQA, or a transfer through it, sets the core's bit in SPDIF_IRQINFO
// (0x7C2, bit 2 + core) and raises the IOP's interrupt 9.
//
// Transfers: the transfer address (TSA, 0x1A8) is where the IOP's DMA
// (channel 4 for core 0, 7 for core 1) and the data port (0x1AC) write.
// STATX (0x344): 0 after a reset, bit 10 while a transfer runs, bit 7 when
// one has ended (LIBSD's DMA handler waits for it).
//
// The effects (reverb) are below. Not modelled yet: volume sweeps (a swept
// volume keeps its last value; DW2 sets none), the input from memory
// (ADMA), the output written back to sound memory, interrupts on the
// effects' area. Samples are interpolated with a cubic, not the SPU2's own
// Gaussian table.
//
// The output: the host's audio device when there is a window, and
// --wav FILE (16-bit stereo at 48 kHz).
#include "os.h"
#include "iop_cpu.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

namespace {

constexpr int RATE = 48000;
constexpr uint32_t MEM_HALVES = 1u << 20;             // 2 MB
std::vector<uint16_t> g_mem(MEM_HALVES);
uint16_t g_reg[0x400];                                 // registers 0x000-0x7FF
uint64_t g_dma_bytes, g_samples, g_irqs, g_keyons;
std::set<uint32_t> g_unhandled;

uint16_t& reg(uint32_t off) { return g_reg[(off & 0x7FF) >> 1]; }
uint16_t creg(int c, uint32_t off) { return g_reg[((c * 0x400 + off) & 0x7FF) >> 1]; }
uint32_t addr_pair(int c, uint32_t off) {              // hi, lo halfwords: an address in halfwords
    return ((uint32_t)creg(c, off) << 16 | creg(c, off + 2)) & (MEM_HALVES - 1);
}
uint32_t mask24(int c, uint32_t off) { return creg(c, off) | (uint32_t)(creg(c, off + 2) & 0xFF) << 16; }
int32_t fixed_vol(uint16_t v) { return (v & 0x8000) ? 0 : (int16_t)(v << 1); }
int32_t clamp16(int32_t v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; }

struct Voice {
    uint32_t nax = 0, lsa = 0;
    bool lsa_set = false;                              // LSAX written since the key on
    int16_t block[28] = {};
    uint8_t flags = 0;
    int pos = 28;                                      // in block (28: the next block is due)
    bool loaded = false;
    int32_t prev1 = 0, prev2 = 0;
    int32_t hist[4] = {};
    uint32_t sp = 0;                                   // pitch counter, 12 fraction bits
    int phase = 0;                                     // 0 off, 1 attack, 2 decay, 3 sustain, 4 release
    int32_t level = 0;                                 // 0-0x7FFF
    int32_t wait = 0;
    int32_t out = 0;                                   // after the envelope (PMON's source)
    int16_t volx_l = 0, volx_r = 0;
};
Voice g_v[2][24];
uint32_t g_tsa[2];
bool g_irq_flag[2];
bool g_irq_now;                                        // an interrupt was raised in this batch of samples
int32_t g_noise_level[2] = {1, 1}, g_noise_timer[2];

const int FILTER[16][2] = {{0, 0}, {60, 0}, {115, -52}, {98, -55}, {122, -60}};

// PS2KIT_SPU2_LOG=1: interrupts, transfers and key-ons, with the sample count.
const bool g_log = [] { const char* e = std::getenv("PS2KIT_SPU2_LOG"); return e && *e && *e != '0'; }();

void irq(int c) {
    if (g_log)
        std::fprintf(stderr, "[spu2 %llu] core %d IRQA %05X hit%s%s\n", (unsigned long long)g_samples, c,
                     addr_pair(c, 0x19C), creg(c, 0x19A) & 0x40 ? "" : " (disabled)", g_irq_flag[c] ? " (flag set)" : "");
    if (!(creg(c, 0x19A) & 0x40) || g_irq_flag[c]) return;
    g_irq_flag[c] = true;
    g_irq_now = true;
    reg(0x7C2) |= (uint16_t)(4 << c);
    ++g_irqs;
    iopos_raise(9);
}
// A read or write of [a, a + n) halfwords touched IRQA.
void irq_check(uint32_t a, uint32_t n) {
    for (int c = 0; c < 2; ++c) {
        uint32_t t = addr_pair(c, 0x19C);
        if (((t - a) & (MEM_HALVES - 1)) < n) irq(c);
    }
}

void key_on(int c, int n) {
    Voice& v = g_v[c][n];
    v.nax = addr_pair(c, 0x1C0 + n * 0xC);
    v.lsa_set = false;
    v.pos = 28;
    v.loaded = false;
    v.prev1 = v.prev2 = 0;
    std::memset(v.hist, 0, sizeof v.hist);
    v.sp = 0;
    v.phase = 1;
    v.level = 0;
    v.wait = 0;
    reg(c * 0x400 + 0x340 + (n >= 16 ? 2 : 0)) &= (uint16_t)~(1u << (n & 15));    // ENDX
    ++g_keyons;
    if (g_log)
        std::fprintf(stderr, "[spu2 %llu] core %d voice %d key on: SSA %05X pitch %04X\n", (unsigned long long)g_samples,
                     c, n, v.nax, creg(c, n * 0x10 + 4));
}
void key_off(int c, int n) {
    Voice& v = g_v[c][n];
    if (v.phase) { v.phase = 4; v.wait = 0; }
}

// One step of the envelope (psx-spx's description of the PS1's, which the
// SPU2 keeps): every 1 << max(0, shift - 11) samples the level moves by
// step << max(0, 11 - shift); exponential: rising above 0x6000 four times
// slower, falling by step * level / 0x8000.
void env_step(Voice& v, int shift, int step, bool expo, bool down) {
    int cycles = 1 << std::max(0, shift - 11);
    int32_t s = step << std::max(0, 11 - shift);
    if (expo && !down && v.level > 0x6000) cycles *= 4;
    if (expo && down) s = (int32_t)((int64_t)s * v.level >> 15);
    if (v.wait > 0) { --v.wait; return; }
    v.wait = cycles - 1;
    v.level = std::clamp<int32_t>(v.level + s, 0, 0x7FFF);
}
void envelope(Voice& v, uint16_t a1, uint16_t a2) {
    switch (v.phase) {
    case 1: {
        int rate = (a1 >> 8) & 0x7F;
        env_step(v, rate >> 2, 7 - (rate & 3), a1 & 0x8000, false);
        if (v.level >= 0x7FFF) { v.phase = 2; v.wait = 0; }
        break;
    }
    case 2: {
        int rate = (a1 >> 4) & 0xF;
        env_step(v, rate, -8, true, true);
        if (v.level <= (int32_t)((a1 & 0xF) + 1) * 0x800) { v.phase = 3; v.wait = 0; }
        break;
    }
    case 3: {
        int rate = (a2 >> 6) & 0x7F;
        bool down = a2 & 0x4000;
        env_step(v, rate >> 2, down ? -8 + (rate & 3) : 7 - (rate & 3), a2 & 0x8000, down);
        break;
    }
    case 4: {
        env_step(v, a2 & 0x1F, -8, a2 & 0x20, true);
        if (v.level <= 0) { v.phase = 0; v.level = 0; }
        break;
    }
    }
}

void decode(Voice& v) {
    const uint16_t* b = &g_mem[v.nax & ~7u & (MEM_HALVES - 1)];
    uint16_t head = b[0];
    int shift = head & 0xF, f = (head >> 4) & 0xF;
    v.flags = (uint8_t)(head >> 8);
    for (int k = 0; k < 28; ++k) {
        int nib = (b[1 + k / 4] >> ((k & 3) * 4)) & 0xF;
        int32_t s = (int32_t)((uint32_t)nib << 28) >> (16 + shift);
        s += (v.prev1 * FILTER[f][0] + v.prev2 * FILTER[f][1] + 32) >> 6;
        s = clamp16(s);
        v.prev2 = v.prev1;
        v.prev1 = s;
        v.block[k] = (int16_t)s;
    }
}

// The voice's next source sample (its block's next, or the next block's first).
void next_sample(int c, int n, Voice& v) {
    if (v.pos >= 28) {
        if (v.loaded) {                                // the block just played ends
            if (v.flags & 1) {
                reg(c * 0x400 + 0x340 + (n >= 16 ? 2 : 0)) |= (uint16_t)(1u << (n & 15));
                v.nax = v.lsa;
                if (!(v.flags & 2)) { v.phase = 0; v.level = 0; }
            } else {
                v.nax = (v.nax + 8) & (MEM_HALVES - 1);
            }
        }
        v.nax &= ~7u;
        irq_check(v.nax, 8);
        decode(v);
        if ((v.flags & 4) && !v.lsa_set) v.lsa = v.nax;
        v.loaded = true;
        v.pos = 0;
    }
    v.hist[0] = v.hist[1];
    v.hist[1] = v.hist[2];
    v.hist[2] = v.hist[3];
    v.hist[3] = v.block[v.pos++];
}

int32_t interpolate(const Voice& v) {
    // Catmull-Rom between hist[1] and hist[2], at sp's fraction.
    int32_t f = (int32_t)(v.sp & 0xFFF);
    int32_t p0 = v.hist[0], p1 = v.hist[1], p2 = v.hist[2], p3 = v.hist[3];
    int64_t a = -p0 + 3 * p1 - 3 * p2 + p3, b = 2 * p0 - 5 * p1 + 4 * p2 - p3, cc = p2 - p0;
    int64_t x = ((((a * f >> 12) + b) * f >> 12) + cc) * f >> 12;
    return clamp16((int32_t)(p1 + (x >> 1)));
}

int32_t noise(int c) {
    uint16_t attr = creg(c, 0x19A);
    int step = ((attr >> 8) & 3) + 4, shift = (attr >> 10) & 0xF;
    int32_t& lv = g_noise_level[c];
    int parity = ((lv >> 15) ^ (lv >> 12) ^ (lv >> 11) ^ (lv >> 10) ^ 1) & 1;
    g_noise_timer[c] -= step;
    if (g_noise_timer[c] < 0) {
        lv = (lv << 1) | parity;
        g_noise_timer[c] += 0x20000 >> shift;
        if (g_noise_timer[c] < 0) g_noise_timer[c] += 0x20000 >> shift;
    }
    return (int16_t)lv;
}

struct Mix { int32_t dl = 0, dr = 0, wl = 0, wr = 0; };

Mix voices(int c) {
    Mix m;
    uint32_t pmon = mask24(c, 0x180), non = mask24(c, 0x184);
    uint32_t vl = mask24(c, 0x188), vel = mask24(c, 0x18C), vr = mask24(c, 0x190), ver = mask24(c, 0x194);
    for (int n = 0; n < 24; ++n) {
        Voice& v = g_v[c][n];
        uint32_t base = (uint32_t)n * 0x10;
        uint32_t step = std::min<uint32_t>(creg(c, base + 4), 0x3FFF);
        if ((pmon >> n & 1) && n > 0) step = (uint32_t)std::clamp<int64_t>((int64_t)step * (g_v[c][n - 1].out + 0x8000) >> 15, 0, 0x3FFF);
        if (v.phase == 0) {
            v.out = 0;
            reg(c * 0x400 + base + 0xA) = 0;
            continue;
        }
        v.sp += step;
        while (v.sp >= 0x1000 && v.phase) {
            v.sp -= 0x1000;
            next_sample(c, n, v);
        }
        int32_t s = (non >> n & 1) ? noise(c) : interpolate(v);
        envelope(v, creg(c, base + 6), creg(c, base + 8));
        v.out = (int32_t)((int64_t)s * v.level >> 15);
        uint16_t rl = creg(c, base), rr = creg(c, base + 2);
        if (!(rl & 0x8000)) v.volx_l = (int16_t)fixed_vol(rl);
        if (!(rr & 0x8000)) v.volx_r = (int16_t)fixed_vol(rr);
        int32_t l = v.out * v.volx_l >> 15, r = v.out * v.volx_r >> 15;
        reg(c * 0x400 + base + 0xA) = (uint16_t)v.level;      // ENVX
        reg(c * 0x400 + base + 0xC) = (uint16_t)v.volx_l;     // VOLXL
        reg(c * 0x400 + base + 0xE) = (uint16_t)v.volx_r;     // VOLXR
        if (vl >> n & 1) m.dl += l;
        if (vr >> n & 1) m.dr += r;
        if (vel >> n & 1) m.wl += l;
        if (ver >> n & 1) m.wr += r;
    }
    return m;
}

// ---- the effects (reverb) ---------------------------------------------------------------------
// psx-spx's description of the PS1's reverb, which each SPU2 core keeps,
// as PCSX2 runs it: one side per output sample (left on even samples,
// right on odd: 24 kHz each), its input brought down and its output up
// by a 39-tap half-band filter. The work area is ESA (0x2E0) to EEA
// (0x33C, its high halfword; the low one is 0xFFFF); the address registers
// (0x2E4-0x33A, high and low halfwords) are offsets from the area's
// current position, which moves on every second sample. The results are
// written back only with ATTR bit 7 (effects on); the output, through
// EVOL, joins the core's dry mix.
const int32_t FIR[39] = {-1, 0, 2, 0, -10, 0, 35, 0, -103, 0, 266, 0, -616, 0, 1332, 0, -2960, 0, 10246, 16384,
                         10246, 0, -2960, 0, 1332, 0, -616, 0, 266, 0, -103, 0, 35, 0, -10, 0, 2, 0, -1};
struct Reverb { int32_t down[2][128] = {}, up[2][128] = {}; int pos = 0; };
Reverb g_rv[2];

enum {                                                 // the address registers, by offset in a core
    FB_SRC_A = 0x2E4, FB_SRC_B = 0x2E8, IIR_DEST_A0 = 0x2EC, IIR_DEST_A1 = 0x2F0, ACC_SRC_A0 = 0x2F4,
    ACC_SRC_A1 = 0x2F8, ACC_SRC_B0 = 0x2FC, ACC_SRC_B1 = 0x300, IIR_SRC_A0 = 0x304, IIR_SRC_A1 = 0x308,
    IIR_DEST_B0 = 0x30C, IIR_DEST_B1 = 0x310, ACC_SRC_C0 = 0x314, ACC_SRC_C1 = 0x318, ACC_SRC_D0 = 0x31C,
    ACC_SRC_D1 = 0x320, IIR_SRC_B1 = 0x324, IIR_SRC_B0 = 0x328, MIX_DEST_A0 = 0x32C, MIX_DEST_A1 = 0x330,
    MIX_DEST_B0 = 0x334, MIX_DEST_B1 = 0x338,
};
enum {                                                 // the coefficients, at 0x774 + core * 0x28
    IIR_ALPHA = 0x0, ACC_COEF_A = 0x2, ACC_COEF_B = 0x4, ACC_COEF_C = 0x6, ACC_COEF_D = 0x8, IIR_COEF = 0xA,
    FB_ALPHA = 0xC, FB_X = 0xE, IN_COEF_L = 0x10, IN_COEF_R = 0x12,
};

int32_t rv_coef(int c, int k) { return (int16_t)reg(0x774 + c * 0x28 + k); }
int32_t mul(int32_t x, int32_t y) { return (int32_t)((int64_t)x * y >> 15); }

int32_t fir(const int32_t* buf, int pos, int shift) {
    int idx = (pos - 39) & 63;
    int64_t s = 0;
    for (int i = 0; i < 39; ++i) s += (int64_t)buf[idx + i] * FIR[i];
    return clamp16((int32_t)(s >> shift));
}

void reverb(int c, int32_t in_l, int32_t in_r, int32_t& out_l, int32_t& out_r) {
    Reverb& rv = g_rv[c];
    uint32_t start = addr_pair(c, 0x2E0), end = ((uint32_t)creg(c, 0x33C) << 16 | 0xFFFF) & (MEM_HALVES - 1);
    out_l = out_r = 0;
    if (start >= end) return;
    rv.down[0][rv.pos] = rv.down[0][rv.pos | 64] = clamp16(in_l);
    rv.down[1][rv.pos] = rv.down[1][rv.pos | 64] = clamp16(in_r);
    bool R = g_samples & 1;
    uint32_t size = end - start + 1, at = (uint32_t)(g_samples >> 1);
    auto ix = [&](uint32_t reg_off, int32_t minus = 0) {
        uint32_t off = addr_pair(c, reg_off) - (uint32_t)minus;
        return start + (uint32_t)((at + off) % size);
    };
    auto a2 = [&](uint32_t a, uint32_t b) { return addr_pair(c, a) - addr_pair(c, b); };
    auto ixv = [&](uint32_t off) { return start + (uint32_t)((at + off) % size); };
    int16_t* m = (int16_t*)g_mem.data();
    uint32_t same_src = ix(R ? IIR_SRC_A1 : IIR_SRC_A0), same_dst = ix(R ? IIR_DEST_A1 : IIR_DEST_A0),
             same_prv = ix(R ? IIR_DEST_A1 : IIR_DEST_A0, 1);
    uint32_t diff_src = ix(R ? IIR_SRC_B0 : IIR_SRC_B1), diff_dst = ix(R ? IIR_DEST_B1 : IIR_DEST_B0),
             diff_prv = ix(R ? IIR_DEST_B1 : IIR_DEST_B0, 1);
    uint32_t comb1 = ix(R ? ACC_SRC_A1 : ACC_SRC_A0), comb2 = ix(R ? ACC_SRC_B1 : ACC_SRC_B0),
             comb3 = ix(R ? ACC_SRC_C1 : ACC_SRC_C0), comb4 = ix(R ? ACC_SRC_D1 : ACC_SRC_D0);
    uint32_t apf1_src = ixv(a2(R ? MIX_DEST_A1 : MIX_DEST_A0, FB_SRC_A)), apf1_dst = ix(R ? MIX_DEST_A1 : MIX_DEST_A0);
    uint32_t apf2_src = ixv(a2(R ? MIX_DEST_B1 : MIX_DEST_B0, FB_SRC_B)), apf2_dst = ix(R ? MIX_DEST_B1 : MIX_DEST_B0);

    int32_t in = mul(rv_coef(c, R ? IN_COEF_R : IN_COEF_L), fir(rv.down[R], rv.pos, 15));
    int32_t iir = rv_coef(c, IIR_ALPHA), wall = rv_coef(c, IIR_COEF);
    int32_t same = mul(iir, in + mul(wall, m[same_src]) - m[same_prv]) + m[same_prv];
    int32_t diff = mul(iir, in + mul(wall, m[diff_src]) - m[diff_prv]) + m[diff_prv];
    int32_t out = mul(rv_coef(c, ACC_COEF_A), m[comb1]) + mul(rv_coef(c, ACC_COEF_B), m[comb2]) +
                  mul(rv_coef(c, ACC_COEF_C), m[comb3]) + mul(rv_coef(c, ACC_COEF_D), m[comb4]);
    int32_t apf1 = out - mul(rv_coef(c, FB_ALPHA), m[apf1_src]);
    out = m[apf1_src] + mul(rv_coef(c, FB_ALPHA), apf1);
    int32_t apf2 = out - mul(rv_coef(c, FB_X), m[apf2_src]);
    out = m[apf2_src] + mul(rv_coef(c, FB_X), apf2);
    if (creg(c, 0x19A) & 0x80) {
        m[same_dst] = (int16_t)clamp16(same);
        m[diff_dst] = (int16_t)clamp16(diff);
        m[apf1_dst] = (int16_t)clamp16(apf1);
        m[apf2_dst] = (int16_t)clamp16(apf2);
    }
    out = clamp16(out);
    rv.up[R][rv.pos] = rv.up[R][rv.pos | 64] = out;
    rv.up[!R][rv.pos] = rv.up[!R][rv.pos | 64] = 0;
    rv.pos = (rv.pos + 1) & 63;
    out_l = fir(rv.up[0], rv.pos, 14);
    out_r = fir(rv.up[1], rv.pos, 14);
}

// A core's mix (PCSX2's V_Core::Mix order): its voices, its input (none
// here: ADMA is not modelled) and the external input, by MMIX's gates; the
// wet part through the effects, then EVOL, into the dry.
void core_mix(int c, const Mix& vm, int32_t ext_l, int32_t ext_r, int32_t& l, int32_t& r) {
    uint16_t mmix = creg(c, 0x198) & (c == 0 ? 0xFF0 : 0xFFF);
    int32_t dl = clamp16(vm.dl), dr = clamp16(vm.dr), wl = clamp16(vm.wl), wr = clamp16(vm.wr);
    l = r = 0;
    if (mmix & 0x800) l += dl;
    if (mmix & 0x400) r += dr;
    if (mmix & 0x008) l += ext_l;
    if (mmix & 0x004) r += ext_r;
    int32_t tl = 0, tr = 0;
    if (mmix & 0x200) tl += wl;
    if (mmix & 0x100) tr += wr;
    if (mmix & 0x002) tl += ext_l;
    if (mmix & 0x001) tr += ext_r;
    int32_t rl, rr;
    reverb(c, tl, tr, rl, rr);
    l += rl * fixed_vol(reg(0x764 + c * 0x28)) >> 15;
    r += rr * fixed_vol(reg(0x766 + c * 0x28)) >> 15;
}

void sample(int16_t out[2]) {
    Mix m0 = voices(0), m1 = voices(1);
    int32_t l0, r0, l1, r1;
    core_mix(0, m0, 0, 0, l0, r0);
    uint16_t mvl0 = reg(0x760), mvr0 = reg(0x762);
    l0 = clamp16(l0 * fixed_vol(mvl0) >> 15);
    r0 = clamp16(r0 * fixed_vol(mvr0) >> 15);
    if (creg(0, 0x19A) & 0x4000) l0 = r0 = 0;          // core 0 muted
    int32_t el = l0 * fixed_vol(reg(0x794)) >> 15, er = r0 * fixed_vol(reg(0x796)) >> 15;   // core 1's BVOL
    core_mix(1, m1, el, er, l1, r1);
    out[0] = (int16_t)clamp16(l1 * fixed_vol(reg(0x788)) >> 15);
    out[1] = (int16_t)clamp16(r1 * fixed_vol(reg(0x78A)) >> 15);
    ++g_samples;
}

// ---- the output ---------------------------------------------------------------------------------
SDL_AudioStream* g_stream;
FILE* g_wav;
uint32_t g_wav_bytes;
double g_t0 = -1;
uint64_t g_made;                                       // samples made since g_t0

void wav_header() {
    uint8_t h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0};
    auto put = [&](int at, uint32_t v, int n) { for (int k = 0; k < n; ++k) h[at + k] = (uint8_t)(v >> (8 * k)); };
    put(4, 36 + g_wav_bytes, 4);
    put(24, RATE, 4);
    put(28, RATE * 4, 4);
    put(32, 4, 2);
    put(34, 16, 2);
    std::memcpy(h + 36, "data", 4);
    put(40, g_wav_bytes, 4);
    std::fseek(g_wav, 0, SEEK_SET);
    std::fwrite(h, 1, 44, g_wav);
    std::fseek(g_wav, 0, SEEK_END);
}

}  // namespace

void spu2_output(bool device, const std::string& wav) {
    if (!iopos_enabled()) return;
    if (device) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            std::fprintf(stderr, "ps2kit: no audio (%s)\n", SDL_GetError());
        } else {
            SDL_AudioSpec spec{SDL_AUDIO_S16LE, 2, RATE};
            g_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
            if (g_stream) SDL_ResumeAudioStreamDevice(g_stream);
            else std::fprintf(stderr, "ps2kit: no audio device (%s)\n", SDL_GetError());
        }
    }
    if (!wav.empty()) {
        g_wav = std::fopen(wav.c_str(), "wb");
        if (!g_wav) rt_die("cannot write %s", wav.c_str());
        wav_header();
    }
}

void spu2_init() {
    std::memset(g_reg, 0, sizeof g_reg);
}

uint16_t spu2_read(uint32_t off) {
    if (off >= 0x800) return 0;
    uint32_t o = off & 0x3FF;
    if (off < 0x760 && o >= 0x1C0 && o < 0x2E0 && ((o - 0x1C0) % 0xC == 8 || (o - 0x1C0) % 0xC == 0xA)) {
        // NAX: the halfword the voice reads next (in its block: after the
        // header, four samples a halfword)
        const Voice& v = g_v[off >> 10][(o - 0x1C0) / 0xC];
        uint32_t a = v.loaded ? ((v.nax & ~7u) + 1 + std::min(v.pos, 27) / 4) & (MEM_HALVES - 1) : v.nax;
        return (o - 0x1C0) % 0xC == 8 ? (uint16_t)(a >> 16) : (uint16_t)a;
    }
    return reg(off);
}

void spu2_write(uint32_t off, uint16_t v) {
    if (off >= 0x800) return;                          // (LIBSD clears 0xB60 + core * 0x28 too; nothing there)
    int c = off < 0x760 ? (int)(off >> 10) : (off >= 0x788 && off < 0x7B0 ? 1 : 0);
    uint32_t o = off < 0x760 ? off & 0x3FF : off;
    uint16_t old = reg(off);
    reg(off) = v;
    // Volume sweeps are not modelled yet: said once.
    bool vol = (off < 0x760 && o < 0x180 && (o & 0xF) < 4) || (off >= 0x760 && off < 0x770) ||
               (off >= 0x788 && off < 0x798);
    if (vol && (v & 0x8000) && g_unhandled.insert(0x20000).second)
        std::fprintf(stderr, "ps2kit: SPU2 volume sweep (%03X = %04X): not modelled yet, the volume stays\n", off, v);
    if (off >= 0x760) return;
    if (o == 0x1A8 || o == 0x1AA) g_tsa[c] = addr_pair(c, 0x1A8);
    else if (o == 0x1AC) {                             // the data port: into memory at TSA
        g_mem[g_tsa[c]] = v;
        irq_check(g_tsa[c], 1);
        g_tsa[c] = (g_tsa[c] + 1) & (MEM_HALVES - 1);
    } else if (o == 0x1A0 || o == 0x1A2) {             // KON (voices 0-15, 16-23; the bits above, none)
        for (int k = 0; k < (o == 0x1A2 ? 8 : 16); ++k)
            if (v >> k & 1) key_on(c, (o == 0x1A2 ? 16 : 0) + k);
        reg(off) = 0;
    } else if (o == 0x1A4 || o == 0x1A6) {             // KOFF
        for (int k = 0; k < (o == 0x1A6 ? 8 : 16); ++k)
            if (v >> k & 1) key_off(c, (o == 0x1A6 ? 16 : 0) + k);
        reg(off) = 0;
    } else if (o == 0x19A) {                           // ATTR
        if (g_log && (v & 0x30) == 0x10 && (old & 0x30) != 0x10)
            std::fprintf(stderr, "[spu2 %llu] core %d manual transfer, TSA now %05X\n", (unsigned long long)g_samples, c,
                         g_tsa[c]);
        if (!(v & 0x8000) && (old & 0x8000)) reg(c * 0x400 + 0x344) = 0;   // the core reset
        if (!(v & 0x40)) {
            g_irq_flag[c] = false;
            reg(0x7C2) &= (uint16_t)~(4 << c);
        }
    } else if (o >= 0x1C0 && o < 0x2E0) {
        uint32_t n = (o - 0x1C0) / 0xC, f = (o - 0x1C0) % 0xC;
        if (f == 4 || f == 6) {                        // LSAX
            g_v[c][n].lsa = addr_pair(c, 0x1C0 + n * 0xC + 4);
            g_v[c][n].lsa_set = true;
        }
    } else if (o == 0x1B0 && v) {
        if (g_unhandled.insert(off).second) std::fprintf(stderr, "ps2kit: SPU2 core %d ADMA %X: not modelled yet\n", c, v);
    }
}

void spu2_dma(int core, uint32_t madr, uint32_t words, bool to_spu) {
    uint32_t& t = g_tsa[core];
    if (g_log) {
        std::fprintf(stderr, "[spu2 %llu] core %d DMA %s %06X: %05X + %X halfwords; voices at", (unsigned long long)g_samples,
                     core, to_spu ? "from" : "to", madr, t, words * 2);
        for (int c = 0; c < 2; ++c)
            for (int n = 0; n < 24; ++n)
                if (g_v[c][n].phase) std::fprintf(stderr, " %d.%d:%05X", c, n, g_v[c][n].nax);
        std::fprintf(stderr, "\n");
    }
    uint8_t* ram = iop_ram();
    irq_check(t, words * 2);
    for (uint32_t k = 0; k < words * 2; ++k) {
        uint32_t a = (madr + 2 * k) & 0x1FFFFF;
        if (to_spu) {
            uint16_t h;
            std::memcpy(&h, ram + a, 2);
            g_mem[t] = h;
        } else {
            std::memcpy(ram + a, &g_mem[t], 2);
        }
        t = (t + 1) & (MEM_HALVES - 1);
    }
    uint16_t& statx = reg(core * 0x400 + 0x344);
    statx = (uint16_t)((statx & ~0x400) | 0x80);
    g_dma_bytes += (uint64_t)words * 4;
    TRACE(2, "[spu2] core %d DMA %s %06X, %u words\n", core, to_spu ? "from" : "to", madr, words);
}

// The samples due by `now`, at most 256 at a time and none after one that
// raised an interrupt: the IOP's thread runs the handler, and whatever it
// wakes, before the SPU2 plays on (the IOP answers in microseconds; a
// stream refilled on an interrupt must not be played past first). The IOP's
// loop calls again while samples are due.
void spu2_advance(double now) {
    if (g_t0 < 0 || now < g_t0) { g_t0 = now; g_made = 0; }
    uint64_t due = (uint64_t)((now - g_t0) * RATE);
    if (due > g_made + RATE / 2) g_made = due - RATE / 2;   // far behind (a host stall): drop the rest
    if (g_made >= due) return;
    int16_t buf[2 * 256];
    int n = 0, most = (int)std::min<uint64_t>(due - g_made, 256);
    g_irq_now = false;
    while (n < most && !g_irq_now) sample(buf + 2 * n++);
    g_made += n;
    if (g_stream) SDL_PutAudioStreamData(g_stream, buf, n * 4);
    if (g_wav) {
        std::fwrite(buf, 4, n, g_wav);
        g_wav_bytes += n * 4;
        wav_header();                                  // (the run can end at any time)
    }
}

bool spu2_behind(double now) { return g_t0 >= 0 && (uint64_t)((now - g_t0) * RATE) > g_made; }

std::string spu2_status() {
    return "SPU2 " + std::to_string(g_samples / RATE) + " s of samples, " + std::to_string(g_keyons) + " key-ons, " +
           std::to_string(g_irqs) + " interrupts, DMA " + std::to_string(g_dma_bytes / 1024) + " KB";
}

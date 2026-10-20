// ps2kit runtime — the Emotion Engine as seen by recompiled code.
//
// Recompiled functions have the signature `void f_XXXXXXXX(EEContext& c)` and
// touch guest state only through this header: registers in EEContext, memory
// through ee_ld*/ee_st* (the guest is little-endian like the host: no swaps),
// and out-of-line services the runtime provides (indirect calls, syscalls,
// traps, hardware registers, VU0 micro programs, interrupt delivery at loop
// back-edges).
//
// Semantics are those of the R5900 core manual: 32-bit operations sign-extend
// their result into the low 64 bits; only MMI, LQ/SQ and the 128-bit COP2
// moves touch bits 64-127 (other writes keep them, as PCSX2's interpreter
// does; to be confirmed on hardware). Where the manual leaves a case
// undefined (division by zero) the choice is written next to the helper.
//
// Floating point: the PS2's FPU and VUs are not IEEE 754. They have no
// denormals (a denormal operand or result is zero), no infinities or NaNs
// (exponent 255 is an ordinary exponent) and round toward zero. The runtime
// flushes denormals, clamps infinities and NaNs to the largest finite value
// with their sign, and computes in host single precision with the host set
// to round toward zero (rt_init_fpu): PCSX2's default model. Known
// differences from the console: exponent 255 is not an ordinary exponent
// here, and the PS2's multiplier can be off by one in the last bit.
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && !defined(__clang__)
#define EE_UNLIKELY(x) (x)
#define EE_INLINE __forceinline
#else
#define EE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define EE_INLINE inline __attribute__((always_inline))
#endif

union EEReg {
    uint64_t ud[2];
    int64_t  sd[2];
    uint32_t uw[4];
    int32_t  sw[4];
    uint16_t uh[8];
    int16_t  sh[8];
    uint8_t  ub[16];
    int8_t   sb[16];
};

union VUReg {
    float    f[4];      // x y z w
    uint32_t u[4];
    int32_t  i[4];
};

union FPR {
    float    f;
    uint32_t u;
    int32_t  i;
};

// VU0 control registers, as cfc2/ctc2 number them.
enum {
    VI_STATUS = 16, VI_MAC = 17, VI_CLIP = 18, VI_R = 20, VI_I = 21, VI_Q = 22,
    VI_TPC = 26, VI_CMSAR0 = 27, VI_FBRST = 28, VI_VPU_STAT = 29, VI_CMSAR1 = 31,
};

struct EEContext {
    EEReg    r[32];
    EEReg    hi, lo;    // ud[0]: HI/LO, ud[1]: HI1/LO1 (the second pipeline)
    uint32_t sa;        // the shift amount register, in bits
    FPR      f[32];
    FPR      acc;
    uint32_t fcr31;     // bit 23: C, the compare result
    VUReg    vf[32];    // vf0 is (0, 0, 0, 1) and read-only
    VUReg    vacc;
    uint32_t vi[32];    // vi0-vi15 (16-bit, vi0 = 0) and the control registers
    uint32_t cop0[32];
    uint32_t pc;        // last guest address the runtime was told about (logs)
};

typedef void (*EEFunc)(EEContext&);
struct EEFuncEntry { uint32_t addr; EEFunc fn; const char* name; };

// ---- services provided by the runtime -------------------------------------------
extern uint8_t* g_ee_ram;                     // 32 MB
extern uint8_t* g_ee_spr;                     // 16 KB scratchpad
extern uint8_t* g_vu0_mem;                    // 4 KB VU0 data memory
void     ee_call(EEContext& c, uint32_t addr);                 // jalr, indirect jr
void     ee_syscall(EEContext& c, uint32_t pc);                // syscall number in v1
void     ee_break(EEContext& c, uint32_t pc, uint32_t code);
void     ee_unimplemented(EEContext& c, uint32_t pc, const char* what);
uint64_t ee_read_slow(uint32_t a, int size);                   // mirrors, scratchpad, hardware
void     ee_write_slow(uint32_t a, uint64_t v, int size);
void     ee_read128_slow(uint32_t a, EEReg& out);
void     ee_write128_slow(uint32_t a, const EEReg& v);
uint32_t ee_mfc0(EEContext& c, int reg);
void     ee_mtc0(EEContext& c, int reg, uint32_t v);
void     ee_ei(EEContext& c);
void     ee_di(EEContext& c);
uint32_t ee_cfc2(EEContext& c, int reg);
void     ee_ctc2(EEContext& c, int reg, uint32_t v);
void     ee_vu0_callms(EEContext& c, uint32_t addr);           // run a VU0 micro program
VUReg*   ee_vu0_vu1reg(uint32_t qaddr);                       // VU1's register at VU0's quadword 0x400+

// Interrupts are delivered at safe points: every loop back-edge.
extern std::atomic<uint32_t> g_ee_pending;
void ee_poll(EEContext& c);
#define EE_POLL(c) do { if (EE_UNLIKELY(g_ee_pending.load(std::memory_order_relaxed))) ee_poll(c); } while (0)

// ---- integers -------------------------------------------------------------------
#define S32(x) ((uint64_t)(int64_t)(int32_t)(uint32_t)(x))

EE_INLINE void ee_mult(EEContext& c, int rd, int pipe, int64_t p) {
    c.lo.ud[pipe] = S32(p);
    c.hi.ud[pipe] = S32((uint64_t)p >> 32);
    if (rd) c.r[rd].ud[0] = c.lo.ud[pipe];
}
EE_INLINE void ee_madd(EEContext& c, int rd, int pipe, int64_t p) {
    uint64_t acc = ((uint64_t)(uint32_t)c.hi.ud[pipe] << 32) | (uint32_t)c.lo.ud[pipe];
    ee_mult(c, rd, pipe, (int64_t)(acc + (uint64_t)p));
}
// Division by zero: LO = (rs < 0 ? 1 : -1), HI = rs; INT_MIN / -1: LO = INT_MIN,
// HI = 0 (the values the R5900's divider is documented to leave by PCSX2).
EE_INLINE void ee_div(EEContext& c, int pipe, int32_t a, int32_t b) {
    if (b == 0) { c.lo.ud[pipe] = S32(a < 0 ? 1 : -1); c.hi.ud[pipe] = S32(a); return; }
    if (a == INT32_MIN && b == -1) { c.lo.ud[pipe] = S32(a); c.hi.ud[pipe] = 0; return; }
    c.lo.ud[pipe] = S32(a / b);
    c.hi.ud[pipe] = S32(a % b);
}
EE_INLINE void ee_divu(EEContext& c, int pipe, uint32_t a, uint32_t b) {
    if (b == 0) { c.lo.ud[pipe] = S32(0xFFFFFFFFu); c.hi.ud[pipe] = S32(a); return; }
    c.lo.ud[pipe] = S32(a / b);
    c.hi.ud[pipe] = S32(a % b);
}

// ---- memory ---------------------------------------------------------------------
// Below 32 MB the address is RAM, read directly; everything else (the
// uncached mirrors at 0x20000000 and 0x30000000, kseg0/1, the scratchpad at
// 0x70000000, the hardware) goes through the runtime.
#define EE_RAM_SIZE 0x02000000u
#define EE_FAST(a) ((a) < EE_RAM_SIZE)

#define EE_LOAD(T, N) \
    EE_INLINE T ee_ld##N(uint32_t a) { \
        if (EE_UNLIKELY(!EE_FAST(a))) return (T)ee_read_slow(a, sizeof(T)); \
        T v; std::memcpy(&v, g_ee_ram + a, sizeof v); return v; }
#define EE_STORE(T, N) \
    EE_INLINE void ee_st##N(uint32_t a, T v) { \
        if (EE_UNLIKELY(!EE_FAST(a))) { ee_write_slow(a, (uint64_t)v, sizeof(T)); return; } \
        std::memcpy(g_ee_ram + a, &v, sizeof v); }
EE_LOAD(uint8_t, 8)
EE_LOAD(uint16_t, 16)
EE_LOAD(uint32_t, 32)
EE_LOAD(uint64_t, 64)
EE_STORE(uint8_t, 8)
EE_STORE(uint16_t, 16)
EE_STORE(uint32_t, 32)
EE_STORE(uint64_t, 64)

EE_INLINE void ee_ld128(uint32_t a, void* out) {
    a &= ~15u;
    if (EE_UNLIKELY(!EE_FAST(a))) { ee_read128_slow(a, *(EEReg*)out); return; }
    std::memcpy(out, g_ee_ram + a, 16);
}
EE_INLINE void ee_st128(uint32_t a, const void* v) {
    a &= ~15u;
    if (EE_UNLIKELY(!EE_FAST(a))) { ee_write128_slow(a, *(const EEReg*)v); return; }
    std::memcpy(g_ee_ram + a, v, 16);
}

// Unaligned pairs, little-endian (MIPS IV): the L form holds the most
// significant bytes, the R form the least. LWL always sign-extends; LWR only
// when it loads the whole word, otherwise bits 32-63 are kept.
EE_INLINE uint64_t ee_lwl(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 3) * 8;
    uint32_t w = ee_ld32(a & ~3u);
    uint32_t keep = sh == 24 ? 0u : (0x00FFFFFFu >> sh);
    return S32(((uint32_t)rt & keep) | (w << (24 - sh)));
}
EE_INLINE uint64_t ee_lwr(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 3) * 8;
    uint32_t w = ee_ld32(a & ~3u);
    if (sh == 0) return S32(w);
    uint32_t keep = 0xFFFFFF00u << (24 - sh);
    return (rt & 0xFFFFFFFF00000000ull) | (((uint32_t)rt & keep) | (w >> sh));
}
EE_INLINE uint64_t ee_ldl(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 7) * 8;
    uint64_t w = ee_ld64(a & ~7u);
    uint64_t keep = sh == 56 ? 0ull : (0x00FFFFFFFFFFFFFFull >> sh);
    return (rt & keep) | (w << (56 - sh));
}
EE_INLINE uint64_t ee_ldr(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 7) * 8;
    uint64_t w = ee_ld64(a & ~7u);
    uint64_t keep = sh == 0 ? 0ull : (0xFFFFFFFFFFFFFF00ull << (56 - sh));
    return (rt & keep) | (w >> sh);
}
EE_INLINE void ee_swl(uint32_t a, uint32_t rt) {
    unsigned sh = (a & 3) * 8;
    uint32_t w = ee_ld32(a & ~3u);
    uint32_t keep = sh == 24 ? 0u : (0xFFFFFF00u << sh);
    ee_st32(a & ~3u, (w & keep) | (rt >> (24 - sh)));
}
EE_INLINE void ee_swr(uint32_t a, uint32_t rt) {
    unsigned sh = (a & 3) * 8;
    uint32_t w = ee_ld32(a & ~3u);
    uint32_t keep = sh == 0 ? 0u : (0x00FFFFFFu >> (24 - sh));
    ee_st32(a & ~3u, (w & keep) | (rt << sh));
}
EE_INLINE void ee_sdl(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 7) * 8;
    uint64_t w = ee_ld64(a & ~7u);
    uint64_t keep = sh == 56 ? 0ull : (0xFFFFFFFFFFFFFF00ull << sh);
    ee_st64(a & ~7u, (w & keep) | (rt >> (56 - sh)));
}
EE_INLINE void ee_sdr(uint32_t a, uint64_t rt) {
    unsigned sh = (a & 7) * 8;
    uint64_t w = ee_ld64(a & ~7u);
    uint64_t keep = sh == 0 ? 0ull : (0x00FFFFFFFFFFFFFFull >> (56 - sh));
    ee_st64(a & ~7u, (w & keep) | (rt << sh));
}

// ---- floating point (FPU and VU) ------------------------------------------------
EE_INLINE float ps2f(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t e = u & 0x7F800000u;
    u &= e ? 0xFFFFFFFFu : 0x80000000u;                 // denormal -> signed zero (zeros are common: no branch)
    if (EE_UNLIKELY(e == 0x7F800000u)) u = (u & 0x80000000u) | 0x7F7FFFFFu;   // inf/NaN -> max
    std::memcpy(&x, &u, 4);
    return x;
}
EE_INLINE float fadd(float a, float b) { return ps2f(ps2f(a) + ps2f(b)); }
EE_INLINE float fsub(float a, float b) { return ps2f(ps2f(a) - ps2f(b)); }
EE_INLINE float fmul(float a, float b) { return ps2f(ps2f(a) * ps2f(b)); }
EE_INLINE float fmax_(float a, float b) { a = ps2f(a); b = ps2f(b); return a > b ? a : b; }
EE_INLINE float fmin_(float a, float b) { a = ps2f(a); b = ps2f(b); return a < b ? a : b; }
// Division by zero gives the largest value with the sign of a ^ b; 0/0 too.
EE_INLINE float fdiv(float a, float b) {
    a = ps2f(a); b = ps2f(b);
    if (b == 0.0f) {
        uint32_t ua, ub; std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4);
        uint32_t u = ((ua ^ ub) & 0x80000000u) | 0x7F7FFFFFu;
        float r; std::memcpy(&r, &u, 4); return r;
    }
    return ps2f(a / b);
}
EE_INLINE float fsqrt(float a) { return ps2f(std::sqrt(std::fabs(ps2f(a)))); }
EE_INLINE float frsqrt(float a, float b) { return fdiv(a, fsqrt(b)); }
// cvt.w.s: truncation toward zero, out of range saturates.
EE_INLINE int32_t fcvtw(float a) {
    a = ps2f(a);
    if (a >= 2147483648.0f) return INT32_MAX;
    if (a <= -2147483648.0f) return INT32_MIN;
    return (int32_t)a;
}
EE_INLINE float fabs_(float a) { uint32_t u; std::memcpy(&u, &a, 4); u &= 0x7FFFFFFFu; std::memcpy(&a, &u, 4); return a; }
EE_INLINE float fneg(float a) { uint32_t u; std::memcpy(&u, &a, 4); u ^= 0x80000000u; std::memcpy(&a, &u, 4); return a; }
#define FPU_C (1u << 23)
EE_INLINE void fcmp(EEContext& c, bool v) { c.fcr31 = v ? (c.fcr31 | FPU_C) : (c.fcr31 & ~FPU_C); }

// VU0 macro mode: an FMAC result per lane, then the MAC flags (per lane:
// zero, sign; lanes outside `dest` clear) and the status flags (Z, S and
// their sticky copies) as the upper instructions set them.
EE_INLINE void vu_flags(EEContext& c, int dest, const VUReg& r) {
    uint32_t mac = 0;
    for (int k = 0; k < 4; ++k) {
        if (!(dest & (8 >> k))) continue;
        uint32_t bit = 8 >> k;                          // x is the high bit of each nibble
        if ((r.u[k] & 0x7FFFFFFFu) == 0) mac |= bit;
        if (r.u[k] & 0x80000000u) mac |= bit << 4;
    }
    c.vi[VI_MAC] = mac;
    uint32_t st = c.vi[VI_STATUS] & ~0xFu;              // FMAC sets Z S U O; the rest stays
    if (mac & 0x000F) st |= 0x001 | 0x040;
    if (mac & 0x00F0) st |= 0x002 | 0x080;
    c.vi[VI_STATUS] = st;
}
EE_INLINE void vu_clip(EEContext& c, const VUReg& s, float w) {
    w = std::fabs(ps2f(w));
    uint32_t f = 0;
    if (ps2f(s.f[0]) > w) f |= 0x01;
    if (ps2f(s.f[0]) < -w) f |= 0x02;
    if (ps2f(s.f[1]) > w) f |= 0x04;
    if (ps2f(s.f[1]) < -w) f |= 0x08;
    if (ps2f(s.f[2]) > w) f |= 0x10;
    if (ps2f(s.f[2]) < -w) f |= 0x20;
    c.vi[VI_CLIP] = ((c.vi[VI_CLIP] << 6) | f) & 0xFFFFFFu;
}
EE_INLINE float vu_q(EEContext& c) { float q; std::memcpy(&q, &c.vi[VI_Q], 4); return q; }
EE_INLINE float vu_i(EEContext& c) { float i; std::memcpy(&i, &c.vi[VI_I], 4); return i; }
EE_INLINE void vu_setq(EEContext& c, float q) { std::memcpy(&c.vi[VI_Q], &q, 4); }
EE_INLINE int32_t vu_ftoi(float a, float scale) {
    float v = ps2f(a) * scale;
    if (v >= 2147483648.0f) return INT32_MAX;
    if (v <= -2147483648.0f) return INT32_MIN;
    return (int32_t)v;
}
// MAX and MINI compare the bit patterns as sign-magnitude numbers (so they
// see denormals and both zeros as the console does) and set no flags.
EE_INLINE uint32_t vu_max(uint32_t a, uint32_t b) {
    int32_t x = (int32_t)a, y = (int32_t)b;
    return (uint32_t)((x < 0 && y < 0) ? (x < y ? x : y) : (x > y ? x : y));
}
EE_INLINE uint32_t vu_min(uint32_t a, uint32_t b) {
    int32_t x = (int32_t)a, y = (int32_t)b;
    return (uint32_t)((x < 0 && y < 0) ? (x > y ? x : y) : (x < y ? x : y));
}
// DIV, SQRT and RSQRT write Q and the status flags I (0x10) and D (0x20),
// with their sticky copies (0x400, 0x800): DIV sets I for 0/0 and D for
// x/0; SQRT sets I for a negative operand (and takes its magnitude); RSQRT
// sets D for x/sqrt(0), I for a negative operand or 0/sqrt(0), whose result
// is a signed zero. Division by zero gives the largest value with the sign
// of a ^ b. (The flag rules and RSQRT's 0/0 are PCSX2's.)
EE_INLINE void vu_divflags_vi(uint32_t* vi, uint32_t f) { vi[VI_STATUS] = (vi[VI_STATUS] & ~0x30u) | f | (f << 6); }
EE_INLINE void vu_divflags(EEContext& c, uint32_t f) { vu_divflags_vi(c.vi, f); }
EE_INLINE void vu_div(EEContext& c, float a, float b) {
    a = ps2f(a); b = ps2f(b);
    vu_divflags(c, b != 0.0f ? 0 : a == 0.0f ? 0x10 : 0x20);
    vu_setq(c, fdiv(a, b));
}
EE_INLINE void vu_sqrt(EEContext& c, float a) {
    a = ps2f(a);
    vu_divflags(c, a < 0.0f ? 0x10 : 0);
    vu_setq(c, fsqrt(a));
}
EE_INLINE void vu_rsqrt(EEContext& c, float a, float b) {
    a = ps2f(a); b = ps2f(b);
    if (b == 0.0f) {
        uint32_t ua, ub; std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4);
        uint32_t u = ((ua ^ ub) & 0x80000000u) | (a == 0.0f ? 0u : 0x7F7FFFFFu);
        float r; std::memcpy(&r, &u, 4);
        vu_divflags(c, a == 0.0f ? 0x30 : 0x20);
        vu_setq(c, r);
        return;
    }
    vu_divflags(c, b < 0.0f ? 0x10 : 0);
    vu_setq(c, frsqrt(a, b));
}
// VU0 data memory as lqi/sqi/lqd/sqd address it: in quadwords, 4 KB wrapping;
// with bit 0x400 set, VU1's registers (DW2 sets VU1's fog factor this way).
EE_INLINE VUReg* vu0_q(uint32_t qaddr) {
    if (EE_UNLIKELY(qaddr & 0x400)) return ee_vu0_vu1reg(qaddr);
    return (VUReg*)(g_vu0_mem + ((qaddr * 16) & 0xFFFu));
}

// ---- MMI ------------------------------------------------------------------------
EE_INLINE int16_t sat16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }
// qfsrv: the 256-bit rs:rt shifted right by SA bits (a multiple of 8), low 128.
EE_INLINE void ee_qfsrv(EEReg& d, const EEReg& rs, const EEReg& rt, uint32_t sa) {
    uint8_t buf[32];
    std::memcpy(buf, &rt, 16);
    std::memcpy(buf + 16, &rs, 16);
    std::memcpy(&d, buf + ((sa >> 3) & 15), 16);
}

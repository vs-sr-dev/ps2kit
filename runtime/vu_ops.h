// ps2kit runtime — what the micro-mode interpreter (vu.cpp) and VU1 microcode
// translated to C++ (vutrans) share: the FMAC's arithmetic, the flags as they
// land, the EFU's functions. One definition for both, so that a translated
// program computes the same bits as the interpreter.
#pragma once
#include "vu.h"

#include <cmath>
#include <cstring>
#include <emmintrin.h>

// The FMAC's add, subtract and multiply, with the host's DAZ and FTZ set while
// a program runs (VuDaz): a denormal operand counts as a signed zero and a
// denormal result becomes one, as ps2f makes them; with rounding toward zero
// a finite result never overflows to an infinity. Only an operand with
// exponent 255 (from memory) needs ps2f's clamp first.
EE_INLINE bool vu_exp255(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (u & 0x7F800000u) == 0x7F800000u; }
// (The rare clamped forms out of line, to keep the common one small.)
[[gnu::noinline, gnu::cold]] inline float vu_fadd_slow(float a, float b) { return fadd(a, b); }
[[gnu::noinline, gnu::cold]] inline float vu_fsub_slow(float a, float b) { return fsub(a, b); }
[[gnu::noinline, gnu::cold]] inline float vu_fmul_slow(float a, float b) { return fmul(a, b); }
EE_INLINE float vu_fadd(float a, float b) { return EE_UNLIKELY(vu_exp255(a) | vu_exp255(b)) ? vu_fadd_slow(a, b) : a + b; }
EE_INLINE float vu_fsub(float a, float b) { return EE_UNLIKELY(vu_exp255(a) | vu_exp255(b)) ? vu_fsub_slow(a, b) : a - b; }
EE_INLINE float vu_fmul(float a, float b) { return EE_UNLIKELY(vu_exp255(a) | vu_exp255(b)) ? vu_fmul_slow(a, b) : a * b; }

// MXCSR: DAZ (bit 6) and FTZ (bit 15) for as long as a program runs.
struct VuDaz {
    unsigned old;
    VuDaz() : old(_mm_getcsr()) { _mm_setcsr(old | 0x8040u); }
    ~VuDaz() { _mm_setcsr(old); }
};

// Whether any lane of v has exponent 255 (an operand the FMAC clamps first).
EE_INLINE bool vu_any255(__m128 v) {
    __m128i e = _mm_and_si128(_mm_castps_si128(v), _mm_set1_epi32(0x7F800000));
    return _mm_movemask_epi8(_mm_cmpeq_epi32(e, _mm_set1_epi32(0x7F800000))) != 0;
}

// The MAC flags of an FMAC result: per lane of dest (x is bit 3 of dest and
// the high bit of each nibble), zero in bits 3-0, sign in bits 7-4.
EE_INLINE uint32_t vu_mac(const VUReg& r, int dest) {
    static constexpr uint8_t REV[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};   // lane k -> bit 3-k
    __m128i v = _mm_loadu_si128((const __m128i*)r.u);
    __m128i zero = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128());
    uint32_t z = REV[_mm_movemask_ps(_mm_castsi128_ps(zero))], s = REV[_mm_movemask_ps(_mm_castsi128_ps(v))];
    return (z | s << 4) & (uint32_t)(dest | dest << 4);
}

// A flag write landing: MAC and status's Z, S (and their sticky copies), or
// the clip flags shifted on by one judgement.
// (On the VI registers as an array: translated programs keep theirs in locals.)
EE_INLINE void vu_land_mac_vi(uint32_t* vi, uint32_t mac) {
    vi[VI_MAC] = mac;
    uint32_t st = vi[VI_STATUS] & ~0xFu;                 // FMAC sets Z S U O and the sticky copies
    if (mac & 0x000F) st |= 0x001 | 0x040;
    if (mac & 0x00F0) st |= 0x002 | 0x080;
    vi[VI_STATUS] = st;
}
EE_INLINE void vu_land_clip_vi(uint32_t* vi, uint32_t bits) { vi[VI_CLIP] = ((vi[VI_CLIP] << 6) | bits) & 0xFFFFFF; }
EE_INLINE void vu_land_mac(EEContext& c, uint32_t mac) { vu_land_mac_vi(c.vi, mac); }
EE_INLINE void vu_land_clip(EEContext& c, uint32_t bits) { vu_land_clip_vi(c.vi, bits); }

// A register's bits as a float (Q, I).
EE_INLINE float vu_bits_f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// The EFU's result for instruction x (its lower word's bits 10-6 and 1-0), from
// fs (fsf: the lane of the one-operand forms). Computed in host single
// precision; ESIN, EATAN and EEXP with the host's library, as the VU manual
// defines them (sin x, atan x, exp(-x)): the console's own series may differ
// in the last bits.
EE_INLINE uint32_t vu_efu(uint32_t x, const VUReg& s, int fsf) {
    auto f = [&](int k) { return ps2f(s.f[k]); };
    float a = f(fsf), r = 0.0f;
    float sq = fadd(fadd(fmul(f(0), f(0)), fmul(f(1), f(1))), fmul(f(2), f(2)));
    switch (x) {
    case 0x70: r = sq; break;                                                     // esadd
    case 0x71: r = fdiv(1.0f, sq); break;                                         // ersadd
    case 0x72: r = fsqrt(sq); break;                                              // eleng
    case 0x73: r = frsqrt(1.0f, sq); break;                                       // erleng
    case 0x74: r = f(0) != 0.0f ? ps2f(std::atan(fdiv(f(1), f(0)))) : 0.0f; break;   // eatanxy: atan(y/x)
    case 0x75: r = f(0) != 0.0f ? ps2f(std::atan(fdiv(f(2), f(0)))) : 0.0f; break;   // eatanxz: atan(z/x)
    case 0x76: r = fadd(fadd(fadd(f(0), f(1)), f(2)), f(3)); break;               // esum
    case 0x78: r = fsqrt(a); break;                                               // esqrt
    case 0x79: r = frsqrt(1.0f, a); break;                                        // ersqrt
    case 0x7A: r = fdiv(1.0f, a); break;                                          // ercpr
    case 0x7C: r = ps2f(std::sin(a)); break;                                      // esin
    case 0x7D: r = ps2f(std::atan(a)); break;                                     // eatan
    case 0x7E: r = ps2f(std::exp(-a)); break;                                     // eexp
    }
    uint32_t v;
    std::memcpy(&v, &r, 4);
    return v;
}

// DIV, SQRT, RSQRT as they enter the FDIV pipeline: Q and the D and I flags
// they will write (which depend on the operands alone), c left as it was.
EE_INLINE void vu_fdiv(EEContext& c, uint32_t x, float s, float t, uint32_t& q, uint32_t& flags) {
    uint32_t q0 = c.vi[VI_Q], st0 = c.vi[VI_STATUS];
    if (x == 0x38) vu_div(c, s, t);
    else if (x == 0x39) vu_sqrt(c, t);
    else vu_rsqrt(c, s, t);
    q = c.vi[VI_Q];
    flags = c.vi[VI_STATUS] & 0x30;
    c.vi[VI_Q] = q0;
    c.vi[VI_STATUS] = st0;
}
inline void vu_fdiv_vals(uint32_t x, float s, float t, uint32_t& q, uint32_t& flags) {
    thread_local EEContext scratch;
    vu_fdiv(scratch, x, s, t, q, flags);
}

// VU1 data memory at a quadword address.
EE_INLINE VUReg* vu1_q(VUUnit& vu, uint32_t qaddr) { return (VUReg*)(vu.mem + ((qaddr * 16) & 0x3FFF)); }

// A VU1 program translated by vutrans, for the micro memory it was translated
// from: `code` lists the instruction pairs it covers, as ranges
// {start (bytes), pairs, then the pairs' two words each}, ended by a range of
// 0 pairs. It is used only while micro memory holds exactly those words.
struct VUTrans {
    uint32_t entry;
    void (*fn)(VUUnit&);
    const uint32_t* code;
};

// The game's translations (its build's vu1_trans.cpp, or vu1_none.cpp: none).
extern const VUTrans g_vu1_trans[];
extern const size_t g_vu1_ntrans;

// The translation of the program at pc for what micro memory holds, if any.
inline void (*vu1_trans_find(const uint8_t* micro, uint32_t pc))(VUUnit&) {
    for (size_t k = 0; k < g_vu1_ntrans; ++k) {
        const VUTrans& t = g_vu1_trans[k];
        if (t.entry != (pc & 0x3FFF)) continue;
        bool same = true;
        for (const uint32_t* c = t.code; same && c[1]; c += 2 + 2 * c[1])
            same = std::memcmp(micro + c[0], c + 2, (size_t)c[1] * 8) == 0;
        if (same) return t.fn;
    }
    return nullptr;
}

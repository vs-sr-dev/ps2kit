// ps2kit runtime — roundcheck: the fast rasteriser's bilinear rounding
// (sample_fast, gs.cpp) against the reference's (sample_with), for every
// float r in [0, 256), the range a filtered channel takes:
//   reference  (uint32_t)((double)r + 0.5)
//   fast       i = (int)r, then i + 1 where r - i >= 0.5, in SSE2 as there
// Exit status 0 when they agree everywhere. (r + 0.5f truncated, the
// obvious single-precision form, differs at 0.49999997.)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <emmintrin.h>

int main() {
    uint64_t n = 0, bad = 0;
    for (uint32_t bits = 0; bits < 0x43800000u; bits += 4) {   // 0x43800000 is 256.0f
        uint32_t b4[4] = {bits, bits + 1, bits + 2, bits + 3};
        __m128 r;
        std::memcpy(&r, b4, 16);
        __m128i i = _mm_cvttps_epi32(r);
        __m128 rest = _mm_sub_ps(r, _mm_cvtepi32_ps(i));
        i = _mm_sub_epi32(i, _mm_castps_si128(_mm_cmpge_ps(rest, _mm_set1_ps(0.5f))));
        uint32_t got[4];
        std::memcpy(got, &i, 16);
        for (int k = 0; k < 4; ++k) {
            float x;
            std::memcpy(&x, &b4[k], 4);
            uint32_t want = (uint32_t)((double)x + 0.5);
            if (want != got[k] && bad++ < 5) std::printf("differs at %08X (%.9g): %u, %u\n", b4[k], x, want, got[k]);
            ++n;
        }
    }
    std::printf("roundcheck: %llu floats in [0, 256), %llu differ\n", (unsigned long long)n, (unsigned long long)bad);
    return bad != 0;
}

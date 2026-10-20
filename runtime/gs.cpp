// ps2kit runtime — the Graphics Synthesizer: local memory, the GIF, drawing,
// and the PCRTC's picture.
//
// Local memory is 4 MB of 256-byte blocks in 8 KB pages; each pixel format
// lays a page out its own way. The layouts are ps2kit.gsmem's (checked there
// against a real disc's textures), ported: PSMCT32/24 and the formats that
// share its words (PSMT8H, PSMT4HL, PSMT4HH), PSMCT16, PSMT8, PSMT4. The Z
// formats and PSMCT16S are stored with the PSMCT32 and PSMCT16 layouts until
// their own tables are written down (a game that reads a Z buffer as a
// texture would show it).
//
// The GIF takes packets on path 3 (DMA channel 2, or the CPU at 0x10006000),
// path 2 (VIF1's DIRECT and DIRECTHL) and path 1 (VU1's XGKICK), each path with its own packet in
// progress (the GIF changes paths only between packets), into the one GS:
// a GIFtag (NLOOP, EOP, PRE/PRIM, FLG, NREG, REGS), then PACKED, REGLIST or
// IMAGE data. PACKED descriptors are unpacked as the GS manual gives them
// (RGBAQ's bytes at 0/32/64/96, ST and its Q, XYZF2's Z in 91:68 and F in
// 107:100, ADC at bit 111 for "no kick", A+D's register in 71:64).
//
// Drawing is a software rasteriser: points, lines, triangles (and strips,
// fans) and sprites; flat and Gouraud colour; textures (ST/Q or UV; nearest
// or bilinear; the clamp modes; TFX modulate, decal, highlight 1 and 2;
// CLUTs in CSM1, loaded into the CLUT buffer as TEX0 and TEX2's CLD asks);
// fog; alpha,
// destination alpha and depth tests; alpha blending with PABE and COLCLAMP;
// FBA; FBMSK and ZMSK; scissor. Pixels are sampled at integer coordinates,
// with exact edge functions and a top-left fill rule. It comes in two forms
// that must write the same memory: a reference, pixel by pixel with the
// registers read at each, and the fast one the runtime uses (see "drawing").
// What is not done yet is said once when first used.
//
// The GS runs on a thread of its own, beside the EE (see "the GS's own
// thread"); PS2KIT_GS_DUMP records what it is given, for gsreplay.
//
// The PCRTC: PMODE, SMODE2, DISPFBn and DISPLAYn decide which part of
// memory is shown; gs_snapshot builds that picture for the window.
#include "os.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <set>
#include <type_traits>
#include <vector>

namespace {

constexpr uint32_t VRAM = 4u << 20;
std::vector<uint8_t> g_vram(VRAM);

thread_local bool t_worker;                          // a drawing thread (see "on several threads")

void once(const char* what) {
    if (t_worker) return;                            // what it would say, the GS's thread says too
    static std::set<std::string> said;
    if (said.insert(what).second) std::fprintf(stderr, "ps2kit: GS: %s is not modelled yet\n", what);
}

// ---- local memory ------------------------------------------------------------------------------
enum {
    PSMCT32 = 0x00, PSMCT24 = 0x01, PSMCT16 = 0x02, PSMCT16S = 0x0A, PSMT8 = 0x13, PSMT4 = 0x14,
    PSMT8H = 0x1B, PSMT4HL = 0x24, PSMT4HH = 0x2C, PSMZ32 = 0x30, PSMZ24 = 0x31, PSMZ16 = 0x32, PSMZ16S = 0x3A,
};

const uint8_t TABLE_A[4][8] = {{0, 1, 4, 5, 16, 17, 20, 21}, {2, 3, 6, 7, 18, 19, 22, 23},
                               {8, 9, 12, 13, 24, 25, 28, 29}, {10, 11, 14, 15, 26, 27, 30, 31}};
const uint8_t TABLE_B[8][4] = {{0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15},
                               {16, 18, 24, 26}, {17, 19, 25, 27}, {20, 22, 28, 30}, {21, 23, 29, 31}};

// Per base layout: page size, and the bit offset of every pixel of a page
// from the page's start (block number * 2048 + offset inside the block).
struct Layout {
    int pw, ph, lpw, lph;                            // page size, and its log2
    std::vector<uint32_t> off;                       // [y * pw + x]
};
Layout g_lay[4];                                     // 0 CT32, 1 CT16, 2 T8, 3 T4

uint32_t word32(int x, int y) { return (uint32_t)((x >> 1 & 3) << 2 | (y & 1) << 1 | (x & 1)); }

void build_layouts() {
    for (int b = 0; b < 4; ++b) {
        static const int PW[4] = {64, 64, 128, 128}, PH[4] = {32, 64, 64, 128};
        static const int BW[4] = {8, 16, 16, 32}, BH[4] = {8, 8, 16, 16};
        Layout& l = g_lay[b];
        l.pw = PW[b];
        l.ph = PH[b];
        l.lpw = b < 2 ? 6 : 7;
        l.lph = b == 0 ? 5 : b == 3 ? 7 : 6;
        l.off.resize((size_t)l.pw * l.ph);
        for (int y = 0; y < l.ph; ++y)
            for (int x = 0; x < l.pw; ++x) {
                int bx = x / BW[b], by = y / BH[b], ix = x % BW[b], iy = y % BH[b];
                uint32_t block = (b == 0 || b == 2) ? TABLE_A[by][bx] : TABLE_B[by][bx];
                uint32_t bits;
                if (b == 0) bits = ((uint32_t)(iy >> 1) * 16 + word32(ix, iy)) * 32;
                else if (b == 1) bits = ((uint32_t)(iy >> 1) * 16 + word32(ix & 7, iy)) * 32 + (uint32_t)(ix >> 3) * 16;
                else {
                    int col = iy >> 2;
                    int shift = (((iy >> 1) & 1) != (col & 1)) ? 4 : 0;
                    uint32_t word = (uint32_t)col * 16 + word32((ix + shift) & 7, iy);
                    uint32_t sub = (uint32_t)((ix >> 3) * 2 + ((iy >> 1) & 1));
                    bits = word * 32 + sub * (b == 2 ? 8 : 4);
                }
                l.off[(size_t)y * l.pw + x] = block * 2048 + bits;
            }
    }
}

constexpr int base_of(uint32_t psm) {
    switch (psm) {
    case PSMCT32: case PSMCT24: case PSMT8H: case PSMT4HL: case PSMT4HH: case PSMZ32: case PSMZ24: return 0;
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: return 1;
    case PSMT8: return 2;
    case PSMT4: return 3;
    }
    return -1;
}

uint32_t rd32(uint32_t bits) { uint32_t v; std::memcpy(&v, &g_vram[bits >> 3], 4); return v; }
void wr32(uint32_t bits, uint32_t v) { std::memcpy(&g_vram[bits >> 3], &v, 4); }
uint16_t rd16(uint32_t bits) { uint16_t v; std::memcpy(&v, &g_vram[bits >> 3], 2); return v; }
void wr16(uint32_t bits, uint16_t v) { std::memcpy(&g_vram[bits >> 3], &v, 2); }

// A buffer in local memory (block bp, bw * 64 pixels wide, a format), its
// addressing decoded once for many pixels.
struct Surf {
    uint32_t psm = 0;
    const Layout* l = nullptr;                       // null: a format without a layout
    uint32_t bp_bits = 0, ppr = 1;                   // bp * 2048; pages per row
    // what reads come from: local memory, or a drawing job's copy of its
    // texture (snap_texture), placed so that the same addresses reach it
    const uint8_t* mem = g_vram.data();

    // The bit address of (x, y). P, when given, is the format, known to be psm,
    // so that the compiler folds the layout's constants and the format's
    // switch (the fast rasteriser); 0 reads them here.
    template <uint32_t P = 0> uint32_t at(int x, int y) const {
        uint32_t px = (uint32_t)x & 2047, py = (uint32_t)y & 2047;
        if constexpr (P == 0) {
            uint32_t page = (py >> l->lph) * ppr + (px >> l->lpw);
            uint32_t bits = bp_bits + page * 65536 + l->off[(py & (l->ph - 1)) << l->lpw | (px & (l->pw - 1))];
            return bits & (VRAM * 8 - 1);
        } else {
            constexpr int b = base_of(P);
            constexpr int lpw = b < 2 ? 6 : 7, lph = b == 0 ? 5 : b == 3 ? 7 : 6;
            uint32_t page = (py >> lph) * ppr + (px >> lpw);
            uint32_t bits = bp_bits + page * 65536 + l->off[(py & ((1u << lph) - 1)) << lpw | (px & ((1u << lpw) - 1))];
            return bits & (VRAM * 8 - 1);
        }
    }
    // A pixel's stored value in its format.
    template <uint32_t P = 0> uint32_t read(int x, int y) const {
        uint32_t a = at<P>(x, y);
        auto r32 = [&] { uint32_t v; std::memcpy(&v, mem + (a >> 3), 4); return v; };
        switch (P ? P : psm) {
        case PSMCT32: case PSMZ32: return r32();
        case PSMCT24: case PSMZ24: return r32() & 0xFFFFFF;
        case PSMT8H: return r32() >> 24;
        case PSMT4HL: return r32() >> 24 & 15;
        case PSMT4HH: return r32() >> 28;
        case PSMT8: return mem[a >> 3];
        case PSMT4: return (uint32_t)(mem[a >> 3] >> (a & 4)) & 15;
        default: { uint16_t v; std::memcpy(&v, mem + (a >> 3), 2); return v; }
        }
    }
    // keep: the bits left as they are (FBMSK).
    template <uint32_t P = 0> void write(int x, int y, uint32_t v, uint32_t keep) const {
        uint32_t a = at<P>(x, y);
        switch (P ? P : psm) {
        case PSMCT32: case PSMZ32: wr32(a, (rd32(a) & keep) | (v & ~keep)); break;
        case PSMCT24: case PSMZ24: wr32(a, (rd32(a) & (0xFF000000u | keep)) | (v & 0xFFFFFF & ~keep)); break;
        case PSMT8H: wr32(a, (rd32(a) & 0x00FFFFFF) | (v & 0xFF) << 24); break;
        case PSMT4HL: wr32(a, (rd32(a) & 0xF0FFFFFF) | (v & 15) << 24); break;
        case PSMT4HH: wr32(a, (rd32(a) & 0x0FFFFFFF) | (v & 15) << 28); break;
        case PSMT8: g_vram[a >> 3] = (uint8_t)v; break;
        case PSMT4: {
            uint8_t& b = g_vram[a >> 3];
            b = (uint8_t)((b & (0xF0 >> (a & 4))) | (v & 15) << (a & 4));
            break;
        }
        default: wr16(a, (uint16_t)((rd16(a) & keep) | (v & ~keep))); break;
        }
    }
};

Surf surf(uint32_t psm, uint32_t bp, uint32_t bw) {
    Surf s;
    s.psm = psm;
    int base = base_of(psm);
    if (base < 0) return s;
    s.l = &g_lay[base];
    s.bp_bits = bp * 2048;
    s.ppr = std::max<uint32_t>(1, (bw * 64) >> s.l->lpw);
    return s;
}

uint32_t pix_read(uint32_t psm, uint32_t bp, uint32_t bw, int x, int y) {
    Surf s = surf(psm, bp, bw);
    if (!s.l) { once("a pixel format"); return 0; }
    return s.read(x, y);
}

void pix_write(uint32_t psm, uint32_t bp, uint32_t bw, int x, int y, uint32_t v, uint32_t keep = 0) {
    Surf s = surf(psm, bp, bw);
    if (!s.l) { once("a pixel format"); return; }
    s.write(x, y, v, keep);
}

// ---- registers --------------------------------------------------------------------------------
uint64_t g_r[0x100];                                 // the general registers, by address
enum {
    PRIM = 0x00, RGBAQ = 0x01, ST = 0x02, UV = 0x03, XYZF2 = 0x04, XYZ2 = 0x05, TEX0_1 = 0x06, TEX0_2 = 0x07,
    CLAMP_1 = 0x08, CLAMP_2 = 0x09, FOG = 0x0A, XYZF3 = 0x0C, XYZ3 = 0x0D, TEX1_1 = 0x14, TEX1_2 = 0x15,
    TEX2_1 = 0x16, TEX2_2 = 0x17, XYOFFSET_1 = 0x18, XYOFFSET_2 = 0x19, PRMODECONT = 0x1A, PRMODE = 0x1B,
    TEXCLUT = 0x1C, SCANMSK = 0x22, TEXA = 0x3B, FOGCOL = 0x3D, TEXFLUSH = 0x3F, SCISSOR_1 = 0x40,
    SCISSOR_2 = 0x41, ALPHA_1 = 0x42, ALPHA_2 = 0x43, DIMX = 0x44, DTHE = 0x45, COLCLAMP = 0x46, TEST_1 = 0x47,
    TEST_2 = 0x48, PABE = 0x49, FBA_1 = 0x4A, FBA_2 = 0x4B, FRAME_1 = 0x4C, FRAME_2 = 0x4D, ZBUF_1 = 0x4E,
    ZBUF_2 = 0x4F, BITBLTBUF = 0x50, TRXPOS = 0x51, TRXREG = 0x52, TRXDIR = 0x53, HWREG = 0x54,
    SIGNAL = 0x60, FINISH = 0x61, LABEL = 0x62,
};
uint64_t field(uint64_t v, int lo, int bits) { return (v >> lo) & ((1ull << bits) - 1); }

float g_q = 1.0f;                                    // ST's Q, kept for RGBAQ (PACKED)

// The fast rasteriser's worker threads (see "on several threads"): what the
// GS's own thread must wait for before it touches local memory.
enum { W_OTHER, W_TEXTURE, W_TARGET, W_XFER, W_CLUT, W_ALONE, W_READ, W_N };   // why a barrier waits
void barrier(int why = W_OTHER);
void hazard_rect(uint32_t psm, uint32_t bp, uint32_t bw, int x0, int y0, int x1, int y1, bool write);
void hazard_clut(uint32_t psm, uint32_t cbp, uint32_t n);
bool hazard_xfer_start(uint32_t psm, uint32_t bp, uint32_t bw, int x0, int y0, int w, int h);
void dirty(uint32_t lo, uint32_t hi);                 // the GS's thread writes (or will) bytes [lo, hi)
void dirty_all();
void dirty_xfer();                                   // the host-to-local transfer's destination
bool hazard_xfer();
void defer_xfer(const uint8_t* q, size_t n);
extern uint64_t g_xmark, g_next;
extern int g_nw;

// ---- host-to-local transfers ----------------------------------------------------------------------
struct Xfer {
    bool active = false;
    uint32_t psm, bp, bw;
    int x0, y0, w, h, x, y;
    std::vector<uint8_t> pend;                       // bytes of a pixel split between quadwords
    Surf s;                                          // the destination's addressing
    bool deferred = false;                           // written by the drawing threads, in their order
} g_x;

struct Readback {                                    // local to host, in progress
    bool active = false;
    uint32_t psm = 0, bp = 0, bw = 0;
    int x0 = 0, y0 = 0, w = 0, h = 0, x = 0, y = 0;
    std::vector<uint8_t> pend;
} g_rb;

void xfer_start() {
    uint64_t blt = g_r[BITBLTBUF], pos = g_r[TRXPOS], reg = g_r[TRXREG];
    uint32_t dir = (uint32_t)field(g_r[TRXDIR], 0, 2);
    int w = (int)field(reg, 0, 12), h = (int)field(reg, 32, 12);
    if (dir == 0) {                                  // host to local
        g_x = Xfer{};
        g_x.active = true;
        g_x.psm = (uint32_t)field(blt, 56, 6);
        g_x.bp = (uint32_t)field(blt, 32, 14);
        g_x.bw = (uint32_t)field(blt, 48, 6);
        g_x.x0 = g_x.x = (int)field(pos, 32, 11);
        g_x.y0 = g_x.y = (int)field(pos, 48, 11);
        g_x.w = w;
        g_x.h = h;
        g_x.s = surf(g_x.psm, g_x.bp, g_x.bw);
        if (!g_x.s.l) once("a pixel format");
        g_x.deferred = hazard_xfer_start(g_x.psm, g_x.bp, g_x.bw, g_x.x0, g_x.y0, w, h);
        TRACE(2, "[gs] host to local: %ux%u at (%d,%d), bp %X bw %u psm %02X\n", w, h, g_x.x0, g_x.y0, g_x.bp,
              g_x.bw, g_x.psm);
    } else if (dir == 2) {                           // local to local
        barrier();
        dirty_all();
        uint32_t spsm = (uint32_t)field(blt, 24, 6), sbp = (uint32_t)field(blt, 0, 14), sbw = (uint32_t)field(blt, 16, 6);
        uint32_t dpsm = (uint32_t)field(blt, 56, 6), dbp = (uint32_t)field(blt, 32, 14), dbw = (uint32_t)field(blt, 48, 6);
        int sx = (int)field(pos, 0, 11), sy = (int)field(pos, 16, 11);
        int dx = (int)field(pos, 32, 11), dy = (int)field(pos, 48, 11);
        if (field(pos, 59, 2)) once("local-to-local transfer directions");
        std::vector<uint32_t> tmp((size_t)w * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) tmp[(size_t)y * w + x] = pix_read(spsm, sbp, sbw, sx + x, sy + y);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) pix_write(dpsm, dbp, dbw, dx + x, dy + y, tmp[(size_t)y * w + x]);
    } else if (dir == 1) {                           // local to host: read out through VIF1 (gs_readback)
        g_rb = Readback{};
        g_rb.active = true;
        g_rb.psm = (uint32_t)field(blt, 24, 6);
        g_rb.bp = (uint32_t)field(blt, 0, 14);
        g_rb.bw = (uint32_t)field(blt, 16, 6);
        g_rb.x0 = g_rb.x = (int)field(pos, 0, 11);
        g_rb.y0 = g_rb.y = (int)field(pos, 16, 11);
        g_rb.w = w;
        g_rb.h = h;
        TRACE(2, "[gs] local to host: %ux%u at (%d,%d), bp %X bw %u psm %02X\n", w, h, g_rb.x0, g_rb.y0, g_rb.bp,
              g_rb.bw, g_rb.psm);
    }
}

// The bytes of a local-to-host transfer, in the transfer's format as the
// host-to-local direction takes them: pixels row by row from the source
// rectangle, 4, 3, 2 or 1 bytes each, two 4-bit pixels a byte (the first in
// the low half). Past its end, zeros.
void readback_fill(uint8_t* dst, size_t n) {
    barrier();
    size_t k = 0;
    while (k < n) {
        if (!g_rb.pend.empty()) {
            dst[k++] = g_rb.pend.front();
            g_rb.pend.erase(g_rb.pend.begin());
            continue;
        }
        if (!g_rb.active) { dst[k++] = 0; continue; }
        auto next = [&]() {
            uint32_t v = pix_read(g_rb.psm, g_rb.bp, g_rb.bw, g_rb.x, g_rb.y);
            if (++g_rb.x >= g_rb.x0 + g_rb.w) {
                g_rb.x = g_rb.x0;
                if (++g_rb.y >= g_rb.y0 + g_rb.h) g_rb.active = false;
            }
            return v;
        };
        size_t bpp;
        switch (g_rb.psm) {
        case PSMCT32: case PSMZ32: bpp = 4; break;
        case PSMCT24: case PSMZ24: bpp = 3; break;
        case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: bpp = 2; break;
        case PSMT8: case PSMT8H: bpp = 1; break;
        case PSMT4: case PSMT4HL: case PSMT4HH: {
            uint32_t lo = next(), hi = g_rb.active ? next() : 0;
            g_rb.pend.push_back((uint8_t)((lo & 15) | (hi & 15) << 4));
            continue;
        }
        default: once("this local-to-host format"); g_rb.active = false; continue;
        }
        uint32_t v = next();
        for (size_t b = 0; b < bpp; ++b) g_rb.pend.push_back((uint8_t)(v >> (8 * b)));
    }
}

// A transfer's pixels, into X's destination: written (W) or only counted, the
// GS's thread following a transfer the drawing threads write (defer_xfer).
template <bool W, uint32_t P = 0> void xfer_pixel(Xfer& X, uint32_t v) {
    if (!X.active) return;
    if (W && (P != 0 || X.s.l)) X.s.write<P>(X.x, X.y, v, 0);
    if (++X.x >= X.x0 + X.w) {
        X.x = X.x0;
        if (++X.y >= X.y0 + X.h) X.active = false;
    }
}

// Whole pixels of a format the compiler knows, no bytes pending.
template <bool W, uint32_t P> void xfer_whole(Xfer& X, const uint8_t* q, size_t n) {
    if constexpr (P == PSMT4) {
        for (size_t k = 0; k < n && X.active; ++k) { xfer_pixel<W, P>(X, q[k] & 15); xfer_pixel<W, P>(X, q[k] >> 4); }
    } else {
        constexpr size_t bpp = P == PSMCT32 ? 4 : P == PSMCT16 ? 2 : 1;
        for (size_t k = 0; k + bpp <= n && X.active; k += bpp) {
            uint32_t v = 0;
            std::memcpy(&v, q + k, bpp);
            xfer_pixel<W, P>(X, v);
        }
    }
}

// Pixels of bpp bytes, little-endian; one split by the data's end waits in
// pend for the rest.
template <bool W, uint32_t P> void xfer_bytes(Xfer& X, const uint8_t* q, size_t n, size_t bpp) {
    size_t k = 0;
    while (!X.pend.empty() && k < n && X.active) {
        X.pend.push_back(q[k++]);
        if (X.pend.size() == bpp) {
            uint32_t v = 0;
            for (size_t b = 0; b < bpp; ++b) v |= (uint32_t)X.pend[b] << (8 * b);
            X.pend.clear();
            xfer_pixel<W, P>(X, v);
        }
    }
    for (; k + bpp <= n && X.active; k += bpp) {
        uint32_t v = 0;
        std::memcpy(&v, q + k, bpp);
        xfer_pixel<W, P>(X, v);
    }
    for (; k < n && X.active; ++k) X.pend.push_back(q[k]);
}

template <bool W> void xfer_run(Xfer& X, const uint8_t* q, size_t n) {   // n bytes of IMAGE / HWREG data
    if (!X.active) return;
    if (X.pend.empty() && X.s.l) {
        switch (X.psm) {
        case PSMCT32: if (n % 4 == 0) { xfer_whole<W, PSMCT32>(X, q, n); return; } break;
        case PSMCT16: if (n % 2 == 0) { xfer_whole<W, PSMCT16>(X, q, n); return; } break;
        case PSMT8: xfer_whole<W, PSMT8>(X, q, n); return;
        case PSMT4: xfer_whole<W, PSMT4>(X, q, n); return;
        }
    }
    size_t bpp = 0;
    switch (X.psm) {
    case PSMCT32: case PSMZ32: bpp = 4; break;
    case PSMCT24: case PSMZ24: bpp = 3; break;
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: bpp = 2; break;
    case PSMT8: case PSMT8H: bpp = 1; break;
    case PSMT4: case PSMT4HL: case PSMT4HH:
        for (size_t k = 0; k < n && X.active; ++k) { xfer_pixel<W>(X, q[k] & 15); xfer_pixel<W>(X, q[k] >> 4); }
        return;
    default: once("this transfer format"); return;
    }
    if (X.psm == PSMCT24 && X.s.l) xfer_bytes<W, PSMCT24>(X, q, n, bpp);
    else xfer_bytes<W, 0>(X, q, n, bpp);
}

// The GS's thread's transfer: written here, unless pending jobs use its
// destination; then the drawing threads write it in their order (defer_xfer)
// and the GS's thread only follows where it has got to.
void xfer_data(const uint8_t* q, size_t n) {
    if (!g_x.active) return;
    dirty_xfer();
    if (g_nw && !g_x.deferred && g_next != g_xmark) g_x.deferred = hazard_xfer();
    if (g_x.deferred) {
        defer_xfer(q, n);
        xfer_run<false>(g_x, q, n);
    } else {
        xfer_run<true>(g_x, q, n);
    }
}

// ---- drawing ------------------------------------------------------------------------------------
// Two rasterisers draw triangles and sprites. The reference takes every
// pixel through put(), which reads the registers there; the fast one, the
// default, decodes them once per primitive (Draw) and steps its edge
// functions along each row. gsreplay --compare checks that both write the
// same memory. What decides a pixel's value they share: coverage from exact
// edge functions, the interpolation (tri_attr, sprite_attr), texel formats
// and addressing (Surf). Lines and points always take the reference's path.
struct Vtx {
    int X, Y;                                        // window coordinates, 12.4 fixed point
    float x, y;                                      // the same in pixels
    double z;
    float r, g, b, a, f;
    float s, t, q, u, v;                             // u, v in texels
};
Vtx g_v[3];
int g_nv;                                            // vertices in the queue
int g_strip_k;                                       // vertices since the primitive began

uint64_t g_draw_px;                                  // pixels drawn (heartbeat)
bool g_ref;                                          // the reference rasteriser (gsreplay --ref)

struct Tex {
    uint32_t tbp, tbw, psm, tw, th, tcc, tfx, cbp, cpsm, csm, csa;
    uint32_t wms, wmt, minu, maxu, minv, maxv;
    bool linear;
    uint32_t ta0, ta1, aem;
};

uint32_t rgba_from(uint32_t v, uint32_t psm, const Tex& t) {   // texel in RGBA (A 0-255 as stored, 0x80 = 1.0)
    switch (psm) {
    case PSMCT32: return v;
    case PSMCT24: {
        uint32_t a = (t.aem && (v & 0xFFFFFF) == 0) ? 0 : t.ta0;
        return (v & 0xFFFFFF) | a << 24;
    }
    default: {                                       // RGB5A1
        uint32_t r = (v & 31) << 3, g = (v >> 5 & 31) << 3, b = (v >> 10 & 31) << 3;
        uint32_t a = (v & 0x8000) ? t.ta1 : ((t.aem && (v & 0x7FFF) == 0) ? 0 : t.ta0);
        return r | g << 8 | b << 16 | a << 24;
    }
    }
}

bool indexed(uint32_t psm) { return psm == PSMT8 || psm == PSMT8H || psm == PSMT4 || psm == PSMT4HL || psm == PSMT4HH; }

// The CLUT buffer. A CLUT is not read from memory as it is drawn with: the GS
// loads it into its own buffer when TEX0 (or TEX2) is written, as CLD says
// (0 no; 1 yes; 2, 3 yes and CBP0 or CBP1 = CBP; 4, 5 only when CBP differs
// from CBP0 or CBP1, which then becomes CBP), and only for an indexed PSM.
// CSM1: the CLUT at CBP is a 16 x 16 rectangle (8-bit textures, entries 8-15
// and 16-23 of every 32 swapped) or 8 x 2 (4-bit), stored from entry CSA * 16
// of the buffer, which a texture reads from the same entry on. Entries are
// kept as stored (16 or 32 bits); 32-bit CLUTs take 256 entries, 16-bit 512.
// (The rules are the GS manual's; the offsets as PCSX2 applies them.)
uint32_t g_clut[512];
uint32_t g_cbp0, g_cbp1;
uint64_t g_clut_gen;                                 // loads so far (the fast rasteriser's cache)

bool four_bit(uint32_t psm) { return psm == PSMT4 || psm == PSMT4HL || psm == PSMT4HH; }

uint32_t clut_base(uint32_t cpsm, uint32_t csa) { return cpsm == PSMCT32 ? (csa & 15) * 16 : csa * 16; }

void clut_load(uint64_t t0) {
    uint32_t psm = (uint32_t)field(t0, 20, 6), cbp = (uint32_t)field(t0, 37, 14), cpsm = (uint32_t)field(t0, 51, 4);
    uint32_t csm = (uint32_t)field(t0, 55, 1), csa = (uint32_t)field(t0, 56, 5), cld = (uint32_t)field(t0, 61, 3);
    if ((psm & 7) < 3) return;                       // not indexed: no load, CBP0 and CBP1 kept
    switch (cld) {
    case 1: break;
    case 2: g_cbp0 = cbp; break;
    case 3: g_cbp1 = cbp; break;
    case 4: if (g_cbp0 == cbp) return; g_cbp0 = cbp; break;
    case 5: if (g_cbp1 == cbp) return; g_cbp1 = cbp; break;
    default: return;
    }
    if (csm) once("CLUT storage mode CSM2");
    uint32_t mpsm = cpsm == PSMCT32 ? PSMCT32 : PSMCT16;
    uint32_t n = four_bit(psm) ? 16 : 256, size = cpsm == PSMCT32 ? 256 : 512, base = clut_base(cpsm, csa);
    hazard_clut(mpsm, cbp, n);
    for (uint32_t k = 0; k < n; ++k) {
        int x, y;
        if (n == 16) { x = (int)(k & 7); y = (int)(k >> 3); }
        else {
            uint32_t j = (k & ~0x18u) | ((k & 8) << 1) | ((k & 16) >> 1);
            x = (int)(j & 15); y = (int)(j >> 4);
        }
        g_clut[(base + k) % size] = pix_read(mpsm, cbp, 1, x, y);
    }
    ++g_clut_gen;
}

uint32_t clut_entry(const Tex& t, uint32_t idx) {
    uint32_t size = t.cpsm == PSMCT32 ? 256 : 512;
    uint32_t v = g_clut[(clut_base(t.cpsm, t.csa) + idx) % size];
    return t.cpsm == PSMCT32 ? v : rgba_from(v, PSMCT16, t);
}

int wrap(int c, int size, uint32_t mode, uint32_t mn, uint32_t mx) {
    switch (mode) {
    case 0: return c & (size - 1);                           // REPEAT
    case 1: return std::max(0, std::min(size - 1, c));       // CLAMP
    case 2: return std::max((int)mn, std::min((int)mx, c));  // REGION_CLAMP
    default: return (int)(((uint32_t)c & mn) | mx);          // REGION_REPEAT
    }
}

uint32_t texel(const Tex& t, int u, int v) {
    u = wrap(u, 1 << t.tw, t.wms, t.minu, t.maxu);
    v = wrap(v, 1 << t.th, t.wmt, t.minv, t.maxv);
    uint32_t raw = pix_read(t.psm, t.tbp, t.tbw, u, v);
    return indexed(t.psm) ? clut_entry(t, raw) : rgba_from(raw, t.psm, t);
}

int ifloor(float f) { int i = (int)f; return i - (f < (float)i); }   // (int)std::floor(f)

// Nearest or bilinear, from texels given by `tex` (u, v in texels).
template <class F> uint32_t sample_with(const Tex& t, float u, float v, const F& tex) {
    if (!t.linear) return tex(ifloor(u), ifloor(v));
    float fu = u - 0.5f, fv = v - 0.5f;
    int u0 = ifloor(fu), v0 = ifloor(fv);
    float au = fu - u0, av = fv - v0;
    uint32_t c00 = tex(u0, v0), c10 = tex(u0 + 1, v0), c01 = tex(u0, v0 + 1), c11 = tex(u0 + 1, v0 + 1);
    uint32_t out = 0;
    for (int k = 0; k < 32; k += 8) {
        float a = (float)(c00 >> k & 255) * (1 - au) + (float)(c10 >> k & 255) * au;
        float b = (float)(c01 >> k & 255) * (1 - au) + (float)(c11 >> k & 255) * au;
        // std::lround for a value >= 0 (halves away from zero), without the library call
        out |= (uint32_t)((double)(a * (1 - av) + b * av) + 0.5) << k;
    }
    return out;
}

Tex tex_of(int ctxt) {
    Tex t{};
    uint64_t t0 = g_r[TEX0_1 + ctxt], cl = g_r[CLAMP_1 + ctxt], t1 = g_r[TEX1_1 + ctxt], ta = g_r[TEXA];
    t.tbp = (uint32_t)field(t0, 0, 14); t.tbw = (uint32_t)field(t0, 14, 6); t.psm = (uint32_t)field(t0, 20, 6);
    t.tw = (uint32_t)std::min<uint64_t>(10, field(t0, 26, 4)); t.th = (uint32_t)std::min<uint64_t>(10, field(t0, 30, 4));
    t.tcc = (uint32_t)field(t0, 34, 1); t.tfx = (uint32_t)field(t0, 35, 2);
    t.cbp = (uint32_t)field(t0, 37, 14); t.cpsm = (uint32_t)field(t0, 51, 4); t.csm = (uint32_t)field(t0, 55, 1);
    t.csa = (uint32_t)field(t0, 56, 5);
    t.wms = (uint32_t)field(cl, 0, 2); t.wmt = (uint32_t)field(cl, 2, 2);
    t.minu = (uint32_t)field(cl, 4, 10); t.maxu = (uint32_t)field(cl, 14, 10);
    t.minv = (uint32_t)field(cl, 24, 10); t.maxv = (uint32_t)field(cl, 34, 10);
    t.linear = field(t1, 5, 1) != 0;                 // MMAG
    t.ta0 = (uint32_t)field(ta, 0, 8); t.aem = (uint32_t)field(ta, 15, 1); t.ta1 = (uint32_t)field(ta, 32, 8);
    return t;
}

int clamp255(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

// What a pixel takes from its primitive.
struct Attr {
    double z;
    float r, g, b, a, f, u, v;                       // u, v in texels
};

// A triangle, set up once for both rasterisers. Coverage: pixel (x, y) is
// sampled at its integer coordinates; edge k runs from V[k+1] to V[k+2] and
// its function, in 12.4 fixed point, is exact in 64 bits. A pixel is in when
// every function is positive, or zero on a top or left edge.
struct Tri {
    const Vtx* V[3];
    int64_t area;                                    // twice the signed area, in 1/256 pixel
    double inv;                                      // 1 / |area|
    bool tl[3];                                      // edge k is a top or left edge
    bool gouraud, fst, tex;
    float tw, th;                                    // ST to texels
    int x0, x1, y0, y1;                              // pixels [x0, x1) x [y0, y1), scissored
};

int ceil16(int v) { return (v + 15) >> 4; }          // ceil(v / 16), v in 12.4

bool tri_setup(Tri& T, uint64_t prim, const Vtx& v0, const Vtx& v1, const Vtx& v2, const Tex* tp) {
    T.V[0] = &v0; T.V[1] = &v1; T.V[2] = &v2;
    T.area = (int64_t)(v1.X - v0.X) * (v2.Y - v0.Y) - (int64_t)(v2.X - v0.X) * (v1.Y - v0.Y);
    if (T.area == 0) return false;
    T.inv = 1.0 / (double)(T.area > 0 ? T.area : -T.area);
    for (int k = 0; k < 3; ++k) {
        const Vtx& p = *T.V[(k + 1) % 3];
        const Vtx& q = *T.V[(k + 2) % 3];
        int dy = q.Y - p.Y, dx = q.X - p.X;
        T.tl[k] = T.area > 0 ? (dy > 0 || (dy == 0 && dx < 0)) : (dy < 0 || (dy == 0 && dx > 0));
    }
    T.gouraud = field(prim, 3, 1);
    T.fst = field(prim, 8, 1);
    T.tex = tp != nullptr;
    T.tw = tp ? (float)(1 << tp->tw) : 1;
    T.th = tp ? (float)(1 << tp->th) : 1;
    uint64_t sc = g_r[SCISSOR_1 + field(prim, 9, 1)];
    T.x0 = std::max(ceil16(std::min({v0.X, v1.X, v2.X})), (int)field(sc, 0, 11));
    T.x1 = std::min(ceil16(std::max({v0.X, v1.X, v2.X})), (int)field(sc, 16, 11) + 1);
    T.y0 = std::max(ceil16(std::min({v0.Y, v1.Y, v2.Y})), (int)field(sc, 32, 11));
    T.y1 = std::min(ceil16(std::max({v0.Y, v1.Y, v2.Y})), (int)field(sc, 48, 11) + 1);
    return true;
}

int64_t edge(const Tri& T, int k, int x, int y) {
    const Vtx& p = *T.V[(k + 1) % 3];
    const Vtx& q = *T.V[(k + 2) % 3];
    int64_t e = (int64_t)(q.X - p.X) * (16 * (int64_t)y - p.Y) - (int64_t)(q.Y - p.Y) * (16 * (int64_t)x - p.X);
    return T.area > 0 ? e : -e;
}

int64_t edge_step(const Tri& T, int k) {             // edge(T, k, x + 1, y) - edge(T, k, x, y)
    int64_t d = -16 * (int64_t)(T.V[(k + 2) % 3]->Y - T.V[(k + 1) % 3]->Y);
    return T.area > 0 ? d : -d;
}

bool covered(const Tri& T, const int64_t e[3]) {
    for (int k = 0; k < 3; ++k)
        if (e[k] < 0 || (e[k] == 0 && !T.tl[k])) return false;
    return true;
}

void tri_attr(const Tri& T, const int64_t e[3], Attr& o) {
    float l0 = (float)((double)e[0] * T.inv), l1 = (float)((double)e[1] * T.inv), l2 = (float)((double)e[2] * T.inv);
    const Vtx &v0 = *T.V[0], &v1 = *T.V[1], &v2 = *T.V[2];
    if (T.gouraud) {
        o.r = l0 * v0.r + l1 * v1.r + l2 * v2.r;
        o.g = l0 * v0.g + l1 * v1.g + l2 * v2.g;
        o.b = l0 * v0.b + l1 * v1.b + l2 * v2.b;
        o.a = l0 * v0.a + l1 * v1.a + l2 * v2.a;
    } else {                                         // flat: the last vertex's colour
        o.r = v2.r; o.g = v2.g; o.b = v2.b; o.a = v2.a;
    }
    o.z = l0 * v0.z + l1 * v1.z + l2 * v2.z;
    o.f = l0 * v0.f + l1 * v1.f + l2 * v2.f;
    o.u = o.v = 0;
    if (T.tex) {
        if (T.fst) {
            o.u = l0 * v0.u + l1 * v1.u + l2 * v2.u;
            o.v = l0 * v0.v + l1 * v1.v + l2 * v2.v;
        } else {
            float sq = l0 * v0.s + l1 * v1.s + l2 * v2.s, tq = l0 * v0.t + l1 * v1.t + l2 * v2.t;
            float qq = l0 * v0.q + l1 * v1.q + l2 * v2.q;
            if (qq != 0) { o.u = sq / qq * T.tw; o.v = tq / qq * T.th; }
        }
    }
}

// A sprite: pixels [x0, x1) x [y0, y1) between its two vertices, the second
// vertex's colour, Z and fog, the texture coordinates interpolated.
struct Spr {
    const Vtx *a, *b;
    float ua, ub, va, vb, dx, dy;
    int x0, x1, y0, y1;
};

void spr_setup(Spr& S, uint64_t prim, const Vtx& a, const Vtx& b, const Tex* tp) {
    S.a = &a; S.b = &b;
    S.x0 = ceil16(std::min(a.X, b.X)); S.x1 = ceil16(std::max(a.X, b.X));
    S.y0 = ceil16(std::min(a.Y, b.Y)); S.y1 = ceil16(std::max(a.Y, b.Y));
    bool fst = field(prim, 8, 1);
    S.ua = fst ? a.u : a.s / a.q * (float)(1 << (tp ? tp->tw : 0));
    S.ub = fst ? b.u : b.s / b.q * (float)(1 << (tp ? tp->tw : 0));
    S.va = fst ? a.v : a.t / a.q * (float)(1 << (tp ? tp->th : 0));
    S.vb = fst ? b.v : b.t / b.q * (float)(1 << (tp ? tp->th : 0));
    S.dx = b.x - a.x;
    S.dy = b.y - a.y;
}

float spr_u(const Spr& S, int x) {                  // u depends on x only, v on y only
    float fu = S.dx != 0 ? ((float)x - S.a->x) / S.dx : 0;
    return S.ua + (S.ub - S.ua) * fu;
}

float spr_v(const Spr& S, int y) {
    float fv = S.dy != 0 ? ((float)y - S.a->y) / S.dy : 0;
    return S.va + (S.vb - S.va) * fv;
}

void spr_attr(const Spr& S, int x, int y, Attr& o) {
    const Vtx& b = *S.b;
    o = Attr{b.z, b.r, b.g, b.b, b.a, b.f, spr_u(S, x), spr_v(S, y)};
}

// ---- the reference: one pixel through the pipeline, the registers read here -----------------------
struct Pix {
    int ctxt;
    uint64_t prim;
    const Tex* tex;
};

void put(const Pix& P, int x, int y, const Attr& at) {
    int c = P.ctxt;
    uint64_t sc = g_r[SCISSOR_1 + c];
    if (x < (int)field(sc, 0, 11) || x > (int)field(sc, 16, 11) || y < (int)field(sc, 32, 11) || y > (int)field(sc, 48, 11)) return;
    uint64_t fr = g_r[FRAME_1 + c], zb = g_r[ZBUF_1 + c], te = g_r[TEST_1 + c];
    uint32_t fbp = (uint32_t)field(fr, 0, 9) * 32, fbw = (uint32_t)field(fr, 16, 6), fpsm = (uint32_t)field(fr, 24, 6);
    uint32_t fbmsk = (uint32_t)field(fr, 32, 32);
    int R = (int)at.r, G = (int)at.g, B = (int)at.b, A = (int)at.a;
    if (P.tex) {
        uint32_t tc = sample_with(*P.tex, at.u, at.v, [&](int u, int v) { return texel(*P.tex, u, v); });
        int tr = tc & 255, tg = tc >> 8 & 255, tb = tc >> 16 & 255, ta = tc >> 24 & 255;
        switch (P.tex->tfx) {
        case 0: R = (R * tr) >> 7; G = (G * tg) >> 7; B = (B * tb) >> 7; if (P.tex->tcc) A = (A * ta) >> 7; break;
        case 1: R = tr; G = tg; B = tb; if (P.tex->tcc) A = ta; break;
        case 2: R = ((R * tr) >> 7) + A; G = ((G * tg) >> 7) + A; B = ((B * tb) >> 7) + A; if (P.tex->tcc) A = A + ta; break;
        default: R = ((R * tr) >> 7) + A; G = ((G * tg) >> 7) + A; B = ((B * tb) >> 7) + A; if (P.tex->tcc) A = ta; break;
        }
        R = clamp255(R); G = clamp255(G); B = clamp255(B); A = clamp255(A);
    }
    if (field(P.prim, 5, 1)) {                       // FGE
        uint64_t fc = g_r[FOGCOL];
        int f = (int)at.f;
        R = (R * f + (int)field(fc, 0, 8) * (255 - f)) >> 8;
        G = (G * f + (int)field(fc, 8, 8) * (255 - f)) >> 8;
        B = (B * f + (int)field(fc, 16, 8) * (255 - f)) >> 8;
    }
    bool fb_write = true, z_write = !field(zb, 32, 1);
    bool rgb_only = false;
    if (field(te, 0, 1)) {                           // ATE
        int aref = (int)field(te, 4, 8);
        bool pass;
        switch (field(te, 1, 3)) {
        case 0: pass = false; break;
        case 1: pass = true; break;
        case 2: pass = A < aref; break;
        case 3: pass = A <= aref; break;
        case 4: pass = A == aref; break;
        case 5: pass = A >= aref; break;
        case 6: pass = A > aref; break;
        default: pass = A != aref; break;
        }
        if (!pass) {
            switch (field(te, 12, 2)) {
            case 0: return;                          // KEEP
            case 1: z_write = false; break;          // FB_ONLY
            case 2: fb_write = false; break;         // ZB_ONLY
            default: z_write = false; rgb_only = true; break;   // RGB_ONLY
            }
        }
    }
    uint32_t dst = pix_read(fpsm, fbp, fbw, x, y);
    uint32_t dst_rgba = fpsm == PSMCT16 || fpsm == PSMCT16S
        ? ((dst & 31) << 3 | (dst >> 5 & 31) << 11 | (dst >> 10 & 31) << 19 | ((dst & 0x8000) ? 0x80u << 24 : 0))
        : dst;
    if (field(te, 14, 1)) {                          // DATE
        bool bit = (fpsm == PSMCT16 || fpsm == PSMCT16S) ? (dst & 0x8000) != 0 : (dst & 0x80000000u) != 0;
        if (bit != (field(te, 15, 1) != 0)) return;
    }
    uint32_t zbp = (uint32_t)field(zb, 0, 9) * 32, zpsm = (uint32_t)field(zb, 24, 4) | 0x30;
    uint32_t z = 0;
    bool zte = field(te, 16, 1);
    if (zte) {
        double zmax = zpsm == PSMZ32 ? 4294967295.0 : zpsm == PSMZ24 ? 16777215.0 : 65535.0;
        z = (uint32_t)std::min(zmax, std::max(0.0, at.z));
        uint32_t zd = pix_read(zpsm, zbp, fbw, x, y);
        switch (field(te, 17, 2)) {
        case 0: return;
        case 1: break;
        case 2: if (z < zd) return; break;
        default: if (z <= zd) return; break;
        }
    } else {
        z_write = false;
    }
    if (field(P.prim, 6, 1)) {                       // ABE
        bool blend = !(g_r[PABE] & 1) || (A & 0x80);
        if (blend) {
            uint64_t al = g_r[ALPHA_1 + c];
            int cs[3] = {R, G, B}, cd[3] = {(int)(dst_rgba & 255), (int)(dst_rgba >> 8 & 255), (int)(dst_rgba >> 16 & 255)};
            int ad = (int)(dst_rgba >> 24);
            if (fpsm == PSMCT24) ad = 0x80;
            int cc = field(al, 4, 2) == 0 ? A : field(al, 4, 2) == 1 ? ad : (int)field(al, 32, 8);
            int out[3];
            for (int k = 0; k < 3; ++k) {
                auto sel = [&](uint64_t s) { return s == 0 ? cs[k] : s == 1 ? cd[k] : 0; };
                int va = sel(field(al, 0, 2)), vb = sel(field(al, 2, 2)), vd = sel(field(al, 6, 2));
                out[k] = (((va - vb) * cc) >> 7) + vd;
                out[k] = (g_r[COLCLAMP] & 1) ? clamp255(out[k]) : (out[k] & 255);
            }
            R = out[0]; G = out[1]; B = out[2];
        }
    }
    if (fb_write) {
        if (field(g_r[FBA_1 + c], 0, 1)) A |= 0x80;
        uint32_t keep = fbmsk;
        uint32_t col;
        if (fpsm == PSMCT16 || fpsm == PSMCT16S) {
            col = (uint32_t)(R >> 3) | (uint32_t)(G >> 3) << 5 | (uint32_t)(B >> 3) << 10 | (A & 0x80 ? 0x8000u : 0);
            keep = (fbmsk >> 3 & 31) | (fbmsk >> 11 & 31) << 5 | (fbmsk >> 19 & 31) << 10 | (fbmsk >> 31) << 15;
            if (rgb_only) keep |= 0x8000;
        } else {
            col = (uint32_t)R | (uint32_t)G << 8 | (uint32_t)B << 16 | (uint32_t)(A & 255) << 24;
            if (rgb_only) keep |= 0xFF000000u;
        }
        pix_write(fpsm, fbp, fbw, x, y, col, keep);
        ++g_draw_px;
    }
    if (z_write) pix_write(zpsm, zbp, fbw, x, y, z);
}

Pix pix_of(uint64_t prim, const Tex* tex) { return Pix{(int)field(prim, 9, 1), prim, tex}; }

void sprite_ref(uint64_t prim, const Vtx& a, const Vtx& b) {
    Tex t;
    const Tex* tp = nullptr;
    if (field(prim, 4, 1)) { t = tex_of((int)field(prim, 9, 1)); tp = &t; }
    Pix P = pix_of(prim, tp);
    Spr S;
    spr_setup(S, prim, a, b, tp);
    Attr at;
    for (int y = S.y0; y < S.y1; ++y)
        for (int x = S.x0; x < S.x1; ++x) {
            spr_attr(S, x, y, at);
            put(P, x, y, at);
        }
}

void triangle_ref(uint64_t prim, const Vtx& v0, const Vtx& v1, const Vtx& v2) {
    Tex t;
    const Tex* tp = nullptr;
    if (field(prim, 4, 1)) { t = tex_of((int)field(prim, 9, 1)); tp = &t; }
    Pix P = pix_of(prim, tp);
    Tri T;
    if (!tri_setup(T, prim, v0, v1, v2, tp)) return;
    Attr at;
    for (int y = T.y0; y < T.y1; ++y)
        for (int x = T.x0; x < T.x1; ++x) {
            int64_t e[3] = {edge(T, 0, x, y), edge(T, 1, x, y), edge(T, 2, x, y)};
            if (!covered(T, e)) continue;
            tri_attr(T, e, at);
            put(P, x, y, at);
        }
}

// ---- the fast rasteriser: the registers decoded once per primitive -----------------------------------
struct Draw {
    int sx0, sx1, sy0, sy1;                          // the scissor, inclusive
    Surf fb, zb;
    bool fb16;                                       // PSMCT16 or PSMCT16S
    uint32_t fbmsk, keep16;                          // FBMSK, and as a PSMCT16 mask
    bool zmsk;
    bool ate; int atst, aref, afail;
    bool date, datm;
    bool zte; int ztst; double zmax;
    bool abe, pabe, colclamp, fba;
    int sa, sb, sc, sd, fix;                         // ALPHA's A, B, C, D and FIX
    bool fge; int fog[3];
    const Tex* tex;
    Surf ts;                                         // the texture's buffer
    bool idx;                                        // indexed: through clut
    const uint32_t* clut;                            // the CLUT buffer as RGBA (clut_table)
};

const uint32_t* clut_table(const Tex& t, uint32_t n);

void draw_setup(Draw& D, uint64_t prim, const Tex* tp) {
    int c = (int)field(prim, 9, 1);
    uint64_t sc = g_r[SCISSOR_1 + c], fr = g_r[FRAME_1 + c], zb = g_r[ZBUF_1 + c], te = g_r[TEST_1 + c];
    D.sx0 = (int)field(sc, 0, 11); D.sx1 = (int)field(sc, 16, 11);
    D.sy0 = (int)field(sc, 32, 11); D.sy1 = (int)field(sc, 48, 11);
    uint32_t fbw = (uint32_t)field(fr, 16, 6), fpsm = (uint32_t)field(fr, 24, 6);
    D.fb = surf(fpsm, (uint32_t)field(fr, 0, 9) * 32, fbw);
    D.fb16 = fpsm == PSMCT16 || fpsm == PSMCT16S;
    D.fbmsk = (uint32_t)field(fr, 32, 32);
    D.keep16 = (D.fbmsk >> 3 & 31) | (D.fbmsk >> 11 & 31) << 5 | (D.fbmsk >> 19 & 31) << 10 | (D.fbmsk >> 31) << 15;
    uint32_t zpsm = (uint32_t)field(zb, 24, 4) | 0x30;
    D.zb = surf(zpsm, (uint32_t)field(zb, 0, 9) * 32, fbw);
    D.zmsk = field(zb, 32, 1);
    D.ate = field(te, 0, 1); D.atst = (int)field(te, 1, 3); D.aref = (int)field(te, 4, 8); D.afail = (int)field(te, 12, 2);
    D.date = field(te, 14, 1); D.datm = field(te, 15, 1);
    D.zte = field(te, 16, 1); D.ztst = (int)field(te, 17, 2);
    D.zmax = zpsm == PSMZ32 ? 4294967295.0 : zpsm == PSMZ24 ? 16777215.0 : 65535.0;
    uint64_t al = g_r[ALPHA_1 + c];
    D.abe = field(prim, 6, 1); D.pabe = g_r[PABE] & 1; D.colclamp = g_r[COLCLAMP] & 1;
    D.fba = field(g_r[FBA_1 + c], 0, 1);
    D.sa = (int)field(al, 0, 2); D.sb = (int)field(al, 2, 2); D.sc = (int)field(al, 4, 2); D.sd = (int)field(al, 6, 2);
    D.fix = (int)field(al, 32, 8);
    D.fge = field(prim, 5, 1);
    uint64_t fc = g_r[FOGCOL];
    D.fog[0] = (int)field(fc, 0, 8); D.fog[1] = (int)field(fc, 8, 8); D.fog[2] = (int)field(fc, 16, 8);
    D.tex = tp;
    if (!tp) return;
    D.ts = surf(tp->psm, tp->tbp, tp->tbw);
    if (!D.ts.l) once("a pixel format");
    D.idx = indexed(tp->psm);
    D.clut = D.idx ? clut_table(*tp, four_bit(tp->psm) ? 16u : 256u) : nullptr;
}

// TP, FP, ZP: the texture's, the frame's and the Z buffer's formats when
// they are among those the fast rasteriser is specialised for, 0 otherwise.
template <uint32_t TP> uint32_t texel_fast(const Draw& D, int u, int v) {
    const Tex& t = *D.tex;
    u = wrap(u, 1 << t.tw, t.wms, t.minu, t.maxu);
    v = wrap(v, 1 << t.th, t.wmt, t.minv, t.maxv);
    if constexpr (TP == PSMT8) return D.clut[D.ts.read<TP>(u, v)];
    else if constexpr (TP != 0) return rgba_from(D.ts.read<TP>(u, v), TP, t);
    uint32_t raw = D.ts.l ? D.ts.read(u, v) : 0;
    return D.idx ? D.clut[raw] : rgba_from(raw, t.psm, t);
}

// put() for a pixel inside the scissor; 1 when it writes the frame.
template <uint32_t TP, uint32_t FP, uint32_t ZP> int put_fast(const Draw& D, int x, int y, const Attr& at) {
    int R = (int)at.r, G = (int)at.g, B = (int)at.b, A = (int)at.a;
    if (D.tex) {
        uint32_t tc = sample_with(*D.tex, at.u, at.v, [&](int u, int v) { return texel_fast<TP>(D, u, v); });
        int tr = tc & 255, tg = tc >> 8 & 255, tb = tc >> 16 & 255, ta = tc >> 24 & 255;
        bool tcc = D.tex->tcc;
        switch (D.tex->tfx) {
        case 0: R = (R * tr) >> 7; G = (G * tg) >> 7; B = (B * tb) >> 7; if (tcc) A = (A * ta) >> 7; break;
        case 1: R = tr; G = tg; B = tb; if (tcc) A = ta; break;
        case 2: R = ((R * tr) >> 7) + A; G = ((G * tg) >> 7) + A; B = ((B * tb) >> 7) + A; if (tcc) A = A + ta; break;
        default: R = ((R * tr) >> 7) + A; G = ((G * tg) >> 7) + A; B = ((B * tb) >> 7) + A; if (tcc) A = ta; break;
        }
        R = clamp255(R); G = clamp255(G); B = clamp255(B); A = clamp255(A);
    }
    if (D.fge) {
        int f = (int)at.f;
        R = (R * f + D.fog[0] * (255 - f)) >> 8;
        G = (G * f + D.fog[1] * (255 - f)) >> 8;
        B = (B * f + D.fog[2] * (255 - f)) >> 8;
    }
    bool fb_write = true, z_write = !D.zmsk, rgb_only = false;
    if (D.ate) {
        bool pass;
        switch (D.atst) {
        case 0: pass = false; break;
        case 1: pass = true; break;
        case 2: pass = A < D.aref; break;
        case 3: pass = A <= D.aref; break;
        case 4: pass = A == D.aref; break;
        case 5: pass = A >= D.aref; break;
        case 6: pass = A > D.aref; break;
        default: pass = A != D.aref; break;
        }
        if (!pass) {
            switch (D.afail) {
            case 0: return 0;
            case 1: z_write = false; break;
            case 2: fb_write = false; break;
            default: z_write = false; rgb_only = true; break;
            }
        }
    }
    uint32_t dst = 0;
    bool have_dst = D.date || D.abe;
    if (have_dst) {
        if constexpr (FP != 0) dst = D.fb.read<FP>(x, y);
        else dst = D.fb.l ? D.fb.read(x, y) : 0;
    }
    if (D.date) {
        bool bit = D.fb16 ? (dst & 0x8000) != 0 : (dst & 0x80000000u) != 0;
        if (bit != D.datm) return 0;
    }
    uint32_t z = 0;
    if (D.zte) {
        z = (uint32_t)std::min(D.zmax, std::max(0.0, at.z));
        uint32_t zd;
        if constexpr (ZP != 0) zd = D.zb.read<ZP>(x, y);
        else zd = D.zb.l ? D.zb.read(x, y) : 0;
        switch (D.ztst) {
        case 0: return 0;
        case 1: break;
        case 2: if (z < zd) return 0; break;
        default: if (z <= zd) return 0; break;
        }
    } else {
        z_write = false;
    }
    if (D.abe && (!D.pabe || (A & 0x80))) {
        uint32_t d = D.fb16
            ? ((dst & 31) << 3 | (dst >> 5 & 31) << 11 | (dst >> 10 & 31) << 19 | ((dst & 0x8000) ? 0x80u << 24 : 0))
            : dst;
        int cs[3] = {R, G, B}, cd[3] = {(int)(d & 255), (int)(d >> 8 & 255), (int)(d >> 16 & 255)};
        int ad = D.fb.psm == PSMCT24 ? 0x80 : (int)(d >> 24);
        int cc = D.sc == 0 ? A : D.sc == 1 ? ad : D.fix;
        int out[3];
        for (int k = 0; k < 3; ++k) {
            int va = D.sa == 0 ? cs[k] : D.sa == 1 ? cd[k] : 0;
            int vb = D.sb == 0 ? cs[k] : D.sb == 1 ? cd[k] : 0;
            int vd = D.sd == 0 ? cs[k] : D.sd == 1 ? cd[k] : 0;
            out[k] = (((va - vb) * cc) >> 7) + vd;
            out[k] = D.colclamp ? clamp255(out[k]) : (out[k] & 255);
        }
        R = out[0]; G = out[1]; B = out[2];
    }
    if (fb_write && (FP != 0 || D.fb.l)) {
        if (D.fba) A |= 0x80;
        uint32_t col, keep;
        if (D.fb16) {
            col = (uint32_t)(R >> 3) | (uint32_t)(G >> 3) << 5 | (uint32_t)(B >> 3) << 10 | (A & 0x80 ? 0x8000u : 0);
            keep = D.keep16 | (rgb_only ? 0x8000u : 0);
        } else {
            col = (uint32_t)R | (uint32_t)G << 8 | (uint32_t)B << 16 | (uint32_t)(A & 255) << 24;
            keep = D.fbmsk | (rgb_only ? 0xFF000000u : 0);
        }
        D.fb.write<FP>(x, y, col, keep);
    }
    if (z_write && (ZP != 0 || D.zb.l)) D.zb.write<ZP>(x, y, z, 0);
    return fb_write;
}

// The rows [ya, yb) of a sprite or a triangle; the pixels written.
template <uint32_t TP, uint32_t FP, uint32_t ZP> uint64_t sprite_rows(const Draw& D, const Spr& S, int ya, int yb) {
    int x0 = std::max(S.x0, D.sx0), x1 = std::min(S.x1, D.sx1 + 1);
    if (x0 >= x1) return 0;
    float us[2048];                                  // 0 <= x0 < x1 <= 2048
    for (int x = x0; x < x1; ++x) us[x - x0] = spr_u(S, x);
    const Vtx& b = *S.b;
    Attr at{b.z, b.r, b.g, b.b, b.a, b.f, 0, 0};
    uint64_t n = 0;
    for (int y = ya; y < yb; ++y) {
        at.v = spr_v(S, y);
        for (int x = x0; x < x1; ++x) {
            at.u = us[x - x0];
            n += put_fast<TP, FP, ZP>(D, x, y, at);
        }
    }
    return n;
}

template <uint32_t TP, uint32_t FP, uint32_t ZP> uint64_t triangle_rows(const Draw& D, const Tri& T, int ya, int yb) {
    int64_t step[3] = {edge_step(T, 0), edge_step(T, 1), edge_step(T, 2)};
    Attr at;
    uint64_t n = 0;
    for (int y = ya; y < yb; ++y) {
        int64_t e[3] = {edge(T, 0, T.x0, y), edge(T, 1, T.x0, y), edge(T, 2, T.x0, y)};
        for (int x = T.x0; x < T.x1; ++x, e[0] += step[0], e[1] += step[1], e[2] += step[2]) {
            if (!covered(T, e)) continue;
            tri_attr(T, e, at);
            n += put_fast<TP, FP, ZP>(D, x, y, at);
        }
    }
    return n;
}

// The specialised loops, by format: texture (none or other, PSMCT32,
// PSMCT16, PSMT8), frame (other, PSMCT32, PSMCT16), Z buffer (other,
// PSMZ32, PSMZ24, PSMZ16).
template <class Fn> void dispatch(const Draw& D, const Fn& fn) {
    auto by_z = [&](auto tp, auto fp) {
        switch (D.zb.psm) {
        case PSMZ32: fn(tp, fp, std::integral_constant<uint32_t, PSMZ32>{}); break;
        case PSMZ24: fn(tp, fp, std::integral_constant<uint32_t, PSMZ24>{}); break;
        case PSMZ16: fn(tp, fp, std::integral_constant<uint32_t, PSMZ16>{}); break;
        default: fn(tp, fp, std::integral_constant<uint32_t, 0>{}); break;
        }
    };
    auto by_f = [&](auto tp) {
        switch (D.fb.psm) {
        case PSMCT32: by_z(tp, std::integral_constant<uint32_t, PSMCT32>{}); break;
        case PSMCT16: by_z(tp, std::integral_constant<uint32_t, PSMCT16>{}); break;
        default: by_z(tp, std::integral_constant<uint32_t, 0>{}); break;
        }
    };
    switch (D.tex ? D.tex->psm : 0xFFu) {
    case PSMCT32: by_f(std::integral_constant<uint32_t, PSMCT32>{}); break;
    case PSMCT16: by_f(std::integral_constant<uint32_t, PSMCT16>{}); break;
    case PSMT8: by_f(std::integral_constant<uint32_t, PSMT8>{}); break;
    default: by_f(std::integral_constant<uint32_t, 0>{}); break;
    }
}

// ---- the fast rasteriser on several threads ----------------------------------------------------------
// PS2KIT_GS_THREADS=n (default 4; 0 draws on the GS's own thread): worker
// threads draw the fast rasteriser's triangles and sprites. The GS's thread
// decodes everything as before and hands each primitive out as a job (its
// vertices, texture and decoded registers copied, a CLUT table from a pool);
// each worker takes every job in order and draws only the rows it owns, strips
// of 8 rows dealt in turn. A pixel's frame and Z are then always written by the
// same worker, in the GS's order, so memory ends as one thread leaves it.
// That holds while no job reads what another job writes elsewhere, and while
// the GS's thread itself does not touch what a job reads or writes; so the GS's
// thread keeps the spans of local memory pending jobs write (by target: a
// buffer's start, layout and width) and read (textures), whole pages, and
// waits for all jobs to finish (a barrier) before
//   a texture (or a CLUT load) that pending jobs may write;
//   a target that overlaps another pending target;
//   transfers into memory pending jobs use, local-to-local, local-to-host;
//   the reference rasteriser, points and lines; the PCRTC's picture,
//   recordings and replays reading memory (gs_vram).
// A primitive that would read its own target (its frame, or a Z buffer it
// writes: DW2's feedback copy reads, filtered, a few texels of the Z buffer
// it only tests against), whose frame and Z buffers overlap, or that reaches
// past its buffer's width (where rows alias) is drawn alone, as a sync job.
constexpr int STRIP = 8;                             // rows a worker owns in a row
constexpr size_t RING = 4096;                        // jobs handed out and not yet drawn by all
constexpr int CLUT_SLOTS = 64;
constexpr int MAX_WORKERS = 16;

// A job: a triangle or a sprite, each worker drawing its rows; or one the
// workers meet at (sync), the last to arrive doing it alone once all before
// it are done: a transfer the GS's thread deferred, a primitive drawn alone.
enum { J_TRI, J_SPRITE, J_XFER, J_ALONE };
struct Job {
    int kind;
    bool sprite;
    int x0, x1, y0, y1;                              // pixels, scissored
    Vtx v[3];
    Tex t;
    Draw D;
    Tri T;
    Spr S;
    Xfer xs;                                         // J_XFER: the transfer where these bytes begin
    std::vector<uint8_t> bytes;
    std::atomic<int> arrived{0};                     // sync jobs: workers there
    std::atomic<bool> done{false};                   // and the last one's work done
};

struct alignas(64) Worker {
    std::atomic<uint64_t> done{0};                   // jobs finished
    std::atomic<uint64_t> px{0};                     // pixels written
    std::atomic<uint64_t> busy_us{0}, sync_us{0};    // time with jobs; of it, waiting at sync jobs (heartbeat)
};

int g_nw;                                            // worker threads
Worker g_w[MAX_WORKERS];
std::unique_ptr<Job[]> g_jobs;                       // RING slots
Job g_job_here;                                      // a primitive drawn on the GS's thread
uint64_t g_next;                                     // jobs handed out (the GS's thread)
std::atomic<uint64_t> g_pub{0};                      // the same, for the workers
std::atomic<int> g_sleepers{0};
std::mutex g_w_mu;
std::condition_variable g_w_cv;
uint64_t g_barriers[W_N], g_alone, g_deferred, g_deferred_n;   // barriers that waited (by cause), primitives drawn alone, deferred transfers (bytes, jobs)

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

template <class Rows> void strips(int y0, int y1, int id, const Rows& rows) {
    int n = g_nw, s = y0 / STRIP;                    // y0 >= 0
    s += ((id - s % n) % n + n) % n;                 // the first strip from s that is id's
    for (; s * STRIP < y1; s += n) rows(std::max(y0, s * STRIP), std::min(y1, (s + 1) * STRIP));
}

uint64_t job_rows(const Job& j, int ya, int yb) {
    uint64_t px = 0;
    dispatch(j.D, [&](auto tp, auto fp, auto zp) {
        px = j.sprite ? sprite_rows<tp.value, fp.value, zp.value>(j.D, j.S, ya, yb)
                      : triangle_rows<tp.value, fp.value, zp.value>(j.D, j.T, ya, yb);
    });
    return px;
}

uint64_t job_strips(const Job& j, int id) {
    uint64_t px = 0;
    dispatch(j.D, [&](auto tp, auto fp, auto zp) {
        strips(j.y0, j.y1, id, [&](int ya, int yb) {
            px += j.sprite ? sprite_rows<tp.value, fp.value, zp.value>(j.D, j.S, ya, yb)
                           : triangle_rows<tp.value, fp.value, zp.value>(j.D, j.T, ya, yb);
        });
    });
    return px;
}

thread_local uint64_t t_sync_us;

// A sync job: every worker arrives; the last does it (everything before it is
// done by all), the others wait for that.
uint64_t job_sync(Job& j) {
    uint64_t px = 0;
    if (j.arrived.fetch_add(1, std::memory_order_acq_rel) == g_nw - 1) {
        if (j.kind == J_XFER) {
            Xfer X = j.xs;
            xfer_run<true>(X, j.bytes.data(), j.bytes.size());
        } else {
            px = job_rows(j, j.y0, j.y1);
        }
        j.done.store(true, std::memory_order_release);
    } else {
        auto t0 = std::chrono::steady_clock::now();
        for (int spin = 0; !j.done.load(std::memory_order_acquire); ++spin) {
            if (spin < 4000) cpu_relax();
            else std::this_thread::yield();
        }
        t_sync_us += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    return px;
}

void worker(int id) {
    t_worker = true;
    if (const char* f = std::getenv("PS2KIT_GS_WORKER_PROFILE"); f && id == 0) profile_this_thread(f);   // the first
    Worker& w = g_w[id];
    uint64_t k = 0, px = 0;
    for (;;) {
        uint64_t pub = g_pub.load(std::memory_order_acquire);
        for (int spin = 0; pub == k && spin < 4000; ++spin) {
            cpu_relax();
            pub = g_pub.load(std::memory_order_acquire);
        }
        if (pub == k) {
            std::unique_lock<std::mutex> lk(g_w_mu);
            g_sleepers.fetch_add(1);
            g_w_cv.wait(lk, [&] { return g_pub.load() != k; });
            g_sleepers.fetch_sub(1);
            continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        for (; k < pub; ++k) {
            Job& j = g_jobs[k % RING];
            px += j.kind == J_XFER || j.kind == J_ALONE ? job_sync(j) : job_strips(j, id);
            w.px.store(px, std::memory_order_relaxed);
            w.done.store(k + 1, std::memory_order_release);
        }
        w.busy_us.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count(),
                            std::memory_order_relaxed);
        w.sync_us.store(t_sync_us, std::memory_order_relaxed);
    }
}

bool all_done(uint64_t target) {
    for (int i = 0; i < g_nw; ++i)
        if (g_w[i].done.load(std::memory_order_acquire) < target) return false;
    return true;
}

// The GS's thread's time waiting for the drawing threads, by cause (a
// barrier's W_*; W_N: a free slot in the ring or in the CLUT tables).
double g_wait_s[W_N + 1];

void wait_done(uint64_t target, int why = W_N) {    // every job before `target` drawn
    if (all_done(target)) return;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < g_nw; ++i)
        for (int spin = 0; g_w[i].done.load(std::memory_order_acquire) < target; ++spin) {
            if (spin < 4000) cpu_relax();
            else std::this_thread::yield();
        }
    g_wait_s[why] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Pending spans of local memory: what jobs not yet drawn by every worker
// write (by target) and read (textures), each with the jobs to wait for
// before it is free: all of them up to the last that used it.
// A set of local memory's blocks (256 bytes each), with the bytes from its
// first block to the end of its last (finish): precise where a span's whole
// pages are not (and where they would run past the end of memory).
struct Blocks {
    uint64_t w[VRAM / 256 / 64];
    uint32_t lo = 0, hi = 0;
    void reset() { std::memset(w, 0, sizeof w); }
    void set() { std::memset(w, 0xFF, sizeof w); }
    void set(uint32_t b) { w[b >> 6] |= 1ull << (b & 63); }
    bool meets(const Blocks& o) const {
        for (size_t k = 0; k < std::size(w); ++k)
            if (w[k] & o.w[k]) return true;
        return false;
    }
    void finish() {
        lo = hi = 0;
        size_t n = std::size(w), a = 0, b = n;
        while (a < n && !w[a]) ++a;
        if (a == n) return;
        while (!w[b - 1]) --b;
        lo = (uint32_t)(a * 64 + __builtin_ctzll(w[a])) * 256;
        hi = (uint32_t)((b - 1) * 64 + 64 - __builtin_clzll(w[b - 1])) * 256;
    }
};
struct Span {
    uint32_t lo, hi;                                 // bytes [lo, hi)
    uint64_t key;                                    // a write's target: start, layout, width; SYNC
    uint64_t seq;                                    // jobs to be done before it is free
    std::shared_ptr<const Blocks> blocks;            // a texture's: its blocks exactly (else the pages)
};
// A sync job's span: later jobs come after it anyway; only the GS's thread's
// own reads and writes wait for it.
constexpr uint64_t SYNC = 1ull << 63;
std::vector<Span> g_pw, g_pr;                        // written, read
Span g_xspan;                                        // the host-to-local transfer's destination
uint64_t g_xmark;                                    // jobs handed out when it was last checked

bool overlap(const Span& a, const Span& b) { return a.lo < b.hi && b.lo < a.hi; }

// The bytes the pixels [x0, x1) x [y0, y1) of s can touch, as whole pages.
Span span_of(const Surf& s, int x0, int y0, int x1, int y1) {
    uint64_t key = (uint64_t)s.bp_bits << 24 | (uint64_t)base_of(s.psm) << 16 | s.ppr;
    if (x0 < 0 || y0 < 0 || x1 > 2048 || y1 > 2048) return Span{0, VRAM, key, 0};   // wraps
    uint32_t lpw = (uint32_t)s.l->lpw, lph = (uint32_t)s.l->lph;
    uint64_t p0 = (uint64_t)((uint32_t)y0 >> lph) * s.ppr + ((uint32_t)x0 >> lpw);
    uint64_t p1 = (uint64_t)((uint32_t)(y1 - 1) >> lph) * s.ppr + ((uint32_t)(x1 - 1) >> lpw) + 1;
    uint64_t lo = s.bp_bits / 8 + p0 * 8192, hi = s.bp_bits / 8 + p1 * 8192;
    if (hi > VRAM) return Span{0, VRAM, key, 0};
    return Span{(uint32_t)lo, (uint32_t)hi, key, 0};
}

uint64_t min_done() {
    uint64_t d = g_next;
    for (int i = 0; i < g_nw; ++i) d = std::min(d, g_w[i].done.load(std::memory_order_acquire));
    return d;
}

void wait_for(uint64_t seq, int why) {               // every job before `seq` drawn
    if (seq && !all_done(seq)) {
        wait_done(seq, why);
        ++g_barriers[why];
    }
}

// Deferred transfers (defer_xfer) with the blocks (256 bytes) they write
// exactly: the GS's thread's own reads (a CLUT) and the transfers it writes
// itself check them, finer than a span's whole pages (a CLUT often shares a
// page with a texture sent just before it).
struct PendingXfer {
    uint64_t key, seq;                               // the transfer (g_xspan.key); its last job
    Span hull;
    std::shared_ptr<Blocks> blocks;
};
std::vector<PendingXfer> g_px;
Blocks g_xblocks;                                    // the transfer in progress's blocks

// The blocks the pixels [x0, x1) x [y0, y1) of s are in: a block is a cell of
// the layout's block size (a page holds 32), every pixel of a cell in it.
void blocks_of(const Surf& s, int x0, int y0, int x1, int y1, Blocks& b) {
    b.reset();
    static const int BW[4] = {8, 16, 16, 32}, BH[4] = {8, 8, 16, 16};
    int base = base_of(s.psm), bw = BW[base], bh = BH[base];
    if (x0 < 0 || y0 < 0 || x1 > 2048 || y1 > 2048) { b.set(); b.finish(); return; }
    for (int y = y0 / bh * bh; y < y1; y += bh)
        for (int x = x0 / bw * bw; x < x1; x += bw) b.set(s.at(std::max(x, x0), std::max(y, y0)) / 8 / 256);
    b.finish();
}

// A texture's blocks (the texels [u0, u1) x [v0, v1) a primitive can
// read), cached by texture and rectangle. The key is exact: start (25
// bits), format (6), pages a row (6); the rectangle (12 bits a side).
struct TexKey {
    uint64_t buf, rect;
    bool operator==(const TexKey& o) const { return buf == o.buf && rect == o.rect; }
};
struct TexKeyHash {
    size_t operator()(const TexKey& k) const { return (size_t)(k.buf * 0x9E3779B97F4A7C15ull ^ k.rect); }
};
TexKey texture_key(const Surf& ts, int u0, int v0, int u1, int v1) {
    return TexKey{(uint64_t)ts.bp_bits | (uint64_t)(ts.psm & 63) << 25 | (uint64_t)(ts.ppr & 63) << 31,
                  (uint64_t)(u0 & 4095) | (uint64_t)(v0 & 4095) << 12 | (uint64_t)(u1 & 4095) << 24 |
                      (uint64_t)(v1 & 4095) << 36};
}

std::shared_ptr<const Blocks> texture_blocks(const Surf& ts, const TexKey& key, int u0, int v0, int u1, int v1) {
    static std::unordered_map<TexKey, std::shared_ptr<const Blocks>, TexKeyHash> cache;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    if (cache.size() > 1024) cache.clear();
    auto b = std::make_shared<Blocks>();
    blocks_of(ts, u0, v0, u1, v1, *b);
    cache[key] = b;
    return b;
}

// ---- the drawing threads' texture copies --------------------------------------------------------------
// A job reads its texture from a copy the GS's thread makes when it hands the
// job out (a slab: the bytes from the texture's first block to the end of its
// last), not from local memory. A transfer into a texture that pending jobs
// still read then needs no wait: DW2's battle loads some 70 textures a field
// into places textures drawn just before were read from, and each such
// transfer, made a sync job, stopped every drawing thread until the slowest
// had caught up (half their time with eight threads). A slab is copied again
// only after something was written into its pages: the GS's thread counts
// writes by page (dirty: transfers, the targets of the jobs it hands out, and
// all of memory for what it draws itself). A slab still read by pending jobs
// is not written again: it is put aside until they are drawn, and a new one
// taken. Textures that reach past the end of memory, larger than MAX_SLAB, or
// that a primitive reads where it draws (drawn alone) are still read in
// place, as before.
constexpr uint32_t PAGE = 8192, MAX_SLAB = 1u << 20;
uint64_t g_wr_n = 1, g_wr_all = 1, g_wr_page[VRAM / PAGE];   // writes so far; the last to all memory, to each page
struct Slab {
    std::vector<uint8_t> data;
    uint32_t lo = 0, hi = 0;                         // local memory's bytes [lo, hi)
    uint64_t at = 0;                                 // g_wr_n when copied
    uint64_t last = 0;                               // the last job reading it (handed out as job last - 1)
};
std::unordered_map<TexKey, std::unique_ptr<Slab>, TexKeyHash> g_slabs;   // by texture_key
std::vector<std::unique_ptr<Slab>> g_slabs_aside;    // replaced while pending jobs read them
uint64_t g_slab_copies, g_slab_bytes;                // (heartbeat)

void dirty(uint32_t lo, uint32_t hi) {
    if (lo >= hi) return;
    ++g_wr_n;
    for (uint32_t p = lo / PAGE; p < (hi + PAGE - 1) / PAGE && p < VRAM / PAGE; ++p) g_wr_page[p] = g_wr_n;
}

void dirty_all() { g_wr_all = ++g_wr_n; }

uint64_t min_done();

// Where a job reads its texture from (Surf::mem): the slab's bytes placed at
// local memory's addresses; null when the texture is read in place.
const uint8_t* snap_texture(const TexKey& key, const Blocks& b, uint64_t seq) {
    if (b.hi <= b.lo || b.hi - b.lo > MAX_SLAB) return nullptr;
    std::unique_ptr<Slab>& sp = g_slabs[key];
    if (!sp) sp = std::make_unique<Slab>();
    Slab* s = sp.get();
    bool fresh = s->hi > s->lo && s->at >= g_wr_all;
    for (uint32_t p = b.lo / PAGE; fresh && p < (b.hi + PAGE - 1) / PAGE; ++p) fresh = g_wr_page[p] <= s->at;
    if (!fresh) {
        uint64_t done = min_done();
        if (s->last > done) {                        // still read: aside, and a new one
            g_slabs_aside.push_back(std::move(sp));
            sp = std::make_unique<Slab>();
            s = sp.get();
            if (g_slabs_aside.size() > 64)
                g_slabs_aside.erase(std::remove_if(g_slabs_aside.begin(), g_slabs_aside.end(),
                                                   [done](const std::unique_ptr<Slab>& a) { return a->last <= done; }),
                                    g_slabs_aside.end());
        }
        s->lo = b.lo;
        s->hi = b.hi;
        s->data.assign(g_vram.begin() + b.lo, g_vram.begin() + b.hi);
        s->at = g_wr_n;
        ++g_slab_copies;
        g_slab_bytes += b.hi - b.lo;
    }
    s->last = seq;
    if (g_slabs.size() > 4096) {                     // textures not used for a while go
        uint64_t done = min_done();
        for (auto it = g_slabs.begin(); it != g_slabs.end();)
            it = it->second->last <= done && it->second.get() != s ? g_slabs.erase(it) : std::next(it);
    }
    return (const uint8_t*)((uintptr_t)s->data.data() - s->lo);
}

// The last job reading a texture whose blocks meet b (0: none).
uint64_t hit_reads(const Span& hull, const Blocks& b) {
    uint64_t n = 0;
    for (const Span& r : g_pr)
        if (overlap(r, hull) && (!r.blocks || r.blocks->meets(b))) n = std::max(n, r.seq);
    return n;
}

// The last job of a deferred transfer whose blocks meet b (0: none).
uint64_t hit_xfers(const Span& hull, const Blocks& b, uint64_t except = 0) {
    uint64_t n = 0;
    for (const PendingXfer& x : g_px)
        if (x.key != except && overlap(x.hull, hull) && x.blocks->meets(b)) n = std::max(n, x.seq);
    return n;
}

void barrier(int why) {
    if (!g_nw) return;
    wait_for(g_next, why);
    g_pw.clear();
    g_pr.clear();
    g_px.clear();
}

void forget_done() {                                 // spans whose jobs are all drawn go
    if (g_pw.empty() && g_pr.empty() && g_px.empty()) return;
    uint64_t d = min_done();
    if (d >= g_next) { g_pw.clear(); g_pr.clear(); g_px.clear(); return; }
    g_px.erase(std::remove_if(g_px.begin(), g_px.end(), [d](const PendingXfer& x) { return x.seq <= d; }), g_px.end());
    if (g_pw.size() + g_pr.size() < 32) return;
    auto gone = [d](const Span& s) { return s.seq <= d; };
    g_pw.erase(std::remove_if(g_pw.begin(), g_pw.end(), gone), g_pw.end());
    g_pr.erase(std::remove_if(g_pr.begin(), g_pr.end(), gone), g_pr.end());
}

// The jobs to wait for before s is touched: 0 when no span of v overlaps it
// (jobs: those of the other jobs, not sync).
uint64_t hit(const std::vector<Span>& v, const Span& s, bool jobs = false) {
    uint64_t n = 0;
    for (const Span& p : v)
        if (overlap(p, s) && !(jobs && (p.key & SYNC))) n = std::max(n, p.seq);
    return n;
}

uint64_t hit_other(const std::vector<Span>& v, const Span& s) {   // another target's, of the other jobs
    uint64_t n = 0;
    for (const Span& p : v)
        if (p.key != s.key && !(p.key & SYNC) && overlap(p, s)) n = std::max(n, p.seq);
    return n;
}

void add_write(const Span& s) {
    for (Span& p : g_pw)
        if (p.key == s.key) {
            p.lo = std::min(p.lo, s.lo);
            p.hi = std::max(p.hi, s.hi);
            p.seq = s.seq;
            return;
        }
    g_pw.push_back(s);
}

void add_read(const Span& s) {
    for (size_t k = g_pr.size(); k-- > 0;)
        if (g_pr[k].blocks == s.blocks && g_pr[k].lo <= s.lo && s.hi <= g_pr[k].hi) { g_pr[k].seq = s.seq; return; }
    g_pr.push_back(s);
}

// The GS's thread is about to write (or read) a rectangle of local memory.
void hazard_rect(uint32_t psm, uint32_t bp, uint32_t bw, int x0, int y0, int x1, int y1, bool write) {
    if (!g_nw || x0 >= x1 || y0 >= y1) return;
    Surf s = surf(psm, bp, bw);
    if (!s.l) { barrier(W_CLUT); return; }
    forget_done();
    Blocks b;
    blocks_of(s, x0, y0, x1, y1, b);
    Span sp{b.lo, b.hi, 0, 0};
    uint64_t wait = std::max({hit(g_pw, sp), write ? hit_reads(sp, b) : 0, hit_xfers(sp, b)});
    wait_for(wait, W_CLUT);
}

void hazard_clut(uint32_t psm, uint32_t cbp, uint32_t n) {
    hazard_rect(psm, cbp, 1, 0, 0, n == 16 ? 8 : 16, n == 16 ? 2 : 16, false);
}

// The host-to-local transfer's destination, checked as it starts and again
// whenever jobs were handed out since: true when pending jobs use it, and
// the transfer is to be deferred.
bool hazard_xfer() {
    if (!g_nw) return false;
    forget_done();
    g_xmark = g_next;
    return hit(g_pw, g_xspan) || hit_reads(g_xspan, g_xblocks) || hit_xfers(g_xspan, g_xblocks, g_xspan.key);
}

void dirty_xfer() {
    if (g_nw) dirty(g_xspan.lo, g_xspan.hi);
}

bool hazard_xfer_start(uint32_t psm, uint32_t bp, uint32_t bw, int x0, int y0, int w, int h) {
    if (!g_nw) return false;
    Surf s = surf(psm, bp, bw);
    bool known = s.l && w > 0 && h > 0;
    if (known) blocks_of(s, x0, y0, x0 + w, y0 + h, g_xblocks);
    else { g_xblocks.set(); g_xblocks.finish(); }
    g_xspan = Span{g_xblocks.lo, g_xblocks.hi, 0, 0};
    static uint64_t transfers;
    g_xspan.key = SYNC | ++transfers;                // this transfer's
    return hazard_xfer();
}

void publish();

// Transfer data the drawing threads write, as a sync job: where the transfer
// is, and the bytes.
void defer_xfer(const uint8_t* q, size_t n) {
    if (g_next >= RING) wait_done(g_next - RING + 1);   // the slot is free
    Job& j = g_jobs[g_next % RING];
    j.kind = J_XFER;
    j.xs = g_x;
    j.bytes.assign(q, q + n);
    j.arrived.store(0, std::memory_order_relaxed);
    j.done.store(false, std::memory_order_relaxed);
    PendingXfer* px = nullptr;
    for (PendingXfer& x : g_px)
        if (x.key == g_xspan.key) px = &x;
    if (!px) {
        g_px.push_back(PendingXfer{g_xspan.key, 0, g_xspan, std::make_shared<Blocks>(g_xblocks)});
        px = &g_px.back();
    }
    px->seq = g_next + 1;
    g_deferred += n;
    ++g_deferred_n;
    publish();
}

// The CLUT buffer as RGBA for a texture, converted again only when the
// buffer, the CLUT's format, its offset or TEXA changed; each conversion in a
// slot of its own, which pending jobs may still read.
struct ClutCache {
    uint64_t gen = ~0ull;
    uint32_t cpsm, csa, n, ta0, ta1, aem;
} g_cc;
uint32_t g_clut_tab[CLUT_SLOTS][256];
int g_clut_slot;
uint64_t g_clut_busy[CLUT_SLOTS];                    // jobs to be done before the slot is written again

const uint32_t* clut_table(const Tex& t, uint32_t n) {
    ClutCache& c = g_cc;
    if (c.gen != g_clut_gen || c.cpsm != t.cpsm || c.csa != t.csa || c.n != n || c.ta0 != t.ta0 || c.ta1 != t.ta1 ||
        c.aem != t.aem) {
        g_clut_slot = (g_clut_slot + 1) % CLUT_SLOTS;
        wait_done(g_clut_busy[g_clut_slot]);
        for (uint32_t k = 0; k < n; ++k) g_clut_tab[g_clut_slot][k] = clut_entry(t, k);
        c.gen = g_clut_gen; c.cpsm = t.cpsm; c.csa = t.csa; c.n = n;
        c.ta0 = t.ta0; c.ta1 = t.ta1; c.aem = t.aem;
    }
    return g_clut_tab[g_clut_slot];
}

// A triangle (v[0..2]) or a sprite (v[0..1]) for the fast rasteriser.
void fast_prim(bool sprite, uint64_t prim, const Vtx* v) {
    bool threaded = g_nw > 0;
    if (threaded && g_next >= RING) wait_done(g_next - RING + 1);   // the slot is free
    Job& j = threaded ? g_jobs[g_next % RING] : g_job_here;
    j.kind = sprite ? J_SPRITE : J_TRI;
    j.sprite = sprite;
    for (int k = 0; k < (sprite ? 2 : 3); ++k) j.v[k] = v[k];
    const Tex* tp = nullptr;
    if (field(prim, 4, 1)) { j.t = tex_of((int)field(prim, 9, 1)); tp = &j.t; }
    if (sprite) spr_setup(j.S, prim, j.v[0], j.v[1], tp);
    else if (!tri_setup(j.T, prim, j.v[0], j.v[1], j.v[2], tp)) return;
    draw_setup(j.D, prim, tp);
    if (sprite) {
        j.x0 = std::max(j.S.x0, j.D.sx0); j.x1 = std::min(j.S.x1, j.D.sx1 + 1);
        j.y0 = std::max(j.S.y0, j.D.sy0); j.y1 = std::min(j.S.y1, j.D.sy1 + 1);
    } else {
        j.x0 = j.T.x0; j.x1 = j.T.x1; j.y0 = j.T.y0; j.y1 = j.T.y1;
    }
    if (j.x0 >= j.x1 || j.y0 >= j.y1) return;
    if (!threaded) { g_draw_px += job_rows(j, j.y0, j.y1); return; }

    const Draw& D = j.D;
    bool hf = D.fb.l != nullptr, hz = D.zte && D.zb.l != nullptr, ht = tp && D.ts.l != nullptr;
    Span sf{}, sz{}, st{};
    if (hf) sf = span_of(D.fb, j.x0, j.y0, j.x1, j.y1);
    if (hz) sz = span_of(D.zb, j.x0, j.y0, j.x1, j.y1);
    TexKey tkey{};
    if (ht) {
        int u0 = 0, v0 = 0, u1 = 1 << tp->tw, v1 = 1 << tp->th;
        if (tp->wms >= 2) u1 = std::max(u1, 1024);   // region modes reach MINU/MAXU
        if (tp->wmt >= 2) v1 = std::max(v1, 1024);
        if (sprite && tp->wms < 2 && tp->wmt < 2) {
            // A sprite reads the texels between its corners' coordinates
            // (filtered, from half a texel before; and the next ones): the
            // rectangle, a texel wider for rounding, clamped (CLAMP), or
            // taken only when no coordinate wraps (REPEAT). DW2's
            // full-screen copies of a 640-wide picture as a 1024-wide
            // texture would otherwise reach their own Z buffer and be drawn
            // alone, every drawing thread waiting.
            auto range = [&](float a, float b, int size, uint32_t mode, int& lo, int& hi) {
                float m = std::min(a, b), x = std::max(a, b);
                if (!(m > -4096.0f && x < 4096.0f)) return;   // (not a number, or far out)
                int l = tp->linear ? ifloor(m - 0.5f) : ifloor(m), h = (tp->linear ? ifloor(x - 0.5f) + 1 : ifloor(x)) + 2;
                if (mode == 1) { l = std::max(l, 0); h = std::min(h, size); }
                if (l >= 0 && h <= size && l < h) { lo = l; hi = h; }
            };
            range(j.S.ua, j.S.ub, u1, tp->wms, u0, u1);
            range(j.S.va, j.S.vb, v1, tp->wmt, v0, v1);
        }
        tkey = texture_key(D.ts, u0, v0, u1, v1);
        st.blocks = texture_blocks(D.ts, tkey, u0, v0, u1, v1);
        st.lo = st.blocks->lo;
        st.hi = st.blocks->hi;
    }
    bool alone = (hf && j.x1 > (int)(D.fb.ppr << D.fb.l->lpw)) || (hz && j.x1 > (int)(D.zb.ppr << D.zb.l->lpw)) ||
                 (hf && hz && sf.key != sz.key && overlap(sf, sz)) ||
                 (ht && ((hf && overlap(st, sf)) || (hz && !D.zmsk && overlap(st, sz))));
    forget_done();
    uint64_t seq = g_next + 1;
    if (alone) {                                     // the workers meet; one draws it
        ++g_alone;
        j.kind = J_ALONE;
        j.arrived.store(0, std::memory_order_relaxed);
        j.done.store(false, std::memory_order_relaxed);
        if (hf) { sf.key |= SYNC; sf.seq = seq; add_write(sf); }
        if (hz) { sz.key |= SYNC; sz.seq = seq; add_write(sz); }
        if (ht) { st.key = SYNC; st.seq = seq; add_read(st); }
    } else {
        // the texture after jobs that write it (copied now: after every
        // one, and after deferred transfers into it; read in place: after
        // those that are not sync jobs); a target after jobs that write
        // another target there, or read it as a texture
        uint64_t wait = 0;
        const uint8_t* slab = nullptr;
        if (ht) {
            if (hit(g_pw, st) || hit_xfers(st, *st.blocks)) {
                wait_for(std::max(hit(g_pw, st), hit_xfers(st, *st.blocks)), W_TEXTURE);
                forget_done();
            }
            slab = snap_texture(tkey, *st.blocks, seq);
            if (!slab) wait_for(hit(g_pw, st, true), W_TEXTURE);
        }
        if (hf) wait = std::max({wait, hit_other(g_pw, sf), hit(g_pr, sf, true)});
        if (hz) wait = std::max({wait, hit_other(g_pw, sz), hit(g_pr, sz, true)});
        wait_for(wait, W_TARGET);
        if (hf) { sf.seq = seq; add_write(sf); }
        if (hz) { sz.seq = seq; add_write(sz); }
        if (slab) j.D.ts.mem = slab;
        else if (ht) { st.seq = seq; add_read(st); }
    }
    if (hf) dirty(sf.lo, sf.hi);
    if (hz) dirty(sz.lo, sz.hi);
    if (D.idx) g_clut_busy[g_clut_slot] = seq;
    publish();
}

// The job at g_next handed to the workers.
void publish() {
    ++g_next;
    g_pub.store(g_next, std::memory_order_seq_cst);
    if (g_sleepers.load(std::memory_order_seq_cst)) {
        { std::lock_guard<std::mutex> lk(g_w_mu); }
        g_w_cv.notify_all();
    }
}

void start_workers() {
    const char* e = std::getenv("PS2KIT_GS_THREADS");
    int n = e && *e ? std::atoi(e) : 4;
    n = std::max(0, std::min(MAX_WORKERS, n));
    if (n <= 1) return;                              // one worker: no gain over the GS's thread
    g_jobs.reset(new Job[RING]);
    g_nw = n;
    for (int i = 0; i < n; ++i) std::thread(worker, i).detach();
}

uint64_t pixels_drawn() {
    uint64_t n = g_draw_px;
    for (int i = 0; i < g_nw; ++i) n += g_w[i].px.load(std::memory_order_relaxed);
    return n;
}

void draw_sprite(uint64_t prim, const Vtx& a, const Vtx& b) {
    if (g_ref) { barrier(); dirty_all(); sprite_ref(prim, a, b); return; }
    Vtx v[2] = {a, b};
    fast_prim(true, prim, v);
}

void draw_triangle(uint64_t prim, const Vtx& v0, const Vtx& v1, const Vtx& v2) {
    if (g_ref) { barrier(); dirty_all(); triangle_ref(prim, v0, v1, v2); return; }
    Vtx v[3] = {v0, v1, v2};
    fast_prim(false, prim, v);
}

void draw_line(uint64_t prim, const Vtx& a, const Vtx& b) {
    barrier();
    dirty_all();
    Pix P = pix_of(prim, nullptr);
    if (field(prim, 4, 1)) once("textured lines");
    float dx = b.x - a.x, dy = b.y - a.y;
    int n = (int)std::max(std::fabs(dx), std::fabs(dy));
    for (int k = 0; k <= n; ++k) {
        float f = n ? (float)k / (float)n : 0;
        Attr at{a.z + (b.z - a.z) * f, a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f,
                a.a + (b.a - a.a) * f, b.f, 0, 0};
        put(P, (int)std::floor(a.x + dx * f + 0.5f), (int)std::floor(a.y + dy * f + 0.5f), at);
    }
}

uint64_t attrs() {
    uint64_t p = g_r[PRIM];
    if (!(g_r[PRMODECONT] & 1)) p = (p & 7) | (g_r[PRMODE] & ~7ull);
    return p;
}

void vertex(uint64_t xyz, bool fog_given, bool kick) {
    uint64_t prim = attrs();
    int c = (int)field(prim, 9, 1);
    uint64_t off = g_r[XYOFFSET_1 + c];
    Vtx v{};
    v.X = (int)field(xyz, 0, 16) - (int)field(off, 0, 16);
    v.Y = (int)field(xyz, 16, 16) - (int)field(off, 32, 16);
    v.x = (float)v.X / 16.0f;
    v.y = (float)v.Y / 16.0f;
    v.z = fog_given ? (double)field(xyz, 32, 24) : (double)field(xyz, 32, 32);
    v.f = fog_given ? (float)field(xyz, 56, 8) : (float)field(g_r[FOG], 56, 8);
    uint64_t c0 = g_r[RGBAQ];
    v.r = (float)field(c0, 0, 8); v.g = (float)field(c0, 8, 8); v.b = (float)field(c0, 16, 8); v.a = (float)field(c0, 24, 8);
    uint32_t qb = (uint32_t)field(c0, 32, 32);
    std::memcpy(&v.q, &qb, 4);
    uint32_t sb = (uint32_t)field(g_r[ST], 0, 32), tb = (uint32_t)field(g_r[ST], 32, 32);
    std::memcpy(&v.s, &sb, 4);
    std::memcpy(&v.t, &tb, 4);
    v.u = (float)field(g_r[UV], 0, 14) / 16.0f;
    v.v = (float)field(g_r[UV], 16, 14) / 16.0f;
    uint32_t type = (uint32_t)field(g_r[PRIM], 0, 3);
    static const int need[8] = {1, 2, 2, 3, 3, 3, 2, 1};
    if (g_nv >= 3) g_nv = 2;
    g_v[g_nv++] = v;
    if (g_nv < need[type]) return;
    if (kick) {
        switch (type) {
        case 0:
            barrier();
            dirty_all();
            put(pix_of(prim, nullptr), (int)v.x, (int)v.y, Attr{v.z, v.r, v.g, v.b, v.a, v.f, 0, 0});
            break;
        case 1: case 2: draw_line(prim, g_v[0], g_v[1]); break;
        case 3: case 4: case 5: draw_triangle(prim, g_v[0], g_v[1], g_v[2]); break;
        case 6: draw_sprite(prim, g_v[0], g_v[1]); break;
        default: once("primitive type 7"); break;
        }
    }
    switch (type) {
    case 0: case 1: case 3: case 6: g_nv = 0; break;
    case 2: g_v[0] = g_v[1]; g_nv = 1; break;                // line strip
    case 4: g_v[0] = g_v[1]; g_v[1] = g_v[2]; g_nv = 2; break;   // triangle strip
    case 5: g_v[1] = g_v[2]; g_nv = 2; break;                // triangle fan
    }
}

// ---- register writes -----------------------------------------------------------------------------
uint64_t g_csr_events;                               // SIGNAL (1), FINISH (2) seen
uint64_t g_siglblid;

void reg_write(uint32_t r, uint64_t v) {
    r &= 0xFF;
    switch (r) {
    case PRIM: g_r[PRIM] = v; g_nv = 0; return;
    case XYZF2: g_r[r] = v; vertex(v, true, true); return;
    case XYZ2: g_r[r] = v; vertex(v, false, true); return;
    case XYZF3: g_r[r] = v; vertex(v, true, false); return;
    case XYZ3: g_r[r] = v; vertex(v, false, false); return;
    case TRXDIR: g_r[r] = v; xfer_start(); return;
    case HWREG: { uint8_t b[8]; std::memcpy(b, &v, 8); xfer_data(b, 8); return; }
    case SIGNAL: {
        uint32_t mask = (uint32_t)(v >> 32), id = (uint32_t)v;
        g_siglblid = (g_siglblid & ~(uint64_t)mask) | (id & mask);
        g_csr_events |= 1;
        return;
    }
    case FINISH: g_csr_events |= 2; return;
    case LABEL: {
        uint32_t mask = (uint32_t)(v >> 32), id = (uint32_t)v;
        g_siglblid = (g_siglblid & ~((uint64_t)mask << 32)) | (uint64_t)(id & mask) << 32;
        return;
    }
    case TEX0_1: case TEX0_2: case TEX2_1: case TEX2_2:
        if (r == TEX2_1 || r == TEX2_2) {            // TEX2 sets TEX0's PSM and CLUT fields
            uint64_t m = 0x3Full << 20 | 0x7FFFFFFFull << 37;
            int t0 = r == TEX2_1 ? TEX0_1 : TEX0_2;
            g_r[t0] = (g_r[t0] & ~m) | (v & m);
            clut_load(g_r[t0]);
        } else {
            g_r[r] = v;
            clut_load(v);
        }
        return;
    default:
        g_r[r] = v;
        return;
    }
}

// ---- the GIF -------------------------------------------------------------------------------------
struct Gif {
    bool in_packet = false;
    uint32_t nloop = 0, flg = 0, nreg = 0, reg = 0;
    bool eop = false;
    uint64_t regs = 0;
    bool half = false;                               // REGLIST: the upper doubleword of a quadword
} g_gif[4];                                          // by path (1-3)

uint64_t g_gif_qw;                                   // quadwords processed (heartbeat)

void packed(uint32_t desc, const uint64_t q[2]) {
    switch (desc) {
    case 0x0: reg_write(PRIM, q[0] & 0x7FF); break;
    case 0x1: {
        uint32_t qb;
        std::memcpy(&qb, &g_q, 4);
        reg_write(RGBAQ, (q[0] & 0xFF) | (q[0] >> 32 & 0xFF) << 8 | (q[1] & 0xFF) << 16 | (q[1] >> 32 & 0xFF) << 24 |
                             (uint64_t)qb << 32);
        break;
    }
    case 0x2: {
        uint32_t qb = (uint32_t)q[1];
        std::memcpy(&g_q, &qb, 4);
        reg_write(ST, q[0]);
        break;
    }
    case 0x3: reg_write(UV, (q[0] & 0x3FFF) | (q[0] >> 32 & 0x3FFF) << 16); break;
    case 0x4: {
        uint64_t v = (q[0] & 0xFFFF) | (q[0] >> 32 & 0xFFFF) << 16 | (q[1] >> 4 & 0xFFFFFF) << 32 | (q[1] >> 36 & 0xFF) << 56;
        reg_write((q[1] >> 47 & 1) ? XYZF3 : XYZF2, v);
        break;
    }
    case 0x5: {
        uint64_t v = (q[0] & 0xFFFF) | (q[0] >> 32 & 0xFFFF) << 16 | (q[1] & 0xFFFFFFFF) << 32;
        reg_write((q[1] >> 47 & 1) ? XYZ3 : XYZ2, v);
        break;
    }
    case 0xA: reg_write(FOG, (q[1] >> 36 & 0xFF) << 56); break;
    case 0xE: reg_write((uint32_t)(q[1] & 0xFF), q[0]); break;
    case 0xF: break;
    default: reg_write(desc, q[0]); break;           // TEX0, CLAMP, XYZF3, XYZ3: the doubleword as is
    }
}

// ---- recording ------------------------------------------------------------------------------------
// PS2KIT_GS_DUMP=file[,start[,fields]]: from the first vertical blank at or
// after `start` seconds, the GS's whole state, then every quadword given to
// paths 1, 2 and 3 and a mark at each vertical blank with the PCRTC's registers,
// for `fields` blanks (default 250). tools: gsreplay. Records: u32 type,
// u32 size in bytes, the bytes. Types: 'S' state, 1, 2 and 3 path data, 'V' a
// vertical blank (7 PCRTC registers, see gs_snapshot, then the field the
// blank begins, which the picture is shown in).
FILE* g_dump;
double g_dump_at;
long g_dump_fields = 250;
bool g_dump_on;

void dump_rec(uint32_t type, const void* p, uint32_t n) {
    uint32_t h[2] = {type, n};
    std::fwrite(h, 4, 2, g_dump);
    std::fwrite(p, 1, n, g_dump);
}

template <class T> void put_raw(std::vector<uint8_t>& s, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    s.insert(s.end(), p, p + sizeof v);
}
template <class T> void get_raw(const uint8_t*& p, T& v) { std::memcpy(&v, p, sizeof v); p += sizeof v; }

}  // namespace

// The state a recording starts from: local memory, the registers, the GIF's
// packets in progress, a transfer in progress, the vertex queue. Stored as
// the runtime's own structures, so a version leads it: change it with them.
constexpr uint32_t GS_STATE_VERSION = 2;          // 2: the CLUT buffer

void gs_state_save(std::vector<uint8_t>& s) {
    barrier();
    s.clear();
    put_raw(s, GS_STATE_VERSION);
    s.insert(s.end(), g_vram.begin(), g_vram.end());
    put_raw(s, g_r); put_raw(s, g_q); put_raw(s, g_v); put_raw(s, g_nv); put_raw(s, g_strip_k);
    put_raw(s, g_gif); put_raw(s, g_siglblid);
    put_raw(s, g_clut); put_raw(s, g_cbp0); put_raw(s, g_cbp1);
    put_raw(s, g_x.active); put_raw(s, g_x.psm); put_raw(s, g_x.bp); put_raw(s, g_x.bw);
    put_raw(s, g_x.x0); put_raw(s, g_x.y0); put_raw(s, g_x.w); put_raw(s, g_x.h); put_raw(s, g_x.x); put_raw(s, g_x.y);
    uint32_t np = (uint32_t)g_x.pend.size();
    put_raw(s, np);
    s.insert(s.end(), g_x.pend.begin(), g_x.pend.end());
}

bool gs_state_load(const uint8_t* p, size_t n) {
    const uint8_t* end = p + n;
    uint32_t version;
    if (n < 4 + VRAM) return false;
    get_raw(p, version);
    if (version != GS_STATE_VERSION) return false;
    barrier();
    dirty_all();
    std::memcpy(g_vram.data(), p, VRAM);
    p += VRAM;
    get_raw(p, g_r); get_raw(p, g_q); get_raw(p, g_v); get_raw(p, g_nv); get_raw(p, g_strip_k);
    get_raw(p, g_gif); get_raw(p, g_siglblid);
    get_raw(p, g_clut); get_raw(p, g_cbp0); get_raw(p, g_cbp1);
    ++g_clut_gen;
    get_raw(p, g_x.active); get_raw(p, g_x.psm); get_raw(p, g_x.bp); get_raw(p, g_x.bw);
    get_raw(p, g_x.x0); get_raw(p, g_x.y0); get_raw(p, g_x.w); get_raw(p, g_x.h); get_raw(p, g_x.x); get_raw(p, g_x.y);
    g_x.s = surf(g_x.psm, g_x.bp, g_x.bw);
    uint32_t np;
    get_raw(p, np);
    g_x.pend.assign(p, p + np);
    p += np;
    g_csr_events = 0;
    return p == end;
}

const uint8_t* gs_vram() { barrier(W_READ); return g_vram.data(); }
void gs_set_reference(bool on) { barrier(); g_ref = on; }
uint64_t gs_pixels() { barrier(); return pixels_drawn(); }

static void record_vblank(double now, const uint64_t regs[8]) {
    if (!g_dump) return;
    if (!g_dump_on) {
        if (now < g_dump_at) return;
        std::vector<uint8_t> s;
        gs_state_save(s);
        dump_rec('S', s.data(), (uint32_t)s.size());
        g_dump_on = true;
        std::fprintf(stderr, "ps2kit: GS: recording %ld fields from %.2f s\n", g_dump_fields, now);
        return;
    }
    dump_rec('V', regs, 8 * 8);
    if (--g_dump_fields <= 0) {
        std::fclose(g_dump);
        g_dump = nullptr;
        g_dump_on = false;
        std::fprintf(stderr, "ps2kit: GS: recording ends at %.2f s\n", now);
    }
}

// ---- the interface (the hardware's lock held) ------------------------------------------------------
void gs_init() {
    build_layouts();
    start_workers();
    if (const char* spec = std::getenv("PS2KIT_GS_DUMP")) {
        std::string s = spec, file = s.substr(0, s.find(','));
        size_t c = s.find(',');
        if (c != std::string::npos) {
            g_dump_at = std::atof(s.c_str() + c + 1);
            size_t c2 = s.find(',', c + 1);
            if (c2 != std::string::npos) g_dump_fields = std::atol(s.c_str() + c2 + 1);
        }
        g_dump = std::fopen(file.c_str(), "wb");
        if (!g_dump) std::fprintf(stderr, "ps2kit: GS: cannot write %s\n", file.c_str());
    }
}

static void gif_path(int path, const uint8_t* data, uint32_t qwc) {
    if (g_dump_on) dump_rec((uint32_t)path, data, qwc * 16);
    for (uint32_t k = 0; k < qwc; ++k) {
        uint64_t q[2];
        std::memcpy(q, data + 16 * k, 16);
        ++g_gif_qw;
        Gif& G = g_gif[path];
        if (!G.in_packet || G.nloop == 0) {
            if (G.in_packet && G.nloop == 0) G.in_packet = false;
            // a GIFtag
            G.nloop = (uint32_t)field(q[0], 0, 15);
            G.eop = field(q[0], 15, 1);
            G.flg = (uint32_t)field(q[0], 58, 2);
            G.nreg = (uint32_t)field(q[0], 60, 4);
            if (G.nreg == 0) G.nreg = 16;
            G.regs = q[1];
            G.reg = 0;
            G.half = false;
            if (field(q[0], 46, 1)) reg_write(PRIM, field(q[0], 47, 11));
            G.in_packet = G.nloop > 0;
            TRACE(2, "[gs] path %d GIFtag NLOOP %u FLG %u NREG %u REGS %016llX%s\n", path, G.nloop, G.flg, G.nreg,
                  (unsigned long long)G.regs, G.eop ? " EOP" : "");
            continue;
        }
        switch (G.flg) {
        case 0:                                      // PACKED
            packed((uint32_t)(G.regs >> (4 * G.reg) & 15), q);
            if (++G.reg == G.nreg) { G.reg = 0; --G.nloop; }
            break;
        case 1:                                      // REGLIST
            for (int h = 0; h < 2 && G.nloop; ++h) {
                uint32_t desc = (uint32_t)(G.regs >> (4 * G.reg) & 15);
                if (desc != 0xE && desc != 0xF) reg_write(desc, q[h]);
                if (++G.reg == G.nreg) { G.reg = 0; --G.nloop; }
            }
            break;
        default: {                                   // IMAGE: as many quadwords as the packet has here
            uint32_t cnt = std::min<uint32_t>(G.nloop, qwc - k);
            xfer_data(data + 16 * k, 16 * (size_t)cnt);
            G.nloop -= cnt;
            k += cnt - 1;
            g_gif_qw += cnt - 1;
            break;
        }
        }
        if (G.nloop == 0) G.in_packet = false;
    }
}

// gs_gif_batch's segments, in turn.
static void gif_segs(const uint8_t* data, const std::vector<uint32_t>& segs) {
    for (uint32_t s : segs) {
        gif_path((int)(s & 3), data, s >> 2);
        data += (size_t)(s >> 2) * 16;
    }
}

// GIF_CTRL's RST: the packets in progress end, a transfer in progress too.
static void gif_reset_locked() {
    for (Gif& G : g_gif) G = Gif{};
    g_x.active = false;
    g_x.pend.clear();
    if (g_dump_on) dump_rec('G', nullptr, 0);
}

void gs_path3(const uint8_t* data, uint32_t qwc) { gif_path(3, data, qwc); }
void gs_path2(const uint8_t* data, uint32_t qwc) { gif_path(2, data, qwc); }
void gs_path1(const uint8_t* data, uint32_t qwc) { gif_path(1, data, qwc); }

// ---- the GS's own thread ----------------------------------------------------------------------------
// The GS works beside the EE, as on the console. What the GIF is given
// (gs_gif) is copied into a queue and drawn by a thread of its own; a DMA
// channel that sent it ends when the GS has taken all of it (gs_channel_end),
// so the guest waits for the GS where it would wait on the console
// (sceDmaSync, the channel's interrupt), and a game that sends faster than
// the GS draws is held back rather than queueing without end. SIGNAL and
// FINISH reach the hardware when the GS meets them; each vertical blank takes
// its place in the queue, for the picture and a recording's mark. The GS's
// state is guarded by g_gs_mu, held for each item drawn and for each picture
// taken.
// PS2KIT_GS_SYNC=1: everything at once, on the caller's thread.
void vblank_locked(double now, const uint64_t regs[8]);

namespace {
std::atomic<bool> g_present{false};                  // pictures are taken (gs_present_enable)
struct Item {
    int kind;                                        // 1, 2, 3: GIF path; 'B': several (segs); 'E': a channel ends; 'V': a vertical blank
    int ch;
    std::vector<uint8_t> data;
    double now;
    uint64_t regs[8];                                // 'V': the PCRTC's registers and FIELD
    uint32_t token = 0;                              // 'E', 'R': which transfer
    uint8_t* dst = nullptr;                          // 'R': where in EE memory (data: its size)
    uint32_t epoch = 0;                              // 1, 2, 3, 'B': the GIF's epoch it was given in
    std::vector<uint32_t> segs;                      // 'B': quadwords << 2 | path, in turn
};
std::mutex g_gs_mu;
std::mutex g_q_mu;
std::condition_variable g_q_cv;
std::deque<Item> g_queue;
bool g_async;
double g_busy_s;                                     // the GS thread's time drawing (heartbeat)
std::atomic<uint32_t> g_gif_epoch{0};                // GIF resets so far
uint32_t g_done_epoch;                               // the epoch of the last packets taken
uint64_t g_dropped_qw;                               // quadwords a reset dropped (heartbeat)
thread_local int64_t t_submit = -1;                  // gs_submit_epoch
void (*g_cb_events)(uint64_t);
void (*g_cb_end)(int, uint32_t);

void gs_thread() {
    if (const char* f = std::getenv("PS2KIT_GS_PROFILE")) profile_this_thread(f);
    for (;;) {
        Item it;
        {
            std::unique_lock<std::mutex> lk(g_q_mu);
            g_q_cv.wait(lk, [] { return !g_queue.empty(); });
            it = std::move(g_queue.front());
            g_queue.pop_front();
        }
        uint64_t e = 0;
        auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(g_gs_mu);
            if ((it.kind >= 1 && it.kind <= 3) || it.kind == 'B') {
                if (it.epoch != g_gif_epoch.load()) g_dropped_qw += it.data.size() / 16;   // a reset came after
                else {
                    if (it.epoch != g_done_epoch) { gif_reset_locked(); g_done_epoch = it.epoch; }
                    if (it.kind != 'B') gif_path(it.kind, it.data.data(), (uint32_t)(it.data.size() / 16));
                    else gif_segs(it.data.data(), it.segs);
                }
            }
            else if (it.kind == 'V') vblank_locked(it.now, it.regs);
            else if (it.kind == 'R') readback_fill(it.dst, it.data.size());
            e = g_csr_events;
            g_csr_events = 0;
            g_busy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        // The hardware's lock is taken only with the GS's released: the EE
        // side holds the hardware's lock when it waits for the GS (pictures).
        if (e) g_cb_events(e);
        if (it.kind == 'E' || it.kind == 'R') g_cb_end(it.ch, it.token);
    }
}

void enqueue(Item&& it) {
    {
        std::lock_guard<std::mutex> lk(g_q_mu);
        g_queue.push_back(std::move(it));
    }
    g_q_cv.notify_one();
}
}  // namespace

void gs_start_thread(void (*events)(uint64_t), void (*channel_end)(int, uint32_t)) {
    const char* sync = std::getenv("PS2KIT_GS_SYNC");
    if (sync && *sync && *sync != '0') return;
    g_cb_events = events;
    g_cb_end = channel_end;
    g_async = true;
    std::thread(gs_thread).detach();
}

void gs_gif(int path, const uint8_t* data, uint32_t qwc) {
    uint32_t epoch = t_submit >= 0 ? (uint32_t)t_submit : g_gif_epoch.load();
    if (!g_async) {
        if (epoch == g_gif_epoch.load()) gif_path(path, data, qwc);
        return;
    }
    Item it{path, 0, std::vector<uint8_t>(data, data + (size_t)qwc * 16), 0, {}};
    it.epoch = epoch;
    enqueue(std::move(it));
}

void gs_gif_batch(std::vector<uint8_t>&& data, std::vector<uint32_t>&& segs) {
    uint32_t epoch = t_submit >= 0 ? (uint32_t)t_submit : g_gif_epoch.load();
    if (!g_async) {
        if (epoch == g_gif_epoch.load()) gif_segs(data.data(), segs);
        return;
    }
    Item it{'B', 0, std::move(data), 0, {}};
    it.segs = std::move(segs);
    it.epoch = epoch;
    enqueue(std::move(it));
}

uint32_t gs_gif_epoch() { return g_gif_epoch.load(); }
void gs_submit_epoch(uint32_t epoch) { t_submit = epoch; }

void gs_gif_reset() {
    g_gif_epoch.fetch_add(1);
    if (!g_async) {                                  // with the thread: at the first packets after it
        std::lock_guard<std::mutex> lk(g_gs_mu);
        gif_reset_locked();
        g_done_epoch = g_gif_epoch.load();
    }
}

bool gs_readback(uint8_t* dst, uint32_t qwc, uint32_t token) {
    if (!g_async) { readback_fill(dst, (size_t)qwc * 16); return true; }
    Item it{'R', 1, std::vector<uint8_t>((size_t)qwc * 16), 0, {}};
    it.token = token;
    it.dst = dst;
    enqueue(std::move(it));
    return false;
}

bool gs_channel_end(int ch, uint32_t token) {
    if (!g_async) return true;
    Item it{'E', ch, {}, 0, {}};
    it.token = token;
    enqueue(std::move(it));
    return false;
}

void gs_vblank(double now, const uint64_t regs[7], int shown) {
    if (!g_dump && !g_present) return;
    uint64_t r[8];
    std::memcpy(r, regs, 7 * 8);
    r[7] = (uint64_t)(shown & 1);
    if (!g_async) {
        std::lock_guard<std::mutex> lk(g_gs_mu);
        vblank_locked(now, r);
        return;
    }
    Item it{'V', 0, {}, now, {}};
    std::memcpy(it.regs, r, sizeof it.regs);
    enqueue(std::move(it));
}

uint64_t gs_take_events() {                         // with the thread, events come through its callback
    if (g_async) return 0;
    uint64_t e = g_csr_events;
    g_csr_events = 0;
    return e;
}

uint64_t gs_siglblid() {
    std::lock_guard<std::mutex> lk(g_gs_mu);
    return g_siglblid;
}

std::string gs_status() {
    std::lock_guard<std::mutex> lk(g_gs_mu);
    char s[800];
    auto b = [](int w) { return (unsigned long long)g_barriers[w]; };
    std::snprintf(s, sizeof s,
                  "GS %llu qw, %llu px, busy %.1f s, %d drawing threads (waits: texture %llu, target %llu, "
                  "transfer %llu, CLUT %llu, alone %llu, picture %llu, other %llu; %llu jobs, %llu drawn alone, %llu transfers deferred, %llu bytes)",
                  (unsigned long long)g_gif_qw, (unsigned long long)pixels_drawn(), g_busy_s, g_nw, b(W_TEXTURE),
                  b(W_TARGET), b(W_XFER), b(W_CLUT), b(W_ALONE), b(W_READ), b(W_OTHER), (unsigned long long)g_next,
                  (unsigned long long)g_alone, (unsigned long long)g_deferred_n, (unsigned long long)g_deferred);
    if (g_nw) {
        uint64_t busy = 0, sync = 0;
        for (int i = 0; i < g_nw; ++i) { busy += g_w[i].busy_us.load(); sync += g_w[i].sync_us.load(); }
        std::snprintf(s + std::strlen(s), sizeof s - std::strlen(s), "; drawing threads busy %.1f s (%.1f s at sync jobs)",
                      busy / 1e6, sync / 1e6);
        double ws = 0;
        for (double w : g_wait_s) ws += w;
        std::snprintf(s + std::strlen(s), sizeof s - std::strlen(s),
                      "; the GS's thread waited %.1f s (texture %.1f, target %.1f, picture %.1f, a slot %.1f); "
                      "texture copies %llu (%llu MB)",
                      ws, g_wait_s[W_TEXTURE], g_wait_s[W_TARGET], g_wait_s[W_READ], g_wait_s[W_N],
                      (unsigned long long)g_slab_copies, (unsigned long long)(g_slab_bytes >> 20));
    }
    if (g_gif_epoch.load())
        std::snprintf(s + std::strlen(s), sizeof s - std::strlen(s), "; GIF resets %u (%llu qw dropped)",
                      g_gif_epoch.load(), (unsigned long long)g_dropped_qw);
    return s;
}

// Interlaced pictures, one field at a time (see snapshot_locked). Each field
// holds only the even or the odd lines of the picture, so fine detail differs
// from one field to the next; a television's phosphor hides it, a monitor
// shows it as flicker. PS2KIT_DEINTERLACE picks how fields are shown:
//   blend  (default) the field and the one before, each at its own lines,
//          averaged: still where the picture is still, a little soft;
//   weave  the field's lines and the other field's lines from the one before:
//          sharp where still, combed where things move;
//   bob    the field alone, its lines doubled: sharp, flickers.
enum class Deint { blend, weave, bob };

Deint deint_mode() {
    static const Deint m = [] {
        const char* e = std::getenv("PS2KIT_DEINTERLACE");
        std::string v = e ? e : "";
        return v == "weave" ? Deint::weave : v == "bob" ? Deint::bob : Deint::blend;
    }();
    return m;
}

struct Fields {                                      // the field before, for blend and weave
    std::vector<uint32_t> rows;
    int w = 0, h = 0, odd = -1;
};

uint32_t avg_rgb(uint32_t a, uint32_t b) {           // per channel, rounded up
    return ((a | b) - (((a ^ b) & 0xFEFEFEFEu) >> 1)) | 0xFF000000u;
}

// The PCRTC's picture: the read circuit(s) of PMODE, from DISPFB and DISPLAY.
// regs: PMODE, SMODE2, DISPFB1, DISPLAY1, DISPFB2, DISPLAY2, BGCOLOR.
static bool snapshot_locked(const uint64_t regs[7], int shown, Fields& before, std::vector<uint32_t>& out, int& w,
                            int& h) {
    uint64_t pmode = regs[0], smode2 = regs[1];
    bool en1 = pmode & 1, en2 = pmode & 2;
    if (!en1 && !en2) return false;
    uint64_t fb = en2 ? regs[4] : regs[2], disp = en2 ? regs[5] : regs[3];
    if (en1 && en2) once("both PCRTC read circuits at once (circuit 2 shown)");
    uint32_t fbp = (uint32_t)field(fb, 0, 9) * 32, fbw = (uint32_t)field(fb, 9, 6), psm = (uint32_t)field(fb, 15, 5);
    int dbx = (int)field(fb, 32, 11), dby = (int)field(fb, 43, 11);
    int magh = (int)field(disp, 23, 4) + 1, magv = (int)field(disp, 27, 2) + 1;
    int dw = (int)field(disp, 32, 12) + 1, dh = (int)field(disp, 44, 11) + 1;
    w = dw / magh;
    h = dh / magv;
    static uint64_t last[7];
    if (g_trace >= 1 && std::memcmp(last, regs, sizeof last)) {
        std::memcpy(last, regs, sizeof last);
        std::fprintf(stderr, "[gs] PCRTC: PMODE %llX SMODE2 %llX DISPFB %llX (fbp %X fbw %u psm %X at %d,%d) "
                     "DISPLAY %llX (dw %d dh %d magh %d magv %d)\n", (unsigned long long)pmode,
                     (unsigned long long)smode2, (unsigned long long)fb, fbp, fbw, psm, dbx, dby,
                     (unsigned long long)disp, dw, dh, magh, magv);
    }
    // Interlaced with FFMD set, both fields read the same lines: the buffer
    // has half of DH + 1 (libgraph's sceGsSetDefDispEnv sets DH to twice the
    // height minus one with FFMD, the height minus one without). A television
    // shows the odd field one scan line below the even one, and games draw
    // for it (DW2 moves XYOFFSET's Y by half a line between fields); so each
    // field is shown as a TV would, its lines doubled, one line lower for the
    // odd field ("bob"). Shown at the same place, the picture would shake.
    // `shown` is the field the vertical blank begins. On DW2's title menu,
    // paired that way, the picture for FIELD 1 lies 0.375 buffer lines above
    // the one for FIELD 0 (measured on its logo; XYOFFSET says half a line),
    // and showing it one line lower brings them within a quarter of a screen
    // line; paired with the field before the blank, the bob doubled the shake.
    bool bob = (smode2 & 1) && (smode2 & 2);
    if (bob) h /= 2;
    if (w <= 0 || h <= 0 || w > 2048 || h > 2048) return false;
    barrier(W_READ);
    Surf sf = surf(psm, fbp, fbw);
    if (!sf.l) { once("a pixel format"); return false; }
    std::vector<uint32_t> rows((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint32_t v = sf.read(dbx + x, dby + y);
            if (psm == PSMCT16 || psm == PSMCT16S) v = (v & 31) << 3 | (v >> 5 & 31) << 11 | (v >> 10 & 31) << 19;
            rows[(size_t)y * w + x] = v | 0xFF000000u;
        }
    if (!bob) {
        out.swap(rows);
        return true;
    }
    int odd = shown & 1;
    Deint mode = deint_mode();
    bool other = before.w == w && before.h == h && before.odd == (odd ^ 1);   // the field before, the other one
    if (!other) mode = Deint::bob;
    out.resize((size_t)w * h * 2);
    for (int k = 0; k < 2 * h; ++k) {
        uint32_t* o = &out[(size_t)k * w];
        const uint32_t* cur = &rows[(size_t)std::max(0, (k - odd) >> 1) * w];
        if (mode == Deint::bob) {
            std::memcpy(o, cur, (size_t)w * 4);
            continue;
        }
        const uint32_t* prv = &before.rows[(size_t)std::max(0, (k - (odd ^ 1)) >> 1) * w];
        if (mode == Deint::weave) std::memcpy(o, (k & 1) == odd ? cur : prv, (size_t)w * 4);
        else
            for (int x = 0; x < w; ++x) o[x] = avg_rgb(cur[x], prv[x]);
    }
    before.rows.swap(rows);
    before.w = w;
    before.h = h;
    before.odd = odd;
    h *= 2;
    return true;
}

bool gs_snapshot(const uint64_t regs[7], int shown, std::vector<uint32_t>& out, int& w, int& h) {
    std::lock_guard<std::mutex> lk(g_gs_mu);
    static Fields before;                            // gsreplay's pictures, field after field
    return snapshot_locked(regs, shown, before, out, w, h);
}

// The picture shown, taken at each vertical blank in the GS's order: after
// everything sent before the blank is drawn, so never half drawn (DW2 draws
// each field into the one buffer it shows).
namespace {
std::mutex g_pic_mu;
std::vector<uint32_t> g_pic;
int g_pic_w, g_pic_h;
bool g_pic_have;
}  // namespace

void gs_present_enable() { g_present = true; }

bool gs_picture(std::vector<uint32_t>& out, int& w, int& h) {
    std::lock_guard<std::mutex> lk(g_pic_mu);
    if (!g_pic_have) return false;
    out = g_pic;
    w = g_pic_w;
    h = g_pic_h;
    return true;
}

static void present_locked(const uint64_t regs[8]) {
    std::vector<uint32_t> px;
    int w = 0, h = 0;
    static Fields before;
    bool have = snapshot_locked(regs, (int)regs[7], before, px, w, h);
    std::lock_guard<std::mutex> lk(g_pic_mu);
    g_pic_have = have;
    if (have) { g_pic.swap(px); g_pic_w = w; g_pic_h = h; }
}

void vblank_locked(double now, const uint64_t regs[8]) {
    record_vblank(now, regs);
    if (g_present) present_locked(regs);
}

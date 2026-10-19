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
// The GIF takes packets on path 3 (DMA channel 2, or the CPU at 0x10006000):
// a GIFtag (NLOOP, EOP, PRE/PRIM, FLG, NREG, REGS), then PACKED, REGLIST or
// IMAGE data. PACKED descriptors are unpacked as the GS manual gives them
// (RGBAQ's bytes at 0/32/64/96, ST and its Q, XYZF2's Z in 91:68 and F in
// 107:100, ADC at bit 111 for "no kick", A+D's register in 71:64).
//
// Drawing is a software rasteriser: points, lines, triangles (and strips,
// fans) and sprites; flat and Gouraud colour; textures (ST/Q or UV; nearest
// or bilinear; the clamp modes; TFX modulate, decal, highlight 1 and 2;
// CLUTs in CSM1 read from memory at draw time); fog; alpha, destination
// alpha and depth tests; alpha blending with PABE and COLCLAMP; FBA;
// FBMSK and ZMSK; scissor. Pixels are sampled at integer coordinates, with
// a top-left fill rule. What is not done yet is said once when first used.
//
// The PCRTC: PMODE, SMODE2, DISPFBn and DISPLAYn decide which part of
// memory is shown; gs_snapshot builds that picture for the window.
#include "os.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

namespace {

constexpr uint32_t VRAM = 4u << 20;
std::vector<uint8_t> g_vram(VRAM);

void once(const char* what) {
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
    int pw, ph;
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

int base_of(uint32_t psm) {
    switch (psm) {
    case PSMCT32: case PSMCT24: case PSMT8H: case PSMT4HL: case PSMT4HH: case PSMZ32: case PSMZ24: return 0;
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: return 1;
    case PSMT8: return 2;
    case PSMT4: return 3;
    }
    return -1;
}

// Bit address of (x, y) in a buffer at block bp, bw * 64 pixels wide.
uint32_t addr(int base, uint32_t bp, uint32_t bw, int x, int y) {
    const Layout& l = g_lay[base];
    uint32_t ppr = std::max<uint32_t>(1, bw * 64 / (uint32_t)l.pw);
    uint32_t px = (uint32_t)x & 2047, py = (uint32_t)y & 2047;
    uint32_t page = (py / l.ph) * ppr + px / l.pw;
    uint32_t bits = (bp * 2048 + page * 65536 + l.off[(py % l.ph) * l.pw + px % l.pw]);
    return bits & (VRAM * 8 - 1);
}

uint32_t rd32(uint32_t bits) { uint32_t v; std::memcpy(&v, &g_vram[bits >> 3], 4); return v; }
void wr32(uint32_t bits, uint32_t v) { std::memcpy(&g_vram[bits >> 3], &v, 4); }
uint16_t rd16(uint32_t bits) { uint16_t v; std::memcpy(&v, &g_vram[bits >> 3], 2); return v; }
void wr16(uint32_t bits, uint16_t v) { std::memcpy(&g_vram[bits >> 3], &v, 2); }

// A pixel's stored value in its format.
uint32_t pix_read(uint32_t psm, uint32_t bp, uint32_t bw, int x, int y) {
    int base = base_of(psm);
    if (base < 0) { once("a pixel format"); return 0; }
    uint32_t a = addr(base, bp, bw, x, y);
    switch (psm) {
    case PSMCT32: case PSMZ32: return rd32(a);
    case PSMCT24: case PSMZ24: return rd32(a) & 0xFFFFFF;
    case PSMT8H: return rd32(a) >> 24;
    case PSMT4HL: return rd32(a) >> 24 & 15;
    case PSMT4HH: return rd32(a) >> 28;
    case PSMT8: return g_vram[a >> 3];
    case PSMT4: return (uint32_t)(g_vram[a >> 3] >> (a & 4)) & 15;
    default: return rd16(a);
    }
}

void pix_write(uint32_t psm, uint32_t bp, uint32_t bw, int x, int y, uint32_t v, uint32_t keep = 0) {
    int base = base_of(psm);
    if (base < 0) { once("a pixel format"); return; }
    uint32_t a = addr(base, bp, bw, x, y);
    switch (psm) {
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

// ---- host-to-local transfers ----------------------------------------------------------------------
struct Xfer {
    bool active = false;
    uint32_t psm, bp, bw;
    int x0, y0, w, h, x, y;
    std::vector<uint8_t> pend;                       // bytes of a pixel split between quadwords
} g_x;

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
        TRACE(2, "[gs] host to local: %ux%u at (%d,%d), bp %X bw %u psm %02X\n", w, h, g_x.x0, g_x.y0, g_x.bp,
              g_x.bw, g_x.psm);
    } else if (dir == 2) {                           // local to local
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
    } else if (dir == 1) {
        once("local-to-host transfers");
    }
}

void xfer_pixel(uint32_t v) {
    if (!g_x.active) return;
    pix_write(g_x.psm, g_x.bp, g_x.bw, g_x.x, g_x.y, v);
    if (++g_x.x >= g_x.x0 + g_x.w) {
        g_x.x = g_x.x0;
        if (++g_x.y >= g_x.y0 + g_x.h) g_x.active = false;
    }
}

void xfer_data(const uint8_t* q, size_t n) {         // n bytes of IMAGE / HWREG data
    if (!g_x.active) return;
    size_t bpp = 0;
    switch (g_x.psm) {
    case PSMCT32: case PSMZ32: bpp = 4; break;
    case PSMCT24: case PSMZ24: bpp = 3; break;
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: bpp = 2; break;
    case PSMT8: case PSMT8H: bpp = 1; break;
    case PSMT4: case PSMT4HL: case PSMT4HH:
        for (size_t k = 0; k < n && g_x.active; ++k) { xfer_pixel(q[k] & 15); xfer_pixel(q[k] >> 4); }
        return;
    default: once("this transfer format"); return;
    }
    for (size_t k = 0; k < n && g_x.active; ++k) {
        g_x.pend.push_back(q[k]);
        if (g_x.pend.size() == bpp) {
            uint32_t v = 0;
            for (size_t b = 0; b < bpp; ++b) v |= (uint32_t)g_x.pend[b] << (8 * b);
            g_x.pend.clear();
            xfer_pixel(v);
        }
    }
}

// ---- drawing ------------------------------------------------------------------------------------
struct Vtx {
    float x, y;                                      // window coordinates (pixels)
    double z;
    float r, g, b, a, f;
    float s, t, q, u, v;                             // u, v in texels
};
Vtx g_v[3];
int g_nv;                                            // vertices in the queue
int g_strip_k;                                       // vertices since the primitive began

struct Ctx {
    uint64_t prim;                                   // the attributes in use (PRIM or PRMODE)
    int ctxt;
};

uint32_t g_draw_px;                                  // pixels drawn (heartbeat)

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

uint32_t clut_entry(const Tex& t, uint32_t idx) {
    if (t.csm) { once("CLUT storage mode CSM2"); }
    uint32_t i;
    if (t.psm == PSMT4 || t.psm == PSMT4HL || t.psm == PSMT4HH) i = t.csa * 16 + idx;
    else i = idx;
    // CSM1: a 16 x 16 rectangle, entries 8-15 and 16-23 of every 32 swapped.
    uint32_t j = (i & ~0x18u) | ((i & 8) << 1) | ((i & 16) >> 1);
    int x = (int)(j & 15), y = (int)(j >> 4);
    if (t.cpsm == PSMCT32) return pix_read(PSMCT32, t.cbp, 1, x, y);
    return rgba_from(pix_read(PSMCT16, t.cbp, 1, x, y), PSMCT16, t);
}

uint32_t texel(const Tex& t, int u, int v) {
    int w = 1 << t.tw, h = 1 << t.th;
    auto wrap = [](int c, int size, uint32_t mode, uint32_t mn, uint32_t mx) {
        switch (mode) {
        case 0: return c & (size - 1);                       // REPEAT
        case 1: return std::max(0, std::min(size - 1, c));   // CLAMP
        case 2: return std::max((int)mn, std::min((int)mx, c));   // REGION_CLAMP
        default: return (int)(((uint32_t)c & mn) | mx);      // REGION_REPEAT
        }
    };
    u = wrap(u, w, t.wms, t.minu, t.maxu);
    v = wrap(v, h, t.wmt, t.minv, t.maxv);
    uint32_t raw = pix_read(t.psm, t.tbp, t.tbw, u, v);
    switch (t.psm) {
    case PSMT8: case PSMT8H: case PSMT4: case PSMT4HL: case PSMT4HH: return clut_entry(t, raw);
    default: return rgba_from(raw, t.psm, t);
    }
}

uint32_t sample(const Tex& t, float u, float v) {      // u, v in texels
    if (!t.linear) return texel(t, (int)std::floor(u), (int)std::floor(v));
    float fu = u - 0.5f, fv = v - 0.5f;
    int u0 = (int)std::floor(fu), v0 = (int)std::floor(fv);
    float au = fu - u0, av = fv - v0;
    uint32_t c00 = texel(t, u0, v0), c10 = texel(t, u0 + 1, v0), c01 = texel(t, u0, v0 + 1), c11 = texel(t, u0 + 1, v0 + 1);
    uint32_t out = 0;
    for (int k = 0; k < 32; k += 8) {
        float a = (float)(c00 >> k & 255) * (1 - au) + (float)(c10 >> k & 255) * au;
        float b = (float)(c01 >> k & 255) * (1 - au) + (float)(c11 >> k & 255) * au;
        out |= (uint32_t)std::lround(a * (1 - av) + b * av) << k;
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

// One pixel through the pipeline: tests, blending, the write.
struct Pix {
    int ctxt;
    uint64_t prim;
    const Tex* tex;
};

void put(const Pix& P, int x, int y, double zf, float r, float g, float b, float a, float fog, float u, float v) {
    int c = P.ctxt;
    uint64_t sc = g_r[SCISSOR_1 + c];
    if (x < (int)field(sc, 0, 11) || x > (int)field(sc, 16, 11) || y < (int)field(sc, 32, 11) || y > (int)field(sc, 48, 11)) return;
    uint64_t fr = g_r[FRAME_1 + c], zb = g_r[ZBUF_1 + c], te = g_r[TEST_1 + c];
    uint32_t fbp = (uint32_t)field(fr, 0, 9) * 32, fbw = (uint32_t)field(fr, 16, 6), fpsm = (uint32_t)field(fr, 24, 6);
    uint32_t fbmsk = (uint32_t)field(fr, 32, 32);
    int R = (int)r, G = (int)g, B = (int)b, A = (int)a;
    if (P.tex) {
        uint32_t tc = sample(*P.tex, u, v);
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
        int f = (int)fog;
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
        z = (uint32_t)std::min(zmax, std::max(0.0, zf));
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

void draw_sprite(uint64_t prim, const Vtx& a, const Vtx& b) {
    int c = (int)field(prim, 9, 1);
    Tex t;
    const Tex* tp = nullptr;
    if (field(prim, 4, 1)) { t = tex_of(c); tp = &t; }
    Pix P = pix_of(prim, tp);
    float x0 = std::min(a.x, b.x), x1 = std::max(a.x, b.x), y0 = std::min(a.y, b.y), y1 = std::max(a.y, b.y);
    int ix0 = (int)std::ceil(x0), ix1 = (int)std::ceil(x1), iy0 = (int)std::ceil(y0), iy1 = (int)std::ceil(y1);
    bool fst = field(prim, 8, 1);
    float ua = fst ? a.u : a.s / a.q * (float)(1 << (tp ? tp->tw : 0)), ub = fst ? b.u : b.s / b.q * (float)(1 << (tp ? tp->tw : 0));
    float va = fst ? a.v : a.t / a.q * (float)(1 << (tp ? tp->th : 0)), vb = fst ? b.v : b.t / b.q * (float)(1 << (tp ? tp->th : 0));
    float dx = b.x - a.x, dy = b.y - a.y;
    for (int y = iy0; y < iy1; ++y)
        for (int x = ix0; x < ix1; ++x) {
            float fu = dx != 0 ? ((float)x - a.x) / dx : 0, fv = dy != 0 ? ((float)y - a.y) / dy : 0;
            put(P, x, y, b.z, b.r, b.g, b.b, b.a, b.f, ua + (ub - ua) * fu, va + (vb - va) * fv);
        }
}

void draw_triangle(uint64_t prim, const Vtx& v0, const Vtx& v1, const Vtx& v2) {
    int c = (int)field(prim, 9, 1);
    Tex t;
    const Tex* tp = nullptr;
    if (field(prim, 4, 1)) { t = tex_of(c); tp = &t; }
    Pix P = pix_of(prim, tp);
    bool gouraud = field(prim, 3, 1), fst = field(prim, 8, 1);
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
    if (area == 0) return;
    int x0 = (int)std::ceil(std::min({v0.x, v1.x, v2.x})), x1 = (int)std::ceil(std::max({v0.x, v1.x, v2.x}));
    int y0 = (int)std::ceil(std::min({v0.y, v1.y, v2.y})), y1 = (int)std::ceil(std::max({v0.y, v1.y, v2.y}));
    uint64_t sc = g_r[SCISSOR_1 + c];
    x0 = std::max(x0, (int)field(sc, 0, 11)); x1 = std::min(x1, (int)field(sc, 16, 11) + 1);
    y0 = std::max(y0, (int)field(sc, 32, 11)); y1 = std::min(y1, (int)field(sc, 48, 11) + 1);
    const Vtx* V[3] = {&v0, &v1, &v2};
    float tw = tp ? (float)(1 << tp->tw) : 1, th = tp ? (float)(1 << tp->th) : 1;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            float w[3];
            bool in = true;
            for (int k = 0; k < 3 && in; ++k) {
                const Vtx& p = *V[(k + 1) % 3];
                const Vtx& q = *V[(k + 2) % 3];
                float e = (q.x - p.x) * ((float)y - p.y) - (q.y - p.y) * ((float)x - p.x);
                e = area > 0 ? e : -e;
                // top-left rule: points on a left or top edge are in
                bool edge_tl = area > 0 ? ((q.y - p.y) > 0 || ((q.y - p.y) == 0 && (q.x - p.x) < 0))
                                        : ((q.y - p.y) < 0 || ((q.y - p.y) == 0 && (q.x - p.x) > 0));
                if (e < 0 || (e == 0 && !edge_tl)) in = false;
                w[k] = e;
            }
            if (!in) continue;
            float sum = w[0] + w[1] + w[2];
            float l0 = w[0] / sum, l1 = w[1] / sum, l2 = w[2] / sum;
            const Vtx& s = gouraud ? v0 : v2;
            float r = gouraud ? l0 * v0.r + l1 * v1.r + l2 * v2.r : s.r;
            float g = gouraud ? l0 * v0.g + l1 * v1.g + l2 * v2.g : s.g;
            float b = gouraud ? l0 * v0.b + l1 * v1.b + l2 * v2.b : s.b;
            float a = gouraud ? l0 * v0.a + l1 * v1.a + l2 * v2.a : s.a;
            double z = l0 * v0.z + l1 * v1.z + l2 * v2.z;
            float f = l0 * v0.f + l1 * v1.f + l2 * v2.f;
            float u = 0, v = 0;
            if (tp) {
                if (fst) { u = l0 * v0.u + l1 * v1.u + l2 * v2.u; v = l0 * v0.v + l1 * v1.v + l2 * v2.v; }
                else {
                    float sq = l0 * v0.s + l1 * v1.s + l2 * v2.s, tq = l0 * v0.t + l1 * v1.t + l2 * v2.t;
                    float qq = l0 * v0.q + l1 * v1.q + l2 * v2.q;
                    if (qq != 0) { u = sq / qq * tw; v = tq / qq * th; }
                }
            }
            put(P, x, y, z, r, g, b, a, f, u, v);
        }
}

void draw_line(uint64_t prim, const Vtx& a, const Vtx& b) {
    Pix P = pix_of(prim, nullptr);
    if (field(prim, 4, 1)) once("textured lines");
    float dx = b.x - a.x, dy = b.y - a.y;
    int n = (int)std::max(std::fabs(dx), std::fabs(dy));
    for (int k = 0; k <= n; ++k) {
        float f = n ? (float)k / (float)n : 0;
        put(P, (int)std::floor(a.x + dx * f + 0.5f), (int)std::floor(a.y + dy * f + 0.5f), a.z + (b.z - a.z) * f,
            a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f, a.a + (b.a - a.a) * f, b.f, 0, 0);
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
    v.x = ((float)field(xyz, 0, 16) - (float)field(off, 0, 16)) / 16.0f;
    v.y = ((float)field(xyz, 16, 16) - (float)field(off, 32, 16)) / 16.0f;
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
        case 0: put(pix_of(prim, nullptr), (int)v.x, (int)v.y, v.z, v.r, v.g, v.b, v.a, v.f, 0, 0); break;
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
        } else {
            g_r[r] = v;
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
} g_gif;

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

}  // namespace

// ---- the interface (the hardware's lock held) ------------------------------------------------------
void gs_init() { build_layouts(); }

void gs_path3(const uint8_t* data, uint32_t qwc) {
    for (uint32_t k = 0; k < qwc; ++k) {
        uint64_t q[2];
        std::memcpy(q, data + 16 * k, 16);
        ++g_gif_qw;
        Gif& G = g_gif;
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
            TRACE(2, "[gs] GIFtag NLOOP %u FLG %u NREG %u REGS %016llX%s\n", G.nloop, G.flg, G.nreg,
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
        default: {                                   // IMAGE
            uint8_t b[16];
            std::memcpy(b, q, 16);
            xfer_data(b, 16);
            --G.nloop;
            break;
        }
        }
        if (G.nloop == 0) G.in_packet = false;
    }
}

uint64_t gs_take_events() { uint64_t e = g_csr_events; g_csr_events = 0; return e; }
uint64_t gs_siglblid() { return g_siglblid; }

std::string gs_status() {
    char s[96];
    std::snprintf(s, sizeof s, "GS %llu qw, %u px", (unsigned long long)g_gif_qw, g_draw_px);
    return s;
}

// The PCRTC's picture: the read circuit(s) of PMODE, from DISPFB and DISPLAY.
// regs: PMODE, SMODE2, DISPFB1, DISPLAY1, DISPFB2, DISPLAY2, BGCOLOR.
bool gs_snapshot(const uint64_t regs[7], std::vector<uint32_t>& out, int& w, int& h) {
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
    // height minus one with FFMD, the height minus one without).
    if ((smode2 & 1) && (smode2 & 2)) h /= 2;
    if (w <= 0 || h <= 0 || w > 2048 || h > 2048) return false;
    out.resize((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint32_t v = pix_read(psm, fbp, fbw, dbx + x, dby + y);
            if (psm == PSMCT16 || psm == PSMCT16S) v = (v & 31) << 3 | (v >> 5 & 31) << 11 | (v >> 10 & 31) << 19;
            out[(size_t)y * w + x] = v | 0xFF000000u;
        }
    return true;
}

// ps2kit runtime — the IPU, the EE's MPEG decoder.
//
// The IPU reads a bit stream from its input FIFO (fed by DMA channel 4,
// toIPU, or by the CPU at 0x10007010) and writes results to its output
// FIFO (drained by channel 3, fromIPU, or read at 0x10007000). Software
// (the SDK's libmpeg) parses the headers itself through FDEC, decodes
// variable-length codes with VDEC, has each macroblock's blocks decoded by
// BDEC (VLC, inverse quantisation, IDCT) into signed 16-bit samples, does
// motion compensation itself, and turns the finished 8-bit pictures into
// RGB with CSC.
//
// Registers: IPU_CMD 0x10002000 (write: a command; read: DATA, bit 63
// BUSY), IPU_CTRL 0x10002010 (IFC 3:0, OFC 7:4, CBP 13:8, ECD 14, SCD 15,
// IDP 17:16, AS 20, IVF 21, QST 22, MP1 23, PCT 26:24, RST 30 written,
// BUSY 31), IPU_BP 0x10002020 (BP 6:0, IFC 11:8, FP 17:16), IPU_TOP
// 0x10002030 (the next 32 bits, bit 63 BUSY while fewer are there).
//
// What libmpeg expects of them, read in the game's copy (mpc.c, init.c,
// csc.c, libipu.c): VDEC's DATA is 0 for an invalid code, so a valid one
// carries its length in bits 31:16 and the value in 15:0 (a motion code of
// 0 must not read as an error); macroblock address increments are 1..33,
// 0x22 stuffing, 0x23 escape (+33); macroblock types have intra 1, pattern
// 2, backward 4, forward 8, quant 0x10. BDEC writes 48 quadwords: luma as a
// 16x16 raster of halfwords (the motion compensation adds its prediction
// row by row), then Cb and Cr as 8x8. CSC reads 384-byte macroblocks (Y
// 16x16, Cb 8x8, Cr 8x8) and writes 64 quadwords of RGBA each. Quantiser
// matrices come in the bit stream's order, zigzag (the SDK's default intra
// matrix starts 8, 16, 16, 19). sceIpuStopDMA rewinds channel 4 by IFC + FP
// quadwords and BCLR puts the bit pointer back, so every quadword in the
// FIFO and the bit buffer came from the channel, in order.
//
// The rest is MPEG-2 (ISO/IEC 13818-2): the code tables of Annex B, the
// inverse quantisation of 7.4, an IDCT computed in double precision (any
// IEEE 1180 IDCT may differ from the IPU's by one). CSC uses the ITU-R
// BT.601 matrix for studio-range samples, chroma repeated over 2x2 pixels.
// Checked against ffmpeg on DW2's KOEILOGO.PSS: the 73 pictures of a
// 20-second run, as libmpeg hands them to CSC (its pictures are stored by
// macroblock columns), against ffmpeg's yuv420p decode: at worst 65.8 dB,
// no sample off by more than 4 (IDCT rounding, carried through P pictures).
// Not checked against the hardware: its IDCT and CSC rounding are unknown.
// The code tables are checked at start-up (their share of the code space).
//
// Debugging aids (environment): PS2KIT_IPU_DUMP=dir writes CSC's input and
// output macroblocks; PS2KIT_IPU_BITLOG=file every bit FDEC, VDEC and BDEC
// consume (it must equal the movie's elementary stream), and with it the
// last 4096 commands are printed on a decoding error; PS2KIT_IPU_CMDLOG=file
// every command (struct Recent); PS2KIT_IPU_TRACE every command on stderr.
//
// The IPU runs on its own host thread, as the unit runs beside the CPU: a
// command that needs more data than the FIFO holds waits (BUSY stays set)
// until the channel brings some; output waits while the output FIFO is
// full and channel 3 is not running. Everything is under the hardware's
// lock; the thread lets it go while it waits.
#include "os.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct QW { uint8_t b[16]; };

// ---- variable-length code tables (ISO/IEC 13818-2 Annex B) -------------------------------------
// Each table is a lookup by the next 16 bits: length 0 means no code.
struct Vlc { uint8_t len; int16_t a, b; };
struct Table {
    std::vector<Vlc> t = std::vector<Vlc>(65536, Vlc{0, 0, 0});
    double kraft = 0;
    void add(const char* code, int a, int b = 0) {
        uint32_t v = 0, n = 0;
        for (const char* p = code; *p; ++p) {
            if (*p == ' ') continue;
            v = v << 1 | (uint32_t)(*p - '0');
            ++n;
        }
        if (n == 0 || n > 16) rt_die("IPU: bad code %s", code);
        uint32_t lo = v << (16 - n), hi = lo | ((1u << (16 - n)) - 1);
        for (uint32_t k = lo; k <= hi; ++k) {
            if (t[k].len) rt_die("IPU: code %s overlaps another", code);
            t[k] = Vlc{(uint8_t)n, (int16_t)a, (int16_t)b};
        }
        kraft += std::ldexp(1.0, -(int)n);
    }
};

enum { EOB = 100, ESC = 101 };
Table g_mbai, g_mbt[5], g_cbp, g_mv, g_dmv, g_dcl, g_dcc, g_dct0, g_dct1;

void tables() {
    // B.1 macroblock_address_increment (34 stuffing is MPEG-1's; 35 escape)
    static const char* mbai[] = {"1", "011", "010", "0011", "0010", "0001 1", "0001 0", "0000 111", "0000 110",
        "0000 1011", "0000 1010", "0000 1001", "0000 1000", "0000 0111", "0000 0110", "0000 0101 11",
        "0000 0101 10", "0000 0101 01", "0000 0101 00", "0000 0100 11", "0000 0100 10", "0000 0100 011",
        "0000 0100 010", "0000 0100 001", "0000 0100 000", "0000 0011 111", "0000 0011 110", "0000 0011 101",
        "0000 0011 100", "0000 0011 011", "0000 0011 010", "0000 0011 001", "0000 0011 000"};
    for (int k = 0; k < 33; ++k) g_mbai.add(mbai[k], k + 1);
    g_mbai.add("0000 0001 000", 0x23);
    // B.2-B.4 macroblock_type: quant 0x10, forward 8, backward 4, pattern 2, intra 1
    g_mbt[1].add("1", 0x01); g_mbt[1].add("01", 0x11);
    g_mbt[2].add("1", 0x0A); g_mbt[2].add("01", 0x02); g_mbt[2].add("001", 0x08); g_mbt[2].add("0001 1", 0x01);
    g_mbt[2].add("0001 0", 0x1A); g_mbt[2].add("0000 1", 0x12); g_mbt[2].add("0000 01", 0x11);
    g_mbt[3].add("10", 0x0C); g_mbt[3].add("11", 0x0E); g_mbt[3].add("010", 0x04); g_mbt[3].add("011", 0x06);
    g_mbt[3].add("0010", 0x08); g_mbt[3].add("0011", 0x0A); g_mbt[3].add("0001 1", 0x01);
    g_mbt[3].add("0001 0", 0x1E); g_mbt[3].add("0000 11", 0x1A); g_mbt[3].add("0000 10", 0x16);
    g_mbt[3].add("0000 01", 0x11);
    g_mbt[4].add("1", 0x01);                                         // MPEG-1 D pictures
    // B.9 coded_block_pattern
    static const struct { const char* c; int v; } cbp[] = {
        {"111", 60}, {"1101", 4}, {"1100", 8}, {"1011", 16}, {"1010", 32}, {"1001 1", 12}, {"1001 0", 48},
        {"1000 1", 20}, {"1000 0", 40}, {"0111 1", 28}, {"0111 0", 44}, {"0110 1", 52}, {"0110 0", 56},
        {"0101 1", 1}, {"0101 0", 61}, {"0100 1", 2}, {"0100 0", 62}, {"0011 11", 24}, {"0011 10", 36},
        {"0011 01", 3}, {"0011 00", 63}, {"0010 111", 5}, {"0010 110", 9}, {"0010 101", 17}, {"0010 100", 33},
        {"0010 011", 6}, {"0010 010", 10}, {"0010 001", 18}, {"0010 000", 34}, {"0001 1111", 7},
        {"0001 1110", 11}, {"0001 1101", 19}, {"0001 1100", 35}, {"0001 1011", 13}, {"0001 1010", 49},
        {"0001 1001", 21}, {"0001 1000", 41}, {"0001 0111", 14}, {"0001 0110", 50}, {"0001 0101", 22},
        {"0001 0100", 42}, {"0001 0011", 15}, {"0001 0010", 51}, {"0001 0001", 23}, {"0001 0000", 43},
        {"0000 1111", 25}, {"0000 1110", 37}, {"0000 1101", 26}, {"0000 1100", 38}, {"0000 1011", 29},
        {"0000 1010", 45}, {"0000 1001", 53}, {"0000 1000", 57}, {"0000 0111", 30}, {"0000 0110", 46},
        {"0000 0101", 54}, {"0000 0100", 58}, {"0000 0011 1", 31}, {"0000 0011 0", 47}, {"0000 0010 1", 55},
        {"0000 0010 0", 59}, {"0000 0001 1", 27}, {"0000 0001 0", 39}, {"0000 0000 1", 0}};
    for (auto& e : cbp) g_cbp.add(e.c, e.v);
    // B.10 motion_code (the sign bit follows, 1 negative), B.11 dmvector
    static const char* mv[] = {"1", "01", "001", "0001", "0000 11", "0000 101", "0000 100", "0000 011",
        "0000 0101 1", "0000 0101 0", "0000 0100 1", "0000 0100 01", "0000 0100 00", "0000 0011 11",
        "0000 0011 10", "0000 0011 01", "0000 0011 00"};
    for (int k = 0; k <= 16; ++k) g_mv.add(mv[k], k);
    g_dmv.add("0", 0); g_dmv.add("10", 1); g_dmv.add("11", -1);
    // B.12, B.13 dct_dc_size
    static const char* dcl[] = {"100", "00", "01", "101", "110", "1110", "1111 0", "1111 10", "1111 110",
        "1111 1110", "1111 1111 0", "1111 1111 1"};
    static const char* dcc[] = {"00", "01", "10", "110", "1110", "1111 0", "1111 10", "1111 110",
        "1111 1110", "1111 1111 0", "1111 1111 10", "1111 1111 11"};
    for (int k = 0; k < 12; ++k) { g_dcl.add(dcl[k], k); g_dcc.add(dcc[k], k); }
    // B.14 and B.15 DCT coefficients (run, level; the sign bit follows).
    // Table zero's "1s" for a first coefficient is handled in the decoder.
    static const struct { const char* c; int run, level; } common[] = {   // the same in both tables
        {"0000 0000 0111 11", 0, 16}, {"0000 0000 0111 10", 0, 17}, {"0000 0000 0111 01", 0, 18},
        {"0000 0000 0111 00", 0, 19}, {"0000 0000 0110 11", 0, 20}, {"0000 0000 0110 10", 0, 21},
        {"0000 0000 0110 01", 0, 22}, {"0000 0000 0110 00", 0, 23}, {"0000 0000 0101 11", 0, 24},
        {"0000 0000 0101 10", 0, 25}, {"0000 0000 0101 01", 0, 26}, {"0000 0000 0101 00", 0, 27},
        {"0000 0000 0100 11", 0, 28}, {"0000 0000 0100 10", 0, 29}, {"0000 0000 0100 01", 0, 30},
        {"0000 0000 0100 00", 0, 31}, {"0000 0000 0011 000", 0, 32}, {"0000 0000 0010 111", 0, 33},
        {"0000 0000 0010 110", 0, 34}, {"0000 0000 0010 101", 0, 35}, {"0000 0000 0010 100", 0, 36},
        {"0000 0000 0010 011", 0, 37}, {"0000 0000 0010 010", 0, 38}, {"0000 0000 0010 001", 0, 39},
        {"0000 0000 0010 000", 0, 40}, {"0000 0000 0011 111", 1, 8}, {"0000 0000 0011 110", 1, 9},
        {"0000 0000 0011 101", 1, 10}, {"0000 0000 0011 100", 1, 11}, {"0000 0000 0011 011", 1, 12},
        {"0000 0000 0011 010", 1, 13}, {"0000 0000 0011 001", 1, 14}, {"0000 0000 0001 0011", 1, 15},
        {"0000 0000 0001 0010", 1, 16}, {"0000 0000 0001 0001", 1, 17}, {"0000 0000 0001 0000", 1, 18},
        {"0000 0000 0001 0100", 6, 3}, {"0000 0000 0001 1010", 11, 2}, {"0000 0000 0001 1001", 12, 2},
        {"0000 0000 0001 1000", 13, 2}, {"0000 0000 0001 0111", 14, 2}, {"0000 0000 0001 0110", 15, 2},
        {"0000 0000 0001 0101", 16, 2}, {"0000 0000 0001 1111", 27, 1}, {"0000 0000 0001 1110", 28, 1},
        {"0000 0000 0001 1101", 29, 1}, {"0000 0000 0001 1100", 30, 1}, {"0000 0000 0001 1011", 31, 1},
        {"0000 0000 1011 0", 1, 6}, {"0000 0000 1010 1", 1, 7}, {"0000 0000 1010 0", 2, 5},
        {"0000 0000 1001 1", 3, 4}, {"0000 0000 1001 0", 5, 3}, {"0000 0000 1000 1", 9, 2},
        {"0000 0000 1000 0", 10, 2}, {"0000 0000 1111 1", 22, 1}, {"0000 0000 1111 0", 23, 1},
        {"0000 0000 1110 1", 24, 1}, {"0000 0000 1110 0", 25, 1}, {"0000 0000 1101 1", 26, 1},
        {"0000 0001 1100", 3, 3}, {"0000 0001 0010", 4, 3}, {"0000 0001 1110", 6, 2},
        {"0000 0001 0101", 7, 2}, {"0000 0001 0001", 8, 2}, {"0000 0001 1111", 17, 1},
        {"0000 0001 1010", 18, 1}, {"0000 0001 1001", 19, 1}, {"0000 0001 0111", 20, 1},
        {"0000 0001 0110", 21, 1}, {"0000 01", ESC, 0}};
    static const struct { const char* c; int run, level; } zero[] = {
        {"10", EOB, 0}, {"11", 0, 1}, {"011", 1, 1}, {"0100", 0, 2}, {"0101", 2, 1}, {"0010 1", 0, 3},
        {"0011 1", 3, 1}, {"0011 0", 4, 1}, {"0001 10", 1, 2}, {"0001 11", 5, 1}, {"0001 01", 6, 1},
        {"0001 00", 7, 1}, {"0000 110", 0, 4}, {"0000 100", 2, 2}, {"0000 111", 8, 1}, {"0000 101", 9, 1},
        {"0010 0110", 0, 5}, {"0010 0001", 0, 6}, {"0010 0101", 1, 3}, {"0010 0100", 3, 2},
        {"0010 0111", 10, 1}, {"0010 0011", 11, 1}, {"0010 0010", 12, 1}, {"0010 0000", 13, 1},
        {"0000 0010 10", 0, 7}, {"0000 0011 00", 1, 4}, {"0000 0010 11", 2, 3}, {"0000 0011 11", 4, 2},
        {"0000 0010 01", 5, 2}, {"0000 0011 10", 14, 1}, {"0000 0011 01", 15, 1}, {"0000 0010 00", 16, 1},
        {"0000 0001 1101", 0, 8}, {"0000 0001 1000", 0, 9}, {"0000 0001 0011", 0, 10},
        {"0000 0001 0000", 0, 11}, {"0000 0001 1011", 1, 5}, {"0000 0001 0100", 2, 4},
        {"0000 0000 1101 0", 0, 12}, {"0000 0000 1100 1", 0, 13}, {"0000 0000 1100 0", 0, 14},
        {"0000 0000 1011 1", 0, 15}};
    static const struct { const char* c; int run, level; } one[] = {
        {"0110", EOB, 0}, {"10", 0, 1}, {"010", 1, 1}, {"110", 0, 2}, {"0010 1", 2, 1}, {"0111", 0, 3},
        {"0011 1", 3, 1}, {"0001 10", 4, 1}, {"0011 0", 1, 2}, {"0001 11", 5, 1}, {"0000 110", 6, 1},
        {"0000 100", 7, 1}, {"1110 0", 0, 4}, {"0000 111", 2, 2}, {"0000 101", 8, 1}, {"1111 000", 9, 1},
        {"1110 1", 0, 5}, {"0001 01", 0, 6}, {"1111 001", 1, 3}, {"0010 0110", 3, 2}, {"1111 010", 10, 1},
        {"0010 0001", 11, 1}, {"0010 0101", 12, 1}, {"0010 0100", 13, 1}, {"0001 00", 0, 7},
        {"0010 0111", 1, 4}, {"1111 1100", 2, 3}, {"1111 1101", 4, 2}, {"0000 0010 0", 5, 2},
        {"0000 0010 1", 14, 1}, {"0000 0011 1", 15, 1}, {"0000 0011 01", 16, 1}, {"1111 011", 0, 8},
        {"1111 100", 0, 9}, {"0010 0011", 0, 10}, {"0010 0010", 0, 11}, {"0010 0000", 1, 5},
        {"0000 0011 00", 2, 4}, {"1111 1010", 0, 12}, {"1111 1011", 0, 13}, {"1111 1110", 0, 14},
        {"1111 1111", 0, 15}};
    for (auto& e : common) { g_dct0.add(e.c, e.run, e.level); g_dct1.add(e.c, e.run, e.level); }
    for (auto& e : zero) g_dct0.add(e.c, e.run, e.level);
    for (auto& e : one) g_dct1.add(e.c, e.run, e.level);
}

const uint8_t ZIGZAG[64] = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
const uint8_t ALTERNATE[64] = {0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3,
    11, 4, 12, 19, 27, 34, 42, 50, 58, 35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29, 36, 44, 52, 60, 37,
    45, 53, 61, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63};
const uint8_t NONLINEAR_Q[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40,
    44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112};
double g_cos[8][8];                                  // C(u)/2 cos((2x+1)u pi/16), [u][x]

// ---- the unit's state, under the hardware's lock -------------------------------------------------
std::mutex* g_lock;
std::condition_variable g_cv;
std::deque<QW> g_in;                                 // the bit buffer (FP) and the FIFO (IFC), in order
uint32_t g_bp;                                       // bit position in g_in[0]
std::deque<QW> g_out;                                // the output FIFO
uint32_t g_ctrl;                                     // the stored fields of IPU_CTRL
uint32_t g_cmd, g_data;                              // the command running, its result
bool g_busy, g_running;
uint32_t g_reset_gen;
uint8_t g_iq[2][64];                                 // intra, non-intra; raster order
uint16_t g_vq[16];
uint32_t g_th0, g_th1;
int g_dc_pred[3];
bool g_wake;                                         // an interrupt was raised: wake the kernel
uint64_t g_out_hi;                                   // the upper half of an output FIFO read
uint64_t g_cmds[10];
double g_cmd_secs[10];                              // host time each command kind took (heartbeat)

struct Abort {};
uint32_t g_run_gen;                                  // g_reset_gen when the running command began
bool g_ch4_fresh;                                    // channel 4 started with QWC 0: its first tag is unread

bool ch_running(int ch) { return hw_dma_reg(ch, 0x00) & 0x100; }

// The tag in CHCR's TAG field (the tag's bits 31:16: IRQ, ID, PCE) ends the
// chain once its data is sent: refe (0), end (7), or IRQ with TIE.
bool tag_ends(uint32_t chcr) {
    uint32_t id = (chcr >> 28) & 7;
    return id == 0 || id == 7 || ((chcr & 0x80) && (chcr >> 31));
}

// One quadword from channel 4 (toIPU), as the DMAC takes it: QWC from MADR,
// then (in chain mode) the next tag. Returns false when the channel has
// nothing. The channel ends right after its last quadword (libmpeg asks
// for more data when it sees QWC 0 and STR clear). A restart with QWC > 0
// (sceIpuRestartDMA) carries the old CHCR, so its TAG field still says
// whether that tag ends the chain.
bool ch4_take(QW& q) {
    uint32_t& chcr = hw_dma_reg(4, 0x00);
    uint32_t& madr = hw_dma_reg(4, 0x10);
    uint32_t& qwc = hw_dma_reg(4, 0x20);
    uint32_t& tadr = hw_dma_reg(4, 0x30);
    if (!(chcr & 0x100)) return false;
    int mode = (chcr >> 2) & 3;
    if (mode > 1) rt_die("IPU: toIPU in mode %d is not modelled", mode);
    while (qwc == 0) {
        if (mode == 0 || (!g_ch4_fresh && tag_ends(chcr))) { hw_dma_end_locked(4); g_wake = true; return false; }
        g_ch4_fresh = false;
        uint64_t tag[2];
        std::memcpy(tag, hw_dma_ptr(tadr, 1), 16);
        uint32_t n = tag[0] & 0xFFFF, id = (tag[0] >> 28) & 7;
        uint32_t addr = (uint32_t)(tag[0] >> 32) & 0xFFFFFFF0u;
        switch (id) {
        case 0: madr = addr; tadr += 16; break;                                     // refe
        case 1: madr = tadr + 16; tadr = madr + n * 16; break;                      // cnt
        case 2: madr = tadr + 16; tadr = addr; break;                               // next
        case 3: case 4: madr = addr; tadr += 16; break;                             // ref, refs
        case 7: madr = tadr + 16; tadr = madr + n * 16; break;                      // end
        default: rt_die("IPU: toIPU tag %u (call/ret) is not modelled", id);
        }
        qwc = n;
        chcr = (chcr & 0xFFFF) | (uint32_t)((tag[0] >> 16) & 0xFFFF0000u);
        TRACE(2, "[ipu] toIPU tag %u: %u qw from %08X\n", id, n, madr);
    }
    std::memcpy(q.b, hw_dma_ptr(madr, 1), 16);
    madr += 16;
    --qwc;
    if (qwc == 0 && (mode == 0 || tag_ends(chcr))) { hw_dma_end_locked(4); g_wake = true; }
    return true;
}

// The DMAC keeps the FIFO full while channel 4 runs: 8 quadwords, and 2 in
// the bit buffer.
void fill() {
    QW q;
    while (g_in.size() < 10 && ch4_take(q)) g_in.push_back(q);
}

void drain() {
    while (!g_out.empty() && ch_running(3)) {
        uint32_t& madr = hw_dma_reg(3, 0x10);
        uint32_t& qwc = hw_dma_reg(3, 0x20);
        if (qwc == 0) { hw_dma_end_locked(3); g_wake = true; break; }
        std::memcpy(hw_dma_ptr(madr, 1), g_out.front().b, 16);
        g_out.pop_front();
        madr += 16;
        if (--qwc == 0) { hw_dma_end_locked(3); g_wake = true; }
    }
}

uint32_t avail_bits() { return (uint32_t)g_in.size() * 128 - g_bp; }

int bit_at(uint32_t k) {
    uint32_t p = g_bp + k;
    return (g_in[p >> 7].b[(p >> 3) & 15] >> (7 - (p & 7))) & 1;
}

std::unique_lock<std::mutex>* g_lk;                  // the IPU thread's hold on the lock

// The lock let go and taken again (an interrupt raised meanwhile wakes the
// kernel, which must not be called with the hardware's lock held). A reset
// in between ends the running command.
void relock(bool and_wait) {
    if (g_wake) { g_wake = false; g_lk->unlock(); kernel_wake(); g_lk->lock(); }
    if (and_wait) g_cv.wait(*g_lk);
    else { g_lk->unlock(); g_lk->lock(); }
    if (g_run_gen != g_reset_gen) throw Abort{};
}
void wait(std::unique_lock<std::mutex>&) { relock(true); }

void need(uint32_t n) {
    for (;;) {
        fill();
        if (avail_bits() >= n) return;
        wait(*g_lk);
    }
}

uint32_t peek(uint32_t n) {                         // n <= 32
    need(n);
    uint32_t v = 0;
    for (uint32_t k = 0; k < n; ++k) v = v << 1 | (uint32_t)bit_at(k);
    return v;
}

uint64_t g_consumed;                                 // bits consumed since the last BCLR (for traces)
bool g_ipu_trace;                                    // PS2KIT_IPU_TRACE: every command, where it read
FILE* g_dump_in;                                     // PS2KIT_IPU_DUMP: CSC input (RAW8 macroblocks)
FILE* g_dump_out;                                    // and output (RGBA macroblocks)

// PS2KIT_IPU_BITLOG: every bit FDEC, VDEC and BDEC consume, in order. libmpeg
// reads the whole video stream through them, so the log should equal the
// movie's elementary stream bit for bit.
FILE* g_bitlog;
FILE* g_cmdlog;                                      // PS2KIT_IPU_CMDLOG: every command (struct Recent)
bool g_logging;
uint64_t g_logpos;                                   // bits logged so far
struct Recent { uint32_t cmd, data, ctrl, top; uint64_t from, to; };
Recent g_recent[4096];                               // the last commands, dumped on an error
uint32_t g_nrecent;
uint8_t g_logbyte;
int g_lognbits;

void skip(uint32_t n) {
    need(n);
    if (g_bitlog && g_logging)
        for (uint32_t k = 0; k < n; ++k) {
            ++g_logpos;
            g_logbyte = (uint8_t)(g_logbyte << 1 | bit_at(k));
            if (++g_lognbits == 8) {
                std::fputc(g_logbyte, g_bitlog);
                g_lognbits = 0;
                if ((std::ftell(g_bitlog) & 0xFFFF) == 0) std::fflush(g_bitlog);
            }
        }
    g_consumed += n;
    g_bp += n;
    while (g_bp >= 128) { g_in.pop_front(); g_bp -= 128; }
    fill();
}

uint32_t get(uint32_t n) { uint32_t v = n ? peek(n) : 0; skip(n); return v; }

// The next code of a table; false (nothing consumed) for none.
bool vlc(const Table& t, Vlc& e) {
    // Up to 16 bits: fewer may be enough near the end of the data.
    fill();
    uint32_t have = std::min<uint32_t>(16, avail_bits());
    for (;;) {
        uint32_t v = 0;
        for (uint32_t k = 0; k < have; ++k) v = v << 1 | (uint32_t)bit_at(k);
        e = t.t[(v << (16 - have)) & 0xFFFF];
        if (e.len && e.len <= have) { skip(e.len); return true; }
        if (have == 16) return false;
        wait(*g_lk);
        fill();
        have = std::min<uint32_t>(16, avail_bits());
    }
}

void put_out(const uint8_t* p, size_t n) {           // n quadwords
    for (size_t k = 0; k < n; ++k) {
        while (g_out.size() >= 8) {
            drain();
            if (g_out.size() >= 8) wait(*g_lk);
        }
        QW q;
        std::memcpy(q.b, p + 16 * k, 16);
        g_out.push_back(q);
        drain();
    }
}

// ---- the commands ---------------------------------------------------------------------------------
uint32_t ctrl_field(int shift, int bits) { return (g_ctrl >> shift) & ((1u << bits) - 1); }

void error(const char* what) {                       // ECD
    g_ctrl |= 1u << 14;
    if (g_bitlog) {
        std::fflush(g_bitlog);
        uint32_t n = std::min<uint32_t>(g_nrecent, 4096);
        for (uint32_t k = g_nrecent - n; k < g_nrecent; ++k) {
            const Recent& r = g_recent[k % 4096];
            std::fprintf(stderr, "[ipu] recent %08X bits %llu-%llu DATA %08X CTRL %08X TOP %08X\n", r.cmd,
                         (unsigned long long)r.from, (unsigned long long)r.to, r.data, r.ctrl, r.top);
        }
        std::fprintf(stderr, "[ipu] now %08X from bit %llu\n", g_cmd, (unsigned long long)g_logpos);
    }
    if (g_trace >= 1) {
        uint32_t have = std::min<uint32_t>(32, avail_bits()), v = 0;
        for (uint32_t k = 0; k < have; ++k) v = v << 1 | (uint32_t)bit_at(k);
        std::fprintf(stderr, "[ipu] error in %s (command %08X, CTRL %08X): next %u bits %08X\n", what, g_cmd,
                     g_ctrl, have, v << (32 - have));
    }
}

uint32_t do_vdec(uint32_t cmd) {
    skip(cmd & 0x3F);
    int tbl = (cmd >> 26) & 3;
    Vlc e;
    switch (tbl) {
    case 0: {                                        // macroblock_address_increment
        if (ctrl_field(23, 1) && peek(11) == 0x00F) { skip(11); return 11u << 16 | 0x22; }   // MPEG-1 stuffing
        if (!vlc(g_mbai, e)) { error("VDEC MBAI"); return 0; }
        return (uint32_t)e.len << 16 | (uint16_t)e.a;
    }
    case 1: {                                        // macroblock_type, by the picture type
        uint32_t pct = ctrl_field(24, 3);
        if (pct < 1 || pct > 4) { error("VDEC MBT picture type"); return 0; }
        if (!vlc(g_mbt[pct], e)) { error("VDEC MBT"); return 0; }
        return (uint32_t)e.len << 16 | (uint16_t)e.a;
    }
    case 2: {                                        // motion_code
        if (!vlc(g_mv, e)) { error("VDEC MV"); return 0; }
        int v = e.a, len = e.len;
        if (v) { if (get(1)) v = -v; ++len; }
        return (uint32_t)len << 16 | (uint16_t)(int16_t)v;
    }
    default:                                         // dmvector
        if (!vlc(g_dmv, e)) { error("VDEC DMV"); return 0; }
        return (uint32_t)e.len << 16 | (uint16_t)e.a;
    }
}

// One block: coefficients in raster order, inverse quantised (7.4).
bool block(int b, bool intra, int qscale, int16_t out[64]) {
    int F[64] = {};
    bool mpeg1 = ctrl_field(23, 1), alt = ctrl_field(20, 1);
    const uint8_t* scan = alt ? ALTERNATE : ZIGZAG;
    const uint8_t* W = g_iq[intra ? 0 : 1];
    int n = 0;
    if (intra) {
        Vlc e;
        if (!vlc(b < 4 ? g_dcl : g_dcc, e)) return false;
        int size = e.a, diff = 0;
        if (size) {
            int v = (int)get((uint32_t)size);
            diff = v < (1 << (size - 1)) ? v - ((1 << size) - 1) : v;
        }
        int c = b < 4 ? 0 : b - 3;
        g_dc_pred[c] += diff;
        F[0] = mpeg1 ? g_dc_pred[c] * 8 : g_dc_pred[c] * (8 >> ctrl_field(16, 2));
        n = 1;
    }
    const Table& t = (intra && ctrl_field(21, 1) && !mpeg1) ? g_dct1 : g_dct0;
    bool first = !intra;
    for (;;) {
        int run, level;
        if (first && peek(1)) {                      // table zero, first coefficient: "1s"
            skip(1);
            run = 0;
            level = get(1) ? -1 : 1;
        } else {
            Vlc e;
            if (!vlc(t, e)) return false;
            if (e.a == EOB) break;
            if (e.a == ESC) {
                run = (int)get(6);
                if (mpeg1) {
                    int v = (int)get(8);
                    if (v == 0) level = (int)get(8);
                    else if (v == 128) level = (int)get(8) - 256;
                    else level = v >= 128 ? v - 256 : v;
                } else {
                    int v = (int)get(12);
                    level = v >= 2048 ? v - 4096 : v;
                    if (level == 0 || level == -2048) return false;
                }
            } else {
                run = e.a;
                level = get(1) ? -e.b : e.b;
            }
        }
        first = false;
        n += run;
        if (n > 63) return false;
        int pos = scan[n++];
        int v;
        if (intra) {
            v = mpeg1 ? (level * qscale * W[pos]) / 8 : (level * W[pos] * qscale * 2) / 32;
        } else {
            int s = level > 0 ? 1 : -1;
            v = mpeg1 ? ((2 * level + s) * qscale * W[pos]) / 16 : ((2 * level + s) * W[pos] * qscale) / 32;
        }
        if (mpeg1 && v != 0 && (v & 1) == 0) v -= v > 0 ? 1 : -1;     // oddification
        F[pos] = std::max(-2048, std::min(2047, v));
        if (n > 63) {                                // the block is full: only EOB may follow
            Vlc e;
            if (!vlc(t, e) || e.a != EOB) return false;
            break;
        }
    }
    if (intra) F[0] = std::max(-2048, std::min(2047, F[0]));
    if (!mpeg1) {                                    // mismatch control
        int sum = 0;
        for (int k = 0; k < 64; ++k) sum += F[k];
        if ((sum & 1) == 0) F[63] ^= 1;
    }
    double tmp[64];
    for (int y = 0; y < 8; ++y)
        for (int u = 0; u < 8; ++u) {
            double s = 0;
            for (int v = 0; v < 8; ++v) s += g_cos[v][y] * F[v * 8 + u];
            tmp[y * 8 + u] = s;
        }
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            double s = 0;
            for (int u = 0; u < 8; ++u) s += g_cos[u][x] * tmp[y * 8 + u];
            int r = (int)std::floor(s + 0.5);
            out[y * 8 + x] = (int16_t)std::max(-256, std::min(255, r));
        }
    return true;
}

void do_bdec(uint32_t cmd) {
    skip(cmd & 0x3F);
    bool intra = cmd & (1u << 27), dcr = cmd & (1u << 26), dt = cmd & (1u << 25);
    int qsc = (cmd >> 16) & 0x1F;
    int qscale = ctrl_field(23, 1) ? qsc : ctrl_field(22, 1) ? NONLINEAR_Q[qsc] : qsc * 2;
    if (dcr) {
        int reset = ctrl_field(23, 1) ? 128 : 1 << (7 + ctrl_field(16, 2));
        g_dc_pred[0] = g_dc_pred[1] = g_dc_pred[2] = reset;
    }
    int cbp = 0x3F;
    if (!intra) {
        Vlc e;
        if (!vlc(g_cbp, e)) { error("BDEC CBP"); return; }
        cbp = e.a;
    }
    g_ctrl = (g_ctrl & ~0x3F00u) | (uint32_t)cbp << 8;
    int16_t mb[384] = {};
    for (int b = 0; b < 6; ++b) {
        if (!(cbp & (1 << (5 - b)))) continue;
        int16_t blk[64];
        if (!block(b, intra, qscale, blk)) { error("BDEC block"); return; }
        if (b < 4) {
            int bx = (b & 1) * 8;
            for (int r = 0; r < 8; ++r) {
                int row = dt ? 2 * r + (b >> 1) : (b >> 1) * 8 + r;
                std::memcpy(&mb[row * 16 + bx], &blk[r * 8], 16);
            }
        } else {
            std::memcpy(&mb[256 + (b - 4) * 64], blk, 128);
        }
    }
    put_out(reinterpret_cast<const uint8_t*>(mb), 48);
}

void do_csc(uint32_t cmd, std::unique_lock<std::mutex>& lk) {
    bool rgb16 = cmd & (1u << 27), dither = cmd & (1u << 26);
    if (rgb16 || dither) rt_die("IPU: CSC to RGB16 or with dithering is not modelled yet");
    (void)lk;
    uint32_t mbc = cmd & 0x7FF;
    for (uint32_t m = 0; m < mbc; ++m) {
        uint8_t raw[384];                            // 24 quadwords: more than the FIFO holds
        for (int k = 0; k < 384;) {
            if ((g_bp & 7) == 0 && !g_bitlog) {      // whole bytes straight from the FIFO
                need(8);
                uint32_t n = std::min<uint32_t>(384 - k, (avail_bits() >> 3));
                n = std::min<uint32_t>(n, 16 - (g_bp >> 3));
                std::memcpy(raw + k, &g_in[0].b[g_bp >> 3], n);
                k += (int)n;
                skip(n * 8);
            } else {
                raw[k++] = (uint8_t)get(8);
            }
        }
        if (g_dump_in) std::fwrite(raw, 1, sizeof raw, g_dump_in);
        uint8_t rgba[1024];
        static double tY[256], tRcr[256], tGcb[256], tGcr[256], tBcb[256];
        static bool tables_made;
        if (!tables_made) {
            for (int k = 0; k < 256; ++k) {
                tY[k] = 1.164 * (k - 16);
                tRcr[k] = 1.596 * (k - 128.0);
                tGcb[k] = 0.391 * (k - 128.0);
                tGcr[k] = 0.813 * (k - 128.0);
                tBcb[k] = 2.018 * (k - 128.0);
            }
            tables_made = true;
        }
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x) {
                double Y = tY[raw[y * 16 + x]];
                int cb = raw[256 + (y >> 1) * 8 + (x >> 1)], cr = raw[320 + (y >> 1) * 8 + (x >> 1)];
                int c[3] = {(int)std::floor(Y + tRcr[cr] + 0.5),
                            (int)std::floor(Y - tGcb[cb] - tGcr[cr] + 0.5),
                            (int)std::floor(Y + tBcb[cb] + 0.5)};
                uint8_t* o = &rgba[(y * 16 + x) * 4];
                for (int k = 0; k < 3; ++k) o[k] = (uint8_t)std::max(0, std::min(255, c[k]));
                uint32_t r = o[0], g = o[1], b = o[2];
                if (r < g_th0 && g < g_th0 && b < g_th0) { o[0] = o[1] = o[2] = 0; o[3] = 0; }
                else if (r < g_th1 && g < g_th1 && b < g_th1) o[3] = 0x40;
                else o[3] = 0x80;
            }
        if (g_dump_out) std::fwrite(rgba, 1, sizeof rgba, g_dump_out);
        put_out(rgba, 64);
        // Let the clock and the guest in between macroblocks. Not after the
        // last: the command ends with its last output, and the end of
        // channel 3 may start the next one (libmpeg's _ch3dmaCSC).
        if (m + 1 < mbc) relock(false);
    }
}

void run(uint32_t cmd, std::unique_lock<std::mutex>& lk) {
    uint32_t code = cmd >> 28;
    ++g_cmds[code < 10 ? code : 0];
    struct Timer {
        uint32_t k; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { g_cmd_secs[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    } timer{code < 10 ? code : 0};
    g_ctrl &= ~(1u << 14);                           // ECD
    uint64_t at = g_consumed;
    struct Log {
        uint32_t cmd; uint64_t at;
        ~Log() {
            if (g_ipu_trace)
                std::fprintf(stderr, "[ipu] %08X at bit %llu: %llu bits, DATA %08X, CTRL %08X\n", cmd,
                             (unsigned long long)at, (unsigned long long)(g_consumed - at), g_data, g_ctrl);
        }
    } log{cmd, at};
    uint64_t logfrom = g_logpos;
    struct Rec {
        uint32_t cmd; uint64_t from;
        ~Rec() {
            if (!g_bitlog && !g_cmdlog) return;
            uint32_t have = std::min<uint32_t>(32, avail_bits()), v = 0;
            for (uint32_t k = 0; k < have; ++k) v = v << 1 | (uint32_t)bit_at(k);
            Recent r{cmd, g_data, g_ctrl, have ? v << (32 - have) : 0, from, g_logpos};
            if (g_cmdlog) std::fwrite(&r, sizeof r, 1, g_cmdlog);
            if (g_bitlog && !(g_logpos == from && (cmd >> 28) != 3 && (cmd >> 28) != 4))
                g_recent[g_nrecent++ % 4096] = r;
        }
    } rec{cmd, logfrom};
    g_logging = false;
    switch (code) {
    case 0:                                          // BCLR
        g_consumed = 0;
        g_in.clear();
        g_bp = cmd & 0x7F;
        fill();
        break;
    case 2: g_logging = true; do_bdec(cmd); g_logging = false; break;
    case 3: g_logging = true; g_data = do_vdec(cmd); g_logging = false; break;
    case 4:                                          // FDEC
        g_logging = true;
        skip(cmd & 0x3F);
        g_data = peek(32);
        g_logging = false;
        break;
    case 5: {                                        // SETIQ
        g_logging = ((hw_dma_reg(4, 0) >> 2) & 3) == 1;   // from the stream, not a default matrix
        skip(cmd & 0x3F);
        uint8_t* m = g_iq[(cmd >> 27) & 1];
        for (int k = 0; k < 64; ++k) m[ZIGZAG[k]] = (uint8_t)get(8);
        g_logging = false;
        break;
    }
    case 6:                                          // SETVQ
        skip(cmd & 0x3F);
        for (int k = 0; k < 16; ++k) { uint32_t lo = get(8); g_vq[k] = (uint16_t)(lo | get(8) << 8); }
        break;
    case 7: do_csc(cmd, lk); break;
    case 9:                                          // SETTH
        g_th0 = cmd & 0x1FF;
        g_th1 = (cmd >> 16) & 0x1FF;
        break;
    default:
        rt_die("IPU: command %X (%08X) is not modelled yet", code, cmd);
    }
}

void ipu_main() {
    std::unique_lock<std::mutex> lk(*g_lock);
    g_lk = &lk;
    for (;;) {
        while (!g_busy) {
            if (g_wake) { g_wake = false; lk.unlock(); kernel_wake(); lk.lock(); }
            if (!g_busy) g_cv.wait(lk);
        }
        g_running = true;
        g_run_gen = g_reset_gen;
        try {
            run(g_cmd, lk);
            g_busy = false;
        } catch (const Abort&) {
        }
        g_running = false;
    }
}

}  // namespace

// ---- the interface, called with the hardware's lock held ------------------------------------------
void ipu_init(std::mutex& hw_lock) {
    tables();
    for (int u = 0; u < 8; ++u)
        for (int x = 0; x < 8; ++x)
            g_cos[u][x] = (u ? 0.5 : 0.5 / std::sqrt(2.0)) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16);
    g_lock = &hw_lock;
    g_ipu_trace = std::getenv("PS2KIT_IPU_TRACE") != nullptr;
    if (const char* f = std::getenv("PS2KIT_IPU_CMDLOG"))
        if (!(g_cmdlog = std::fopen(f, "wb"))) rt_die("IPU: cannot write %s", f);
    if (const char* f = std::getenv("PS2KIT_IPU_BITLOG"))
        if (!(g_bitlog = std::fopen(f, "wb"))) rt_die("IPU: cannot write %s", f);
    if (const char* d = std::getenv("PS2KIT_IPU_DUMP")) {
        g_dump_in = std::fopen((std::string(d) + "/csc_in.raw").c_str(), "wb");
        g_dump_out = std::fopen((std::string(d) + "/csc_out.raw").c_str(), "wb");
        if (!g_dump_in || !g_dump_out) rt_die("IPU: cannot write the dumps in %s", d);
    }
    // A transcription check of the tables: each code's share of the code
    // space (2^-length, the sign bit not counted) must sum to 1 less the
    // prefixes Annex B leaves unused. MBAI: 0000 0000, 0000 0010 and
    // 0000 0001 other than 000 (escape); motion_code: 0000 000, 0000 0010;
    // coded_block_pattern: 0000 0000 0; macroblock_type: 00 (I), 0000 00
    // (P, B); DCT table zero: twelve zeros; table one, besides those: the six
    // 12-bit and four 13-bit codes of table zero it does not use.
    const struct { const char* name; const Table& t; double want; } kraft[] = {
        {"macroblock_address_increment", g_mbai, 1 - 2.0 / 256 - 7.0 / 2048},
        {"motion_code", g_mv, 1 - 1.0 / 128 - 1.0 / 256},
        {"dmvector", g_dmv, 1},
        {"coded_block_pattern", g_cbp, 1 - 1.0 / 512},
        {"macroblock_type I", g_mbt[1], 0.75},
        {"macroblock_type P", g_mbt[2], 1 - 1.0 / 64},
        {"macroblock_type B", g_mbt[3], 1 - 1.0 / 64},
        {"dct_dc_size_luminance", g_dcl, 1},
        {"dct_dc_size_chrominance", g_dcc, 1},
        {"DCT coefficients, table zero", g_dct0, 1 - 1.0 / 4096},
        {"DCT coefficients, table one", g_dct1, 1 - 7.0 / 4096 - 4.0 / 8192},
    };
    for (auto& k : kraft)
        if (std::fabs(k.t.kraft - k.want) > 1e-12)
            rt_die("IPU: the %s table covers %.9f of the code space, not %.9f", k.name, k.t.kraft, k.want);
    std::thread(ipu_main).detach();
}

uint64_t ipu_read_locked(uint32_t a) {
    switch (a) {
    case 0x10002000: return g_data | (uint64_t)g_busy << 63;
    case 0x10002010: {
        uint32_t fp = (uint32_t)std::min<size_t>(g_in.size(), 2), ifc = (uint32_t)g_in.size() - fp;
        return (g_ctrl & 0x07F7FF00u) | ifc | (uint32_t)std::min<size_t>(g_out.size(), 15) << 4 |
               (uint32_t)g_busy << 31;
    }
    case 0x10002020: {
        fill();
        uint32_t fp = (uint32_t)std::min<size_t>(g_in.size(), 2), ifc = (uint32_t)g_in.size() - fp;
        return g_bp | ifc << 8 | fp << 16;
    }
    case 0x10002030: {
        fill();
        uint32_t have = std::min<uint32_t>(32, avail_bits()), v = 0;
        for (uint32_t k = 0; k < have; ++k) v = v << 1 | (uint32_t)bit_at(k);
        v <<= 32 - have;
        return v | (uint64_t)(have < 32) << 63;
    }
    case 0x10007000: {                               // the output FIFO
        if (g_out.empty()) return 0;
        QW q = g_out.front();
        g_out.pop_front();
        uint64_t lo;
        std::memcpy(&lo, q.b, 8);
        std::memcpy(&g_out_hi, q.b + 8, 8);
        g_cv.notify_all();
        return lo;
    }
    case 0x10007008: return g_out_hi;
    }
    return 0;
}

void ipu_write_locked(uint32_t a, uint64_t v) {
    if (a == 0x10002000) {
        if (g_busy) rt_die("IPU: command %08X written while %08X runs", (uint32_t)v, g_cmd);
        g_cmd = (uint32_t)v;
        TRACE(2, "[ipu] command %08X\n", g_cmd);
        // A short command whose bits are all in the FIFO already runs here,
        // in the writer's thread: it cannot wait, and libmpeg issues hundreds
        // of thousands of FDECs a movie, each followed by a wait for BUSY to
        // clear (a hand-over to the IPU's thread costs far more than the
        // command). The upper bounds: FDEC and VDEC read at most FBP + 32
        // bits (no code is longer than 17), SETIQ FBP + 512, SETVQ FBP + 256.
        uint32_t code = g_cmd >> 28, fbp = g_cmd & 0x3F;
        int64_t need = code == 0 || code == 9 ? 0 : code == 3 || code == 4 ? fbp + 32
                     : code == 5 ? fbp + 512 : code == 6 ? fbp + 256 : -1;
        if (need >= 0) {
            fill();
            if (avail_bits() >= (uint64_t)need) {
                g_run_gen = g_reset_gen;
                std::unique_lock<std::mutex> none;           // not used: these commands never wait
                run(g_cmd, none);
                return;
            }
        }
        g_busy = true;
        g_cv.notify_all();
    } else if (a == 0x10002010) {
        if (v & 0x40000000u) {                       // RST
            ++g_reset_gen;
            g_busy = false;
            g_in.clear();
            g_out.clear();
            g_bp = 0;
            g_ctrl = 0;
            g_cv.notify_all();
        }
        g_ctrl = (g_ctrl & ~0x07F30000u) | ((uint32_t)v & 0x07F30000u);   // IDP, AS, IVF, QST, MP1, PCT
    } else {
        rt_die("IPU: write to %08X is not modelled", a);
    }
}

void ipu_fifo_write_locked(const uint8_t q[16]) {
    if (g_in.size() >= 10) rt_die("IPU: input FIFO written while full");
    QW w;
    std::memcpy(w.b, q, 16);
    g_in.push_back(w);
    g_cv.notify_all();
}

void ipu_dma_locked(int ch) {
    if (ch == 4) {
        g_ch4_fresh = hw_dma_reg(4, 0x20) == 0;
        fill();
    } else {
        drain();
    }
    g_cv.notify_all();
}

bool ipu_take_wake() { bool w = g_wake; g_wake = false; return w; }

std::string ipu_status() {
    static const char* names[10] = {"BCLR", "IDEC", "BDEC", "VDEC", "FDEC", "SETIQ", "SETVQ", "CSC", "PACK", "SETTH"};
    std::string s;
    for (int k = 0; k < 10; ++k)
        if (g_cmds[k]) {
            char t[64];
            std::snprintf(t, sizeof t, "%s%s %llu (%.1f s)", s.empty() ? "" : " ", names[k],
                          (unsigned long long)g_cmds[k], g_cmd_secs[k]);
            s += t;
        }
    return s;
}

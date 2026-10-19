// ps2kit runtime — VU0 in micro mode, and VIF0 in front of it.
//
// VIF0 takes the codes DMA channel 0 (or the VIF0 FIFO) brings: NOP, STCYCL,
// STMASK, STMOD, STROW, STCOL, MARK, ITOP, FLUSHE are kept or ignored (VU0
// is idle whenever VIF0 runs, see below); MPG writes microcode into VU0's
// 4 KB micro memory. Anything else stops the run until a game needs it.
//
// A micro program (vcallms, vcallmsr) runs at once, to its end, on the
// calling thread's VU0 registers (the vf/vi/acc of EEContext), in the
// interpreter both units share (vu.cpp): VU0 is infinitely fast here, so the
// EE's interlocked reads (qmfc2.i, cfc2.i) and the next vcallms always find
// it finished. DW2's trigonometry library waits exactly 4 pairs after CLIP
// and FMAC and 7 after DIV, so it reads its flags and Q once they are written.
#include "vu.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

uint8_t g_micro[0x1000];                 // VU0 micro memory: 512 instruction pairs
VUUnit g_vu0{0, nullptr, g_micro, 0xFFF, nullptr, 0xFFF};

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
            ++g_vu0.micro_gen;
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

}  // namespace

// ---- the interface ---------------------------------------------------------------------------
void vif0_data(const uint8_t* p, size_t n) {
    g_vif0.fifo.insert(g_vif0.fifo.end(), p, p + n);
    vif0_run();
}

// PS2KIT_VU0_LOG=N: the first N runs from each start address are printed,
// with vf1-vf2 (and vf16-vf19 when they change) before and after, as data
// for checking the programs against what they compute.
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
    g_vu0.mem = g_vu0_mem;
    static const int log_n = std::getenv("PS2KIT_VU0_LOG") ? std::atoi(std::getenv("PS2KIT_VU0_LOG")) : 0;
    static std::map<uint32_t, int> logged;
    bool log = log_n > 0 && logged[start & 0xFFF]++ < log_n;
    if (log) vu0_log("in ", start & 0xFFF, c);
    g_vu0.c = &c;
    vu_exec(g_vu0, start);
    if (log) vu0_log("out", start & 0xFFF, c);
}

std::string vu0_status() {
    if (!g_vu0.runs && !g_vif0.mpg_bytes) return "";
    return std::to_string(g_vif0.mpg_bytes / 8) + " microcode pairs, " + std::to_string(g_vu0.runs) + " runs, " +
           std::to_string(g_vu0.pairs) + " pairs";
}

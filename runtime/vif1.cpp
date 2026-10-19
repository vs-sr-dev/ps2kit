// ps2kit runtime — VIF1, in front of VU1 and GIF path 2.
//
// The codes DMA channel 1 brings are walked and counted by kind (with
// their data sizes): the census says what VU1 will have to do. DIRECT and
// DIRECTHL hand their data to GIF path 2 (with no path 3 transfer ever in
// progress at the same time here, the two are the same). The rest is only
// counted for now. MPG and DIRECT data are counted when they do not start
// on a 64- or 128-bit boundary of the stream (DW2 never does that).
#include "os.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> g_fifo;
uint64_t g_base;                                     // stream position of g_fifo[0]
uint64_t g_misaligned, g_tte;
std::map<uint32_t, uint64_t> g_count;                // command (UNPACK: its format) -> codes
uint64_t g_unpack_bytes, g_mpg_bytes, g_direct_bytes;

const char* name_of(uint32_t cmd) {
    switch (cmd) {
    case 0x00: return "NOP"; case 0x01: return "STCYCL"; case 0x02: return "OFFSET"; case 0x03: return "BASE";
    case 0x04: return "ITOP"; case 0x05: return "STMOD"; case 0x06: return "MSKPATH3"; case 0x07: return "MARK";
    case 0x10: return "FLUSHE"; case 0x11: return "FLUSH"; case 0x13: return "FLUSHA"; case 0x14: return "MSCAL";
    case 0x15: return "MSCALF"; case 0x17: return "MSCNT"; case 0x20: return "STMASK"; case 0x30: return "STROW";
    case 0x31: return "STCOL"; case 0x4A: return "MPG"; case 0x50: return "DIRECT"; case 0x51: return "DIRECTHL";
    default: return nullptr;
    }
}

// The size of an UNPACK's data: num vectors of vn components of vl bits,
// in whole words; with write masking (CL < WL) num counts written vectors,
// and the VIF reads only the filled ones (CL of every WL).
size_t unpack_bytes(uint32_t cmd, uint32_t num, uint32_t cycle) {
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    uint32_t n = num ? num : 256;
    uint32_t cl = cycle & 0xFF, wl = (cycle >> 8) & 0xFF;
    if (wl > cl && cl) n = (n / wl) * cl + std::min(n % wl, cl);
    uint32_t bits = vl == 3 ? 16 : (32 >> vl) * (vn + 1);
    return ((size_t)n * bits + 31) / 32 * 4;
}

uint32_t g_cycle;

void run() {
    size_t p = 0;
    while (p + 4 <= g_fifo.size()) {
        uint32_t code;
        std::memcpy(&code, &g_fifo[p], 4);
        uint32_t cmd = (code >> 24) & 0x7F, num = (code >> 16) & 0xFF, imm = code & 0xFFFF;
        size_t data = 0, align = 0;
        if (cmd >= 0x60) data = unpack_bytes(cmd, num, g_cycle);
        else if (cmd == 0x20) data = 4;
        else if (cmd == 0x30 || cmd == 0x31) data = 16;
        else if (cmd == 0x4A) { data = (size_t)(num ? num : 256) * 8; align = 8; }
        else if (cmd == 0x50 || cmd == 0x51) { data = (size_t)(imm ? imm : 65536) * 16; align = 16; }
        else if (!name_of(cmd)) rt_die("VIF1: code %08X (command %02X) is not one", code, cmd);
        size_t start = p + 4;
        if (start + data > g_fifo.size()) break;
        if (align && (g_base + start) % align) ++g_misaligned;
        if (cmd == 0x01) g_cycle = imm;
        ++g_count[cmd >= 0x60 ? (0x60 | (cmd & 0x0F)) : cmd];
        if (cmd >= 0x60) g_unpack_bytes += data;
        else if (cmd == 0x4A) g_mpg_bytes += data;
        else if (cmd == 0x50 || cmd == 0x51) {
            g_direct_bytes += data;
            gs_gif(2, g_fifo.data() + start, (uint32_t)(data / 16));
        }
        p = start + data;
    }
    g_fifo.erase(g_fifo.begin(), g_fifo.begin() + (std::ptrdiff_t)p);
    g_base += p;
}

}  // namespace

void vif1_data(const uint8_t* p, size_t n, bool tag) {
    if (tag) {                                       // a DMA tag's upper half: offset 8 of its quadword
        ++g_tte;
        if (g_fifo.empty()) g_base = (g_base + 15) / 16 * 16 + 8;
    } else if (g_fifo.empty()) {
        g_base = (g_base + 15) / 16 * 16;
    }
    g_fifo.insert(g_fifo.end(), p, p + n);
    run();
}

std::string vif1_status() {
    if (g_count.empty()) return "";
    static const char* const FMT[16] = {"S-32", "S-16", "S-8", "?3", "V2-32", "V2-16", "V2-8", "?7",
                                        "V3-32", "V3-16", "V3-8", "?B", "V4-32", "V4-16", "V4-8", "V4-5"};
    std::string s;
    for (auto& e : g_count) {
        char b[64];
        if (e.first >= 0x60) std::snprintf(b, sizeof b, " UNPACK %s %llu", FMT[e.first & 15], (unsigned long long)e.second);
        else std::snprintf(b, sizeof b, " %s %llu", name_of(e.first), (unsigned long long)e.second);
        s += b;
    }
    return s + "; bytes: UNPACK " + std::to_string(g_unpack_bytes) + " MPG " + std::to_string(g_mpg_bytes) +
           " DIRECT " + std::to_string(g_direct_bytes) + "; misaligned " + std::to_string(g_misaligned) + ", TTE " +
           std::to_string(g_tte);
}

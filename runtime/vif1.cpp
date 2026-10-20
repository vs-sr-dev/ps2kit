// ps2kit runtime — VIF1, VU1 behind it, and GIF paths 1 and 2.
//
// The codes DMA channel 1 brings are carried out in order:
// * STCYCL, STMASK, STMOD, STROW, STCOL, MARK, ITOP, OFFSET, BASE set VIF1's
//   registers; NOP, the FLUSHes and MSKPATH3 do nothing here (VU1 and the
//   GIF are never busy when VIF1 looks: see below).
// * MPG writes microcode into VU1's 16 KB micro memory.
// * UNPACK writes vectors into VU1's 16 KB data memory: num vectors of the
//   format (S, V2, V3, V4 of 32, 16 or 8 bits; V4-5), signed or not (USN),
//   at the code's address (plus TOPS with FLG), with STCYCL's write cycle:
//   CL >= WL writes WL vectors and skips CL - WL quadwords, CL < WL reads
//   CL vectors of every WL and fills the others. Each lane is written as
//   STMASK says for the vector's place in the cycle (the fourth mask row for
//   every place from the fourth on): the data (with STMOD: as is, plus ROW,
//   or added into ROW), ROW, COL, or nothing. The lanes a format leaves
//   undefined are PCSX2's: V2 writes x y x y, V3 takes w from the element
//   after its z (0 past the code's data), V4-5 ignores STMOD; a filled
//   vector takes the data the next read vector will use.
// * MSCAL, MSCALF and MSCNT start VU1 (at the code's address, or where its
//   last program ended): TOP becomes TOPS and ITOP becomes ITOPS, and the
//   double buffer flips (TOPS is BASE + OFFSET after a start with DBF clear,
//   BASE after one with DBF set; OFFSET clears DBF and sets TOPS to BASE).
//   The program runs at once, to its end (vu.cpp).
// * DIRECT and DIRECTHL hand their data to GIF path 2. VU1's XGKICK hands
//   the GIF packet at its address (to the tag with EOP, wrapping around the
//   data memory) to path 1. (With VU1 infinitely fast and no path 3
//   transfer ever in progress at the same time here, the order the paths
//   reach the GS is the order the guest started them.)
//
// VIF1 and VU1 work beside the EE, as on the console: what DMA channel 1
// brings is copied into a queue and carried out by a thread of their own,
// which hands paths 1 and 2 to the GS's queue; the channel ends when the GS
// has taken the last of it (the channel's end goes through both queues). So
// the guest's vblank handler, which sends each frame's display list, returns
// at once, where carrying out the list in it took longer than a field and
// starved the game's main thread. PS2KIT_VIF1_SYNC=1 (or PS2KIT_GS_SYNC=1)
// carries the codes out at once, on the caller's thread.
//
// The codes are also counted by kind (with their data sizes), for the
// heartbeat. A word that is not a VIF code means the walk has lost its place
// (or the game sent something not understood yet): the codes before it are
// printed, the data waiting is dropped, and the walk starts again with the
// next transfer.
#include "vu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<uint8_t> g_fifo;
uint64_t g_base;                                     // stream position of g_fifo[0]
std::atomic<uint32_t> g_epoch{0};                    // VIF1_FBRST resets so far
uint32_t g_cur_epoch;                                // the epoch of the data being carried out
bool aborted() { return g_cur_epoch != g_epoch.load(std::memory_order_relaxed); }
uint64_t g_misaligned, g_tte;
uint64_t g_count[128];                               // codes by command (UNPACK: 0x60 + its format)
uint64_t g_unpack_bytes, g_mpg_bytes, g_direct_bytes, g_lost, g_kicks, g_kick_qw;

struct Seen { uint64_t at; uint32_t code; size_t data; };
Seen g_last[16];                                     // the last codes walked, for the report above
unsigned g_nlast;

// VIF1's registers.
struct Vif1 {
    uint32_t cycle = 0, mask = 0, mode = 0, row[4] = {}, col[4] = {}, mark = 0;
    uint32_t base = 0, ofst = 0, tops = 0, top = 0, itops = 0, itop = 0;
    bool dbf = false;
} g_vif1;

uint8_t g_vu1_micro[0x4000];
uint8_t g_vu1_mem[0x4000 + 16];
EEContext g_vu1_regs;
std::vector<uint8_t> g_kick_buf;

void xgkick(uint32_t qaddr);
uint32_t xtop() { return g_vif1.top; }
uint32_t xitop() { return g_vif1.itop; }
VUUnit g_vu1{1, &g_vu1_regs, g_vu1_micro, 0x3FFF, g_vu1_mem, 0x3FFF, 0, xgkick, xtop, xitop};

// XGKICK: the GIF packet at qaddr, its tags followed to the one with EOP.
void xgkick(uint32_t qaddr) {
    g_kick_buf.clear();
    uint32_t a = (qaddr * 16) & 0x3FF0;
    for (int guard = 0;; ++guard) {
        if (guard > 1024) rt_die("VU1: XGKICK at %04X: no EOP in 1024 GIF tags", qaddr * 16);
        uint64_t tag[2];
        std::memcpy(tag, g_vu1_mem + a, 16);
        uint32_t nloop = tag[0] & 0x7FFF, eop = (tag[0] >> 15) & 1, flg = (tag[0] >> 58) & 3;
        uint32_t nreg = (tag[0] >> 60) & 15;
        if (!nreg) nreg = 16;
        uint32_t qw = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        for (uint32_t k = 0; k <= qw; ++k) {
            g_kick_buf.insert(g_kick_buf.end(), g_vu1_mem + a, g_vu1_mem + a + 16);
            a = (a + 16) & 0x3FF0;
        }
        if (eop) break;
    }
    ++g_kicks;
    g_kick_qw += g_kick_buf.size() / 16;
    gs_gif(1, g_kick_buf.data(), (uint32_t)(g_kick_buf.size() / 16));
}

// PS2KIT_VU1_DUMP=dir: VU1's micro memory, written to dir/micro_N.bin at the
// first start after each change (up to 64 files), with the start addresses
// in dir/starts.txt: for the disassembler (ps2kit.vu). At the first start
// from each address (with each micro memory), the data memory and the
// registers as the program finds them: dir/mem_N_ADDR.bin, dir/regs_N_ADDR.txt.
const char* g_dump_dir;
bool g_micro_dirty = true;
int g_dumps;
FILE* g_starts;

void vu1_start(uint32_t pc) {
    Vif1& v = g_vif1;
    v.top = v.tops & 0x3FF;
    v.itop = v.itops & 0x3FF;
    if (v.dbf) { v.tops = v.base; v.dbf = false; }
    else { v.tops = (v.base + v.ofst) & 0x3FF; v.dbf = true; }
    if (g_dump_dir) {
        if (g_micro_dirty && g_dumps < 64) {
            std::string f = std::string(g_dump_dir) + "/micro_" + std::to_string(g_dumps++) + ".bin";
            if (FILE* o = std::fopen(f.c_str(), "wb")) { std::fwrite(g_vu1_micro, 1, sizeof g_vu1_micro, o); std::fclose(o); }
            g_micro_dirty = false;
        }
        if (!g_starts) g_starts = std::fopen((std::string(g_dump_dir) + "/starts.txt").c_str(), "w");
        if (g_starts) { std::fprintf(g_starts, "%d %04X\n", g_dumps - 1, pc); std::fflush(g_starts); }
        static std::set<uint32_t> seen;
        if (seen.insert((uint32_t)g_dumps << 16 | pc).second) {
            char name[64];
            std::snprintf(name, sizeof name, "/mem_%d_%04X.bin", g_dumps - 1, pc);
            if (FILE* o = std::fopen((std::string(g_dump_dir) + name).c_str(), "wb")) {
                std::fwrite(g_vu1_mem, 1, 0x4000, o);
                std::fclose(o);
            }
            std::snprintf(name, sizeof name, "/regs_%d_%04X.txt", g_dumps - 1, pc);
            if (FILE* o = std::fopen((std::string(g_dump_dir) + name).c_str(), "w")) {
                const EEContext& c = g_vu1_regs;
                for (int r = 0; r < 32; ++r)
                    std::fprintf(o, "vf%-2d %.9g %.9g %.9g %.9g\n", r, c.vf[r].f[0], c.vf[r].f[1], c.vf[r].f[2], c.vf[r].f[3]);
                for (int r = 0; r < 16; ++r) std::fprintf(o, "vi%-2d %04X\n", r, c.vi[r]);
                std::fprintf(o, "TOP %03X ITOP %03X TOPS %03X BASE %03X OFFSET %03X\n", g_vif1.top, g_vif1.itop, g_vif1.tops,
                             g_vif1.base, g_vif1.ofst);
                std::fclose(o);
            }
        }
    }
    static const bool off = std::getenv("PS2KIT_VU1_OFF") != nullptr;   // debugging: VU1 does nothing
    if (!off) vu_exec(g_vu1, pc);
}

// PS2KIT_VU1_VF=N:x,y,z,w[;N:x,y,z,w...]: VF registers VU1 starts with,
// instead of zeros: for experiments with a register a game reads before
// any program of its own writes it. (DW2's vf30, once thought to be such a
// register, is written by the EE through VU0: ee_vu0_vu1reg.)
void preset_registers() {
    const char* spec = std::getenv("PS2KIT_VU1_VF");
    for (const char* s = spec; s && *s;) {
        int r;
        float v[4];
        int used = 0;
        if (std::sscanf(s, "%d:%f,%f,%f,%f%n", &r, &v[0], &v[1], &v[2], &v[3], &used) != 5 || r < 1 || r > 31) {
            std::fprintf(stderr, "ps2kit: PS2KIT_VU1_VF: cannot read %s\n", s);
            return;
        }
        std::memcpy(g_vu1_regs.vf[r].f, v, 16);
        s += used;
        if (*s == ';') ++s;
    }
}

// ---- UNPACK -----------------------------------------------------------------------------------
uint32_t g_wl() { uint32_t wl = (g_vif1.cycle >> 8) & 0xFF; return wl ? wl : 256; }

// The size of an UNPACK's data: the vectors read (with CL < WL, CL of every
// WL), in whole words.
size_t unpack_bytes(uint32_t cmd, uint32_t num) {
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    uint32_t n = num ? num : 256;
    uint32_t cl = g_vif1.cycle & 0xFF, wl = g_wl();
    if (cl < wl) n = (n / wl) * cl + std::min(n % wl, cl);
    uint32_t bits = vl == 3 ? 16 : (32 >> vl) * (vn + 1);
    return ((size_t)n * bits + 31) / 32 * 4;
}

std::set<uint32_t> g_said;
void say_once(uint32_t key, const char* fmt, uint32_t a) {
    if (g_said.insert(key).second) { std::fprintf(stderr, "ps2kit: VIF1: "); std::fprintf(stderr, fmt, a); std::fprintf(stderr, "\n"); }
}

void unpack(uint32_t code, const uint8_t* d, size_t bytes) {
    Vif1& v = g_vif1;
    uint32_t cmd = (code >> 24) & 0x7F, imm = code & 0xFFFF;
    uint32_t n = (code >> 16) & 0xFF;
    if (!n) n = 256;
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    bool masked = cmd & 0x10, usn = imm & 0x4000;
    if (vl == 3 && vn != 3) rt_die("VIF1: UNPACK %08X: no such format", code);
    if (v.mode == 3) rt_die("VIF1: UNPACK with STMOD 3 (not a documented mode)");
    uint32_t addr = (imm & 0x3FF) + ((imm & 0x8000) ? v.tops : 0);
    uint32_t cl = v.cycle & 0xFF, wl = g_wl();
    bool fill = cl < wl;
    size_t esize = vl == 3 ? 2 : (size_t)(4 >> vl);  // bytes per element
    size_t vsize = vl == 3 ? 2 : esize * (vn + 1);   // bytes per vector read
    size_t sp = 0;
    uint32_t pos = 0;
    auto elem = [&](size_t at) -> uint32_t {
        if (at + esize > bytes) return 0;
        if (vl == 0) { uint32_t x; std::memcpy(&x, d + at, 4); return x; }
        if (vl == 1) { uint16_t x; std::memcpy(&x, d + at, 2); return usn ? x : (uint32_t)(int32_t)(int16_t)x; }
        uint8_t x = d[at];
        return usn ? x : (uint32_t)(int32_t)(int8_t)x;
    };
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t e[4];
        if (vl == 3) {                                       // V4-5: RGBA 5551
            uint16_t x = 0;
            if (sp + 2 <= bytes) std::memcpy(&x, d + sp, 2);
            e[0] = (x & 0x1F) << 3; e[1] = (x >> 5 & 0x1F) << 3; e[2] = (x >> 10 & 0x1F) << 3; e[3] = (x >> 15 & 1) << 7;
        } else {
            switch (vn) {
            case 0: e[0] = e[1] = e[2] = e[3] = elem(sp); break;
            case 1: e[0] = e[2] = elem(sp); e[1] = e[3] = elem(sp + esize); break;
            default: for (int k = 0; k < 4; ++k) e[k] = elem(sp + k * esize); break;   // V3: w from the next element
            }
        }
        uint32_t rowi = std::min(pos, 3u);
        VUReg* q = (VUReg*)(g_vu1_mem + ((addr & 0x3FF) * 16));
        for (int k = 0; k < 4; ++k) {
            uint32_t m = masked ? (v.mask >> ((rowi * 4 + k) * 2)) & 3 : 0;
            switch (m) {
            case 0:
                if (fill && pos >= cl) say_once(1, "a filled vector takes data (%u)", (unsigned)pos);
                if (vl == 3 || v.mode == 0) q->u[k] = e[k];
                else if (v.mode == 1) q->u[k] = e[k] + v.row[k];
                else q->u[k] = v.row[k] = v.row[k] + e[k];
                break;
            case 1: q->u[k] = v.row[k]; break;
            case 2: q->u[k] = v.col[rowi]; break;
            case 3: break;
            }
        }
        ++addr;
        ++pos;
        if (fill) {
            if (pos <= cl) sp += vsize;
            else if (pos == wl) pos = 0;
        } else {
            sp += vsize;
            if (pos >= wl) { addr += cl - wl; pos = 0; }
        }
    }
}

// ---- the walk -----------------------------------------------------------------------------------
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

void run() {
    Vif1& v = g_vif1;
    size_t p = 0;
    while (p + 4 <= g_fifo.size()) {
        if (aborted()) { g_fifo.clear(); return; }
        uint32_t code;
        std::memcpy(&code, &g_fifo[p], 4);
        uint32_t cmd = (code >> 24) & 0x7F, num = (code >> 16) & 0xFF, imm = code & 0xFFFF;
        size_t data = 0, align = 0;
        if (cmd >= 0x60) data = unpack_bytes(cmd, num);
        else if (cmd == 0x20) data = 4;
        else if (cmd == 0x30 || cmd == 0x31) data = 16;
        else if (cmd == 0x4A) { data = (size_t)(num ? num : 256) * 8; align = 8; }
        else if (cmd == 0x50 || cmd == 0x51) { data = (size_t)(imm ? imm : 65536) * 16; align = 16; }
        else if (!name_of(cmd)) {
            if (++g_lost <= 3) {
                std::fprintf(stderr, "ps2kit: VIF1: word %08X at stream offset %llX is not a code; before it (STCYCL %04X):\n",
                             code, (unsigned long long)(g_base + p), v.cycle);
                for (unsigned k = g_nlast > 16 ? g_nlast - 16 : 0; k < g_nlast; ++k) {
                    const Seen& e = g_last[k % 16];
                    std::fprintf(stderr, "  %llX: %08X, %zu bytes of data\n", (unsigned long long)e.at, e.code, e.data);
                }
                std::fprintf(stderr, "  the %zu bytes waiting are dropped\n", g_fifo.size() - p);
            }
            p = g_fifo.size();
            break;
        }
        size_t start = p + 4;
        if (start + data > g_fifo.size()) break;
        if (code & 0x80000000u) rt_die("VIF1: code %08X with its interrupt bit is not modelled yet", code);
        if (align && (g_base + start) % align) ++g_misaligned;
        g_last[g_nlast++ % 16] = Seen{g_base + p, code, data};
        ++g_count[cmd >= 0x60 ? (0x60 | (cmd & 0x0F)) : cmd];
        const uint8_t* d = g_fifo.data() + start;
        if (cmd >= 0x60) {
            g_unpack_bytes += data;
            unpack(code, d, data);
        } else {
            switch (cmd) {
            case 0x01: v.cycle = imm; break;                                  // STCYCL
            case 0x02: v.ofst = imm & 0x3FF; v.dbf = false; v.tops = v.base; break;   // OFFSET
            case 0x03: v.base = imm & 0x3FF; break;                           // BASE
            case 0x04: v.itops = imm & 0x3FF; break;                          // ITOP
            case 0x05: v.mode = imm & 3; break;                               // STMOD
            case 0x07: v.mark = imm; break;                                   // MARK
            case 0x14: case 0x15: vu1_start((imm * 8) & 0x3FFF); break;       // MSCAL, MSCALF
            case 0x17: vu1_start((g_vu1_regs.vi[VI_TPC] * 8) & 0x3FFF); break;   // MSCNT
            case 0x20: std::memcpy(&v.mask, d, 4); break;                     // STMASK
            case 0x30: std::memcpy(v.row, d, 16); break;                      // STROW
            case 0x31: std::memcpy(v.col, d, 16); break;                      // STCOL
            case 0x4A: {                                                      // MPG: num pairs at imm (in pairs)
                uint32_t at = (imm * 8) & 0x3FFF;
                for (size_t k = 0; k < data; k += 8) std::memcpy(g_vu1_micro + ((at + k) & 0x3FFF), d + k, 8);
                g_mpg_bytes += data;
                g_micro_dirty = true;
                ++g_vu1.micro_gen;
                break;
            }
            case 0x50: case 0x51:                                             // DIRECT, DIRECTHL
                g_direct_bytes += data;
                gs_gif(2, d, (uint32_t)(data / 16));
                break;
            default: break;                                                   // NOP, FLUSH*, MSKPATH3
            }
        }
        p = start + data;
    }
    g_fifo.erase(g_fifo.begin(), g_fifo.begin() + (std::ptrdiff_t)p);
    g_base += p;
}

void feed(const uint8_t* p, size_t n, bool tag) {
    static bool init = [] {
        g_vu1_regs.vf[0].f[3] = 1.0f;
        preset_registers();
        g_dump_dir = std::getenv("PS2KIT_VU1_DUMP");
        return true;
    }();
    (void)init;
    if (tag) {                                       // a DMA tag's upper half: offset 8 of its quadword
        ++g_tte;
        if (g_fifo.empty()) g_base = (g_base + 15) / 16 * 16 + 8;
    } else if (g_fifo.empty()) {
        g_base = (g_base + 15) / 16 * 16;
    }
    g_fifo.insert(g_fifo.end(), p, p + n);
    run();
}

// ---- the thread ------------------------------------------------------------------------------
struct Item {
    char kind;                                       // 'D' data, 'E' the channel's end
    bool tag;
    uint32_t token;
    std::vector<uint8_t> data;
    uint32_t epoch = 0, gif_epoch = 0;               // VIF1's and the GIF's when given
};
std::mutex g_q_mu;
std::condition_variable g_q_cv, g_idle_cv;
std::deque<Item> g_queue;
bool g_async, g_busy;
double g_busy_s;                                     // the thread's time carrying out codes (heartbeat)
void (*g_channel_end)(int, uint32_t);

void vif1_thread() {
    rt_init_fpu();
    for (;;) {
        Item it;
        {
            std::unique_lock<std::mutex> lk(g_q_mu);
            g_q_cv.wait(lk, [] { return !g_queue.empty(); });
            it = std::move(g_queue.front());
            g_queue.pop_front();
            g_busy = true;
        }
        auto t0 = std::chrono::steady_clock::now();
        if (it.kind == 'D') {
            if (it.epoch != g_cur_epoch) {           // the first data after a reset: nothing in progress
                g_fifo.clear();
                g_cur_epoch = it.epoch;
            }
            gs_submit_epoch(it.gif_epoch);
            if (!aborted()) feed(it.data.data(), it.data.size(), it.tag);
        } else if (gs_channel_end(1, it.token)) g_channel_end(1, it.token);
        g_busy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(g_q_mu);
            g_busy = false;
            if (g_queue.empty()) g_idle_cv.notify_all();
        }
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

// VU0 sees VU1's registers in its data memory from byte 0x4000: vf0-vf31,
// then vi0-vi15 a quadword each (0x4200). DW2's SetFogOff stores VU1's fog
// factor (vf30) from the EE with a macro-mode vsqi to 0x41E. The VI half is
// not modelled; vf0 stays (0, 0, 0, 1) whatever is stored to it.
VUReg* ee_vu0_vu1reg(uint32_t qaddr) {
    uint32_t r = qaddr & 0x3F;
    if (r >= 32) rt_die("VU0 access to VU1's vi%u (%04X) is not modelled yet", r - 32, (qaddr * 16) & 0xFFFF);
    if (r == 0) {
        static VUReg vf0;
        vf0.f[0] = vf0.f[1] = vf0.f[2] = 0.0f;
        vf0.f[3] = 1.0f;
        return &vf0;
    }
    return &g_vu1_regs.vf[r];
}

void vif1_start_thread(void (*channel_end)(int ch, uint32_t token)) {
    for (const char* e : {"PS2KIT_VIF1_SYNC", "PS2KIT_GS_SYNC"}) {
        const char* v = std::getenv(e);
        if (v && *v && *v != '0') return;
    }
    g_channel_end = channel_end;
    g_async = true;
    std::thread(vif1_thread).detach();
}

void vif1_data(const uint8_t* p, size_t n, bool tag) {
    if (!g_async) { feed(p, n, tag); return; }
    enqueue(Item{'D', tag, 0, std::vector<uint8_t>(p, p + n), g_epoch.load(), gs_gif_epoch()});
}

void vif1_reset() {
    if (!g_async) { g_fifo.clear(); return; }
    std::lock_guard<std::mutex> lk(g_q_mu);
    g_epoch.fetch_add(1);
    // the data goes; a channel's end stays (one the game stopped is ignored by its token)
    g_queue.erase(std::remove_if(g_queue.begin(), g_queue.end(), [](const Item& i) { return i.kind == 'D'; }),
                  g_queue.end());
    if (g_queue.empty() && !g_busy) g_idle_cv.notify_all();
}

bool vif1_channel_end(uint32_t token) {
    if (!g_async) return gs_channel_end(1, token);
    enqueue(Item{'E', false, token, {}});
    return false;
}

void vif1_drain() {
    if (!g_async) return;
    std::unique_lock<std::mutex> lk(g_q_mu);
    g_idle_cv.wait(lk, [] { return g_queue.empty() && !g_busy; });
}

std::string vif1_status() {
    bool any = false;
    for (uint64_t c : g_count) any |= c != 0;
    if (!any) return "";
    static const char* const FMT[16] = {"S-32", "S-16", "S-8", "?3", "V2-32", "V2-16", "V2-8", "?7",
                                        "V3-32", "V3-16", "V3-8", "?B", "V4-32", "V4-16", "V4-8", "V4-5"};
    std::string s;
    for (uint32_t k = 0; k < 128; ++k) {
        if (!g_count[k]) continue;
        char b[64];
        if (k >= 0x60) std::snprintf(b, sizeof b, " UNPACK %s %llu", FMT[k & 15], (unsigned long long)g_count[k]);
        else std::snprintf(b, sizeof b, " %s %llu", name_of(k), (unsigned long long)g_count[k]);
        s += b;
    }
    s += "; bytes: UNPACK " + std::to_string(g_unpack_bytes) + " MPG " + std::to_string(g_mpg_bytes) + " DIRECT " +
         std::to_string(g_direct_bytes) + "; misaligned " + std::to_string(g_misaligned) + ", TTE " +
         std::to_string(g_tte) + (g_lost ? ", lost its place " + std::to_string(g_lost) : "");
    if (g_vu1.runs) {
        s += "; VU1 " + std::to_string(g_vu1.runs) + " runs, " + std::to_string(g_vu1.pairs) + " pairs, " +
             std::to_string(g_vu1.cycles) + " cycles, XGKICK " + std::to_string(g_kicks) + " (" +
             std::to_string(g_kick_qw) + " qw)";
        if (g_async) s += ", VIF1 busy " + std::to_string((int)g_busy_s) + " s";
        if (g_vu1.old_q || g_vu1.old_p || g_vu1.old_flags)
            s += ", reads in flight: Q " + std::to_string(g_vu1.old_q) + " P " + std::to_string(g_vu1.old_p) +
                 " flags " + std::to_string(g_vu1.old_flags);
    }
    return s;
}

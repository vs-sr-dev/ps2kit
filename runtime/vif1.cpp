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
#include "vu_ops.h"


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
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace {

std::vector<uint8_t> g_fifo;
// The VIF1 thread's time in VU1 (XGKICKs included), in UNPACK, and in
// handing XGKICK packets to the GS (heartbeat): in the processor's time
// stamp counter, cheap to read (the system's clock at each UNPACK and each
// start of VU1 took some 4% of the thread), turned into seconds against the
// system's clock since the start.
uint64_t g_t_vu, g_t_unpack, g_t_kick;
struct TimeIt {
    uint64_t& acc;
    uint64_t t0 = __rdtsc();
    ~TimeIt() { acc += __rdtsc() - t0; }
};
const uint64_t g_tsc0 = __rdtsc();
const auto g_clock0 = std::chrono::steady_clock::now();
double tsc_seconds(uint64_t ticks) {
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_clock0).count();
    uint64_t t = __rdtsc() - g_tsc0;
    return t && s > 0 ? ticks * (s / t) : 0;
}
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
// Paths 1 and 2 not yet given to the GS (XGKICK packets, DIRECT data):
// given together, in order (one item, one lock and one wake-up for many),
// at the end of each batch of VIF1's data and every KICK_BATCH bytes, so
// that the GS draws while VU1 works. Given at the end of each DMA transfer,
// as before, they were some 10 000 items for one of DW2's battle frames.
std::vector<uint8_t> g_kick_buf;
std::vector<uint32_t> g_kick_segs;                   // quadwords << 2 | path (gs_gif_batch)
constexpr size_t KICK_BATCH = 32 * 1024;

void kick_put(int path, const uint8_t* p, size_t bytes) {
    g_kick_buf.insert(g_kick_buf.end(), p, p + bytes);
    uint32_t q = (uint32_t)(bytes / 16);
    if (!g_kick_segs.empty() && (g_kick_segs.back() & 3) == (uint32_t)path) g_kick_segs.back() += q << 2;
    else g_kick_segs.push_back(q << 2 | (uint32_t)path);
}

void kick_flush() {
    if (g_kick_segs.empty()) return;
    gs_gif_batch(std::move(g_kick_buf), std::move(g_kick_segs));
    g_kick_buf = std::vector<uint8_t>();
    g_kick_buf.reserve(KICK_BATCH + 4096);
    g_kick_segs = std::vector<uint32_t>();
}

void kick_drop() {
    g_kick_buf.clear();
    g_kick_segs.clear();
}

void xgkick(uint32_t qaddr);
uint32_t xtop() { return g_vif1.top; }
uint32_t xitop() { return g_vif1.itop; }
VUUnit g_vu1{1, &g_vu1_regs, g_vu1_micro, 0x3FFF, g_vu1_mem, 0x3FFF, 0, xgkick, xtop, xitop};

// XGKICK: the GIF packet at qaddr, its tags followed to the one with EOP.
void xgkick(uint32_t qaddr) {
    size_t before = g_kick_buf.size();
    uint32_t a = (qaddr * 16) & 0x3FF0;
    for (int guard = 0;; ++guard) {
        if (guard > 1024) rt_die("VU1: XGKICK at %04X: no EOP in 1024 GIF tags", qaddr * 16);
        uint64_t tag[2];
        std::memcpy(tag, g_vu1_mem + a, 16);
        uint32_t nloop = tag[0] & 0x7FFF, eop = (tag[0] >> 15) & 1, flg = (tag[0] >> 58) & 3;
        uint32_t nreg = (tag[0] >> 60) & 15;
        if (!nreg) nreg = 16;
        uint32_t qw = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        for (uint32_t left = qw + 1; left;) {        // the tag and its data, wrapping at the end
            uint32_t n = std::min(left, (0x4000 - a) / 16);
            kick_put(1, g_vu1_mem + a, (size_t)n * 16);
            a = (a + n * 16) & 0x3FF0;
            left -= n;
        }
        if (eop) break;
    }
    ++g_kicks;
    g_kick_qw += (g_kick_buf.size() - before) / 16;
    if (g_kick_buf.size() >= KICK_BATCH) {
        TimeIt ti{g_t_kick};
        kick_flush();
    }
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

// VU1 programs translated to C++ (vutrans; vu_ops.h), each used while micro
// memory holds what it was translated from, looked up again after every
// change. PS2KIT_VU1_INTERP=1: the interpreter for all.
uint64_t g_trans_runs;

void (*translated(uint32_t pc))(VUUnit&) {
    static const bool interp = [] {
        const char* e = std::getenv("PS2KIT_VU1_INTERP");
        return e && *e && *e != '0';
    }();
    if (interp || !g_vu1_ntrans) return nullptr;
    static uint64_t gen = ~0ull;
    static void (*fn[2048])(VUUnit&);
    static bool known[2048];
    if (gen != g_vu1.micro_gen) {
        std::memset(known, 0, sizeof known);
        gen = g_vu1.micro_gen;
    }
    uint32_t i = (pc & 0x3FFF) / 8;
    if (!known[i]) {
        known[i] = true;
        fn[i] = vu1_trans_find(g_vu1_micro, pc);
    }
    return fn[i];
}

// PS2KIT_VU1_CHECK=1: every translated start is run by the interpreter too,
// on a copy of VU1's state (its XGKICKs noted, not sent), then by the
// translation; data memory, registers, P, the XGKICKs and the counts must
// be the same. The first difference stops the game.
std::vector<uint32_t> g_check_kicks;
uint64_t g_checked;

void check_translation(void (*fn)(VUUnit&), uint32_t pc) {
    static std::vector<uint8_t> mem0, mem1;
    mem0.assign(g_vu1_mem, g_vu1_mem + 0x4000);
    EEContext regs0 = g_vu1_regs;
    VUUnit& u = g_vu1;
    uint32_t p0 = u.p;
    uint64_t c0[6] = {u.runs, u.pairs, u.cycles, u.old_q, u.old_p, u.old_flags};
    auto kick = u.xgkick;
    g_check_kicks.clear();
    u.xgkick = [](uint32_t q) { g_check_kicks.push_back(q); };
    vu_exec(u, pc);
    u.xgkick = kick;
    std::vector<uint32_t> kicks_i = g_check_kicks;
    mem1.assign(g_vu1_mem, g_vu1_mem + 0x4000);
    EEContext regs1 = g_vu1_regs;
    uint32_t p1 = u.p;
    uint64_t c1[6] = {u.runs - c0[0], u.pairs - c0[1], u.cycles - c0[2], u.old_q - c0[3], u.old_p - c0[4],
                      u.old_flags - c0[5]};
    std::memcpy(g_vu1_mem, mem0.data(), 0x4000);
    g_vu1_regs = regs0;
    u.p = p0;
    uint64_t c2[6] = {u.runs, u.pairs, u.cycles, u.old_q, u.old_p, u.old_flags};
    g_check_kicks.clear();
    u.xgkick = [](uint32_t q) { g_check_kicks.push_back(q); xgkick(q); };
    fn(u);
    u.xgkick = kick;
    uint64_t c3[6] = {u.runs - c2[0], u.pairs - c2[1], u.cycles - c2[2], u.old_q - c2[3], u.old_p - c2[4],
                      u.old_flags - c2[5]};
    const char* what = nullptr;
    if (std::memcmp(g_vu1_mem, mem1.data(), 0x4000)) what = "data memory";
    else if (std::memcmp(g_vu1_regs.vf, regs1.vf, sizeof regs1.vf) || std::memcmp(&g_vu1_regs.vacc, &regs1.vacc, 16))
        what = "VF registers or ACC";
    else if (std::memcmp(g_vu1_regs.vi, regs1.vi, sizeof regs1.vi)) what = "VI registers";
    else if (u.p != p1) what = "P";
    else if (g_check_kicks != kicks_i) what = "XGKICKs";
    else if (std::memcmp(c1, c3, sizeof c1)) what = "counts";
    if (what)
        rt_die("VU1: the translation of %04X differs from the interpreter (%s) after %llu checked starts", pc, what,
               (unsigned long long)g_checked);
    ++g_checked;
}

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
    if (off) return;
    TimeIt ti{g_t_vu};
    auto fn = translated(pc);
    if (!fn) { vu_exec(g_vu1, pc); return; }
    ++g_trans_runs;
    static const bool check = [] {
        const char* e = std::getenv("PS2KIT_VU1_CHECK");
        return e && *e && *e != '0';
    }();
    if (check) check_translation(fn, pc);
    else fn(g_vu1);
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

void unpack_generic(uint32_t code, const uint8_t* d, size_t bytes) {
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

// The common UNPACKs, without the mask, CL >= WL (no filling), not V4-5:
// the format known to the compiler, each vector whole. False when the
// UNPACK is not one of them (unpack_generic does it). PS2KIT_VIF1_CHECK=1
// runs both and compares.
template <int VN, int VL>
void unpack_plain(uint32_t n, uint32_t addr, uint32_t cl, uint32_t wl, bool usn, const uint8_t* d, size_t bytes) {
    Vif1& v = g_vif1;
    constexpr size_t esize = (size_t)4 >> VL, vsize = esize * (VN + 1);
    constexpr size_t need = esize * (VN == 0 ? 1 : VN == 1 ? 2 : 4);   // V3 reads w from the next element
    auto el = [&](size_t at) -> uint32_t {
        if (at + esize > bytes) return 0;
        if constexpr (VL == 0) { uint32_t x; std::memcpy(&x, d + at, 4); return x; }
        else if constexpr (VL == 1) { uint16_t x; std::memcpy(&x, d + at, 2); return usn ? x : (uint32_t)(int32_t)(int16_t)x; }
        else { uint8_t x = d[at]; return usn ? x : (uint32_t)(int32_t)(int8_t)x; }
    };
    uint32_t pos = 0;
    size_t sp = 0;
    uint32_t mode = v.mode;
    for (uint32_t i = 0; i < n; ++i, sp += vsize) {
        uint32_t e[4];
        if (VN == 3 && VL == 0 && sp + 16 <= bytes) std::memcpy(e, d + sp, 16);
        else if (VN == 0) e[0] = e[1] = e[2] = e[3] = el(sp);
        else if (VN == 1) { e[0] = e[2] = el(sp); e[1] = e[3] = el(sp + esize); }
        else for (int k = 0; k < 4; ++k) e[k] = el(sp + k * esize);
        (void)need;
        VUReg* q = (VUReg*)(g_vu1_mem + ((addr & 0x3FF) * 16));
        if (mode == 0) std::memcpy(q->u, e, 16);
        else if (mode == 1) for (int k = 0; k < 4; ++k) q->u[k] = e[k] + v.row[k];
        else for (int k = 0; k < 4; ++k) q->u[k] = v.row[k] = v.row[k] + e[k];
        ++addr;
        if (++pos >= wl) { addr += cl - wl; pos = 0; }
    }
}

bool unpack_fast(uint32_t code, const uint8_t* d, size_t bytes) {
    Vif1& v = g_vif1;
    uint32_t cmd = (code >> 24) & 0x7F, imm = code & 0xFFFF;
    uint32_t n = (code >> 16) & 0xFF;
    if (!n) n = 256;
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    uint32_t cl = v.cycle & 0xFF, wl = g_wl();
    if ((cmd & 0x10) || cl < wl || vl == 3 || v.mode == 3) return false;
    uint32_t addr = (imm & 0x3FF) + ((imm & 0x8000) ? v.tops : 0);
    bool usn = imm & 0x4000;
    switch (vn * 4 + vl) {
    case 0: unpack_plain<0, 0>(n, addr, cl, wl, usn, d, bytes); return true;
    case 1: unpack_plain<0, 1>(n, addr, cl, wl, usn, d, bytes); return true;
    case 2: unpack_plain<0, 2>(n, addr, cl, wl, usn, d, bytes); return true;
    case 4: unpack_plain<1, 0>(n, addr, cl, wl, usn, d, bytes); return true;
    case 5: unpack_plain<1, 1>(n, addr, cl, wl, usn, d, bytes); return true;
    case 6: unpack_plain<1, 2>(n, addr, cl, wl, usn, d, bytes); return true;
    case 8: unpack_plain<2, 0>(n, addr, cl, wl, usn, d, bytes); return true;
    case 9: unpack_plain<2, 1>(n, addr, cl, wl, usn, d, bytes); return true;
    case 10: unpack_plain<2, 2>(n, addr, cl, wl, usn, d, bytes); return true;
    case 12: unpack_plain<3, 0>(n, addr, cl, wl, usn, d, bytes); return true;
    case 13: unpack_plain<3, 1>(n, addr, cl, wl, usn, d, bytes); return true;
    case 14: unpack_plain<3, 2>(n, addr, cl, wl, usn, d, bytes); return true;
    default: return false;
    }
}

uint64_t g_unpack_checked;

void unpack(uint32_t code, const uint8_t* d, size_t bytes) {
    static const bool check = [] {
        const char* e = std::getenv("PS2KIT_VIF1_CHECK");
        return e && *e && *e != '0';
    }();
    if (!check) {
        if (!unpack_fast(code, d, bytes)) unpack_generic(code, d, bytes);
        return;
    }
    static uint8_t mem0[0x4000], mem1[0x4000];
    uint32_t row0[4], row1[4];
    std::memcpy(mem0, g_vu1_mem, 0x4000);
    std::memcpy(row0, g_vif1.row, 16);
    if (!unpack_fast(code, d, bytes)) { unpack_generic(code, d, bytes); return; }
    std::memcpy(mem1, g_vu1_mem, 0x4000);
    std::memcpy(row1, g_vif1.row, 16);
    std::memcpy(g_vu1_mem, mem0, 0x4000);
    std::memcpy(g_vif1.row, row0, 16);
    unpack_generic(code, d, bytes);
    if (std::memcmp(mem1, g_vu1_mem, 0x4000) || std::memcmp(row1, g_vif1.row, 16))
        rt_die("VIF1: UNPACK %08X (STMOD %u, STCYCL %04X): the fast form differs, after %llu checked", code, g_vif1.mode,
               g_vif1.cycle, (unsigned long long)g_unpack_checked);
    ++g_unpack_checked;
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

// The codes in buf (stream position g_base): the bytes used, those of
// whole codes with their data; the rest waits for more (g_fifo).
size_t run(const uint8_t* buf, size_t size) {
    Vif1& v = g_vif1;
    size_t p = 0;
    while (p + 4 <= size) {
        if (aborted()) { kick_drop(); return size; }
        uint32_t code;
        std::memcpy(&code, buf + p, 4);
        if (code == 0) { ++g_count[0]; p += 4; continue; }   // NOP
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
                std::fprintf(stderr, "  the %zu bytes waiting are dropped\n", size - p);
            }
            p = size;
            break;
        }
        size_t start = p + 4;
        if (start + data > size) break;
        if (code & 0x80000000u) rt_die("VIF1: code %08X with its interrupt bit is not modelled yet", code);
        if (align && (g_base + start) % align) ++g_misaligned;
        g_last[g_nlast++ % 16] = Seen{g_base + p, code, data};
        ++g_count[cmd >= 0x60 ? (0x60 | (cmd & 0x0F)) : cmd];
        const uint8_t* d = buf + start;
        if (cmd >= 0x60) {
            g_unpack_bytes += data;
            TimeIt ti{g_t_unpack};
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
                kick_put(2, d, data);
                if (g_kick_buf.size() >= KICK_BATCH) kick_flush();
                break;
            default: break;                                                   // NOP, FLUSH*, MSKPATH3
            }
        }
        p = start + data;
    }
    return p;
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
    // without bytes waiting, the codes are taken from the data in place
    if (g_fifo.empty()) {
        size_t used = run(p, n);
        if (aborted()) return;
        if (used < n) g_fifo.assign(p + used, p + n);
        g_base += used;
    } else {
        g_fifo.insert(g_fifo.end(), p, p + n);
        size_t used = run(g_fifo.data(), g_fifo.size());
        if (aborted()) { g_fifo.clear(); return; }
        g_fifo.erase(g_fifo.begin(), g_fifo.begin() + (std::ptrdiff_t)used);
        g_base += used;
    }
}

// ---- the thread ------------------------------------------------------------------------------
// A chain's transfers are given to the thread together: a battle's list in
// DW2 is some 25 000 tags, and one item each (an allocation, the queue's
// lock, the thread woken) kept VIF1 waiting more than working. The data of
// many transfers goes in one item, each transfer's size and kind in segs,
// in turn, until BATCH bytes or the channel's end.
struct Item {
    char kind;                                       // 'D' data, 'E' the channel's end
    uint32_t token = 0;
    std::vector<uint8_t> data;
    std::vector<uint32_t> segs;                      // 'D': each transfer's bytes << 1 | 1 for a DMA tag's half
    uint32_t epoch = 0, gif_epoch = 0;               // VIF1's and the GIF's when given
};
constexpr size_t BATCH = 64 * 1024;
Item g_batch{'D'};                                   // being filled (hardware's lock held)
std::mutex g_q_mu;
std::condition_variable g_q_cv, g_idle_cv;
std::deque<Item> g_queue;
bool g_async, g_busy;
double g_busy_s;                                     // the thread's time carrying out codes (heartbeat)
void (*g_channel_end)(int, uint32_t);

void vif1_thread() {
    rt_init_fpu();
    if (const char* f = std::getenv("PS2KIT_VIF1_PROFILE")) profile_this_thread(f);
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
            size_t at = 0;
            for (uint32_t s : it.segs) {
                if (aborted()) break;
                feed(it.data.data() + at, s >> 1, s & 1);
                at += s >> 1;
            }
            if (aborted()) kick_drop();
            else kick_flush();
        } else {
            static const bool tl = [] { const char* e = std::getenv("PS2KIT_TIMELINE"); return e && *e && *e != '0'; }();
            if (tl) std::fprintf(stderr, "vif1done %.4f\n", hw_now());   // (PS2KIT_TIMELINE, hw.cpp)
            if (gs_channel_end(1, it.token)) g_channel_end(1, it.token);
        }
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

void flush_batch() {
    if (g_batch.segs.empty()) return;
    enqueue(std::move(g_batch));
    g_batch = Item{'D'};
    g_batch.data.reserve(BATCH + 4096);
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
    if (!g_async) {
        feed(p, n, tag);
        kick_flush();
        return;
    }
    uint32_t epoch = g_epoch.load(), gif_epoch = gs_gif_epoch();
    if (!g_batch.segs.empty() && (g_batch.epoch != epoch || g_batch.gif_epoch != gif_epoch)) flush_batch();
    g_batch.epoch = epoch;
    g_batch.gif_epoch = gif_epoch;
    g_batch.data.insert(g_batch.data.end(), p, p + n);
    g_batch.segs.push_back((uint32_t)n << 1 | (tag ? 1u : 0u));
    if (g_batch.data.size() >= BATCH) flush_batch();
}

void vif1_reset() {
    if (!g_async) { g_fifo.clear(); return; }
    g_batch = Item{'D'};
    std::lock_guard<std::mutex> lk(g_q_mu);
    g_epoch.fetch_add(1);
    // the data goes; a channel's end stays (one the game stopped is ignored by its token)
    g_queue.erase(std::remove_if(g_queue.begin(), g_queue.end(), [](const Item& i) { return i.kind == 'D'; }),
                  g_queue.end());
    if (g_queue.empty() && !g_busy) g_idle_cv.notify_all();
}

bool vif1_channel_end(uint32_t token) {
    if (!g_async) return gs_channel_end(1, token);
    flush_batch();
    Item e{'E'};
    e.token = token;
    enqueue(std::move(e));
    return false;
}

void vif1_drain() {
    if (!g_async) return;
    flush_batch();
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
        char tb[160];
        std::snprintf(tb, sizeof tb, "; time in VU1 %.1f s (XGKICK %.1f s), UNPACK %.1f s", tsc_seconds(g_t_vu),
                      tsc_seconds(g_t_kick), tsc_seconds(g_t_unpack));
        s += tb;
        if (g_unpack_checked) s += "; " + std::to_string(g_unpack_checked) + " UNPACKs checked";
        s += "; VU1 " + std::to_string(g_vu1.runs) + " runs (" + std::to_string(g_trans_runs) + " translated" +
             (g_checked ? ", " + std::to_string(g_checked) + " checked" : std::string()) + "), " + std::to_string(g_vu1.pairs) + " pairs, " +
             std::to_string(g_vu1.cycles) + " cycles, XGKICK " + std::to_string(g_kicks) + " (" +
             std::to_string(g_kick_qw) + " qw)";
        if (g_async) s += ", VIF1 busy " + std::to_string((int)g_busy_s) + " s";
        if (g_vu1.old_q || g_vu1.old_p || g_vu1.old_flags)
            s += ", reads in flight: Q " + std::to_string(g_vu1.old_q) + " P " + std::to_string(g_vu1.old_p) +
                 " flags " + std::to_string(g_vu1.old_flags);
    }
    return s;
}

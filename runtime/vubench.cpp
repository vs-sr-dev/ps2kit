// ps2kit runtime — vubench: one VU1 micro program, as the game started it,
// run again without the game: to time the interpreter, and to check that a
// change to it leaves the program's results as they were.
//
//   vubench DIR N ADDR [--repeat R] [--out FILE] [--profile]
//
// DIR holds what PS2KIT_VU1_DUMP wrote: micro_N.bin (micro memory),
// mem_N_ADDR.bin (data memory) and regs_N_ADDR.txt (registers) at a start
// from ADDR (hex). Each repetition starts from the same state; XGKICK packets
// are counted, not drawn. --out writes the data memory and the registers after
// one run, to compare two builds; --profile samples the host's instruction
// pointer (Windows; image offsets for addr2line, as gsreplay --profile).
#include "vu.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

int g_trace = 0;
double g_run_seconds = 0;

// What the interpreter calls besides itself.
void gs_gif(int, const uint8_t*, uint32_t) {}

namespace {

uint8_t g_micro[0x4000], g_mem[0x4000 + 16], g_mem0[0x4000];
EEContext g_regs, g_regs0;
uint64_t g_kicks;
uint32_t g_top, g_itop;

bool read_file(const std::string& path, uint8_t* out, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fread(out, 1, n, f) == n;
    std::fclose(f);
    return ok;
}

bool read_regs(const std::string& path, EEContext& c) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        int r;
        float x, y, z, w;
        unsigned v, a, b, t, o, s;
        if (std::sscanf(line, "vf%d %f %f %f %f", &r, &x, &y, &z, &w) == 5 && r >= 0 && r < 32) {
            c.vf[r].f[0] = x; c.vf[r].f[1] = y; c.vf[r].f[2] = z; c.vf[r].f[3] = w;
        } else if (std::sscanf(line, "vi%d %x", &r, &v) == 2 && r >= 0 && r < 16) {
            c.vi[r] = v;
        } else if (std::sscanf(line, "TOP %x ITOP %x TOPS %x BASE %x OFFSET %x", &a, &b, &t, &o, &s) == 5) {
            g_top = a; g_itop = b;
        }
    }
    std::fclose(f);
    return true;
}

#ifdef _WIN32
std::atomic<bool> g_prof_stop;
std::map<uint64_t, uint64_t> g_prof;

void profiler(HANDLE h) {
    uint64_t image = (uint64_t)GetModuleHandleA(nullptr);
    while (!g_prof_stop) {
        Sleep(1);
        if (SuspendThread(h) == (DWORD)-1) continue;
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(h, &ctx)) ++g_prof[ctx.Rip - image];
        ResumeThread(h);
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vubench DIR N ADDR [--repeat R] [--out FILE] [--profile]\n");
        return 2;
    }
    std::string dir = argv[1], n = argv[2];
    uint32_t addr = (uint32_t)std::strtoul(argv[3], nullptr, 16);
    int repeat = 1000;
    bool profile = false;
    std::string out;
    for (int k = 4; k < argc; ++k) {
        if (!std::strcmp(argv[k], "--repeat") && k + 1 < argc) repeat = std::atoi(argv[++k]);
        else if (!std::strcmp(argv[k], "--out") && k + 1 < argc) out = argv[++k];
        else if (!std::strcmp(argv[k], "--profile")) profile = true;
    }
    char name[64];
    std::snprintf(name, sizeof name, "%04X", addr);
    if (!read_file(dir + "/micro_" + n + ".bin", g_micro, sizeof g_micro) ||
        !read_file(dir + "/mem_" + n + "_" + name + ".bin", g_mem0, sizeof g_mem0) ||
        !read_regs(dir + "/regs_" + n + "_" + name + ".txt", g_regs0)) {
        std::fprintf(stderr, "vubench: cannot read the dump in %s\n", dir.c_str());
        return 1;
    }
    rt_init_fpu();
    VUUnit u{1, &g_regs, g_micro, 0x3FFF, g_mem, 0x3FFF, 0, [](uint32_t) { ++g_kicks; },
             [] { return g_top; }, [] { return g_itop; }};
#ifdef _WIN32
    std::thread prof;
    if (profile) {
        HANDLE h;
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h, 0, FALSE, DUPLICATE_SAME_ACCESS);
        prof = std::thread(profiler, h);
    }
#endif
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < repeat; ++k) {
        std::memcpy(g_mem, g_mem0, sizeof g_mem0);
        g_regs = g_regs0;
        vu_exec(u, addr);
        if (k == 0 && !out.empty()) {
            FILE* f = std::fopen(out.c_str(), "wb");
            if (f) {
                std::fwrite(g_mem, 1, 0x4000, f);
                std::fwrite(&g_regs.vf, 1, sizeof g_regs.vf, f);
                std::fwrite(&g_regs.vi, 1, sizeof g_regs.vi, f);
                std::fwrite(&g_regs.vacc, 1, sizeof g_regs.vacc, f);
                std::fclose(f);
            }
        }
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#ifdef _WIN32
    if (profile) {
        g_prof_stop = true;
        prof.join();
    }
#endif
    std::printf("%d runs, %llu pairs (%llu a run), %llu XGKICKs: %.3f s, %.1f M pairs/s\n", repeat,
                (unsigned long long)u.pairs, (unsigned long long)(u.pairs / repeat), (unsigned long long)g_kicks, s,
                (double)u.pairs / s / 1e6);
#ifdef _WIN32
    if (profile) {
        std::vector<std::pair<uint64_t, uint64_t>> v(g_prof.begin(), g_prof.end());
        uint64_t total = 0;
        for (auto& e : v) total += e.second;
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        std::printf("profile: %llu samples; image offsets (add the image base for addr2line)\n", (unsigned long long)total);
        for (size_t k = 0; k < v.size() && k < 100000; ++k)
            std::printf("  %5.1f%%  %llX\n", 100.0 * v[k].second / total, (unsigned long long)v[k].first);
    }
#endif
    return 0;
}

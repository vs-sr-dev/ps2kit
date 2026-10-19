// ps2kit runtime — gsreplay: a GS recording (PS2KIT_GS_DUMP, gs.cpp) played
// again without the game, to time the rasteriser and to compare its two
// forms, the fast one and the reference, on the same packets.
//
//   gsreplay FILE                    time the fast rasteriser, one hash per field
//   gsreplay FILE --ref              the same with the reference rasteriser
//   gsreplay FILE --compare          both; local memory compared at every field,
//                                    and at the first difference packet by packet
//   gsreplay FILE --shots DIR        the PCRTC's picture at every field, as BMP
//   gsreplay FILE --repeat N         play it N times (timing)
//   gsreplay FILE --profile          where the time goes: the host's instruction
//                                    pointer every millisecond, as addresses in
//                                    the executable for addr2line (Windows)
#include "os.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <map>
#include <thread>

#include <chrono>
#include <cstdlib>
#include <cstring>

int g_trace = 0;
double g_run_seconds = 0;

void gs_set_reference(bool on);
uint64_t gs_pixels();

namespace {

struct Rec {
    uint32_t type;
    std::vector<uint8_t> data;
};

std::vector<Rec> g_recs;

bool load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t h[2];
    while (std::fread(h, 4, 2, f) == 2) {
        Rec r{h[0], std::vector<uint8_t>(h[1])};
        if (h[1] && std::fread(r.data.data(), 1, h[1], f) != h[1]) break;
        g_recs.push_back(std::move(r));
    }
    std::fclose(f);
    return !g_recs.empty() && g_recs[0].type == 'S';
}

uint64_t hash_vram() {
    const uint8_t* p = gs_vram();
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (size_t k = 0; k < (4u << 20); k += 8) {
        uint64_t w;
        std::memcpy(&w, p + k, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 29;
    }
    return h;
}

void play(size_t from, size_t to) {                 // records [from, to)
    for (size_t k = from; k < to; ++k) {
        const Rec& r = g_recs[k];
        if (r.type == 2) gs_path2(r.data.data(), (uint32_t)(r.data.size() / 16));
        else if (r.type == 3) gs_path3(r.data.data(), (uint32_t)(r.data.size() / 16));
        gs_take_events();
    }
}

void restart() {
    if (!gs_state_load(g_recs[0].data.data(), g_recs[0].data.size())) {
        std::fprintf(stderr, "gsreplay: the state record is not this runtime version of it (record again)\n");
        std::exit(1);
    }
}

// Field ends: the index of each 'V' record.
std::vector<size_t> marks() {
    std::vector<size_t> m;
    for (size_t k = 0; k < g_recs.size(); ++k)
        if (g_recs[k].type == 'V') m.push_back(k);
    return m;
}

void save_bmp(const std::string& path, const std::vector<uint32_t>& px, int w, int h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    int row = (w * 3 + 3) & ~3;
    uint8_t hdr[54] = {'B', 'M'};
    auto put32 = [&](int at, uint32_t v) { for (int k = 0; k < 4; ++k) hdr[at + k] = (uint8_t)(v >> (8 * k)); };
    put32(2, 54 + (uint32_t)row * (uint32_t)h); put32(10, 54); put32(14, 40); put32(18, (uint32_t)w);
    put32(22, (uint32_t)h); hdr[26] = 1; hdr[28] = 24; put32(34, (uint32_t)row * (uint32_t)h);
    std::fwrite(hdr, 1, 54, f);
    std::vector<uint8_t> line((size_t)row);
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            uint32_t c = px[(size_t)y * w + x];
            line[(size_t)x * 3] = (uint8_t)(c >> 16);
            line[(size_t)x * 3 + 1] = (uint8_t)(c >> 8);
            line[(size_t)x * 3 + 2] = (uint8_t)c;
        }
        std::fwrite(line.data(), 1, line.size(), f);
    }
    std::fclose(f);
}

// Plays every field with one rasteriser; returns the hash after each.
std::vector<uint64_t> run(bool ref, const std::string& shots, bool print) {
    gs_set_reference(ref);
    restart();
    std::vector<size_t> m = marks();
    std::vector<uint64_t> hs;
    size_t at = 1;
    uint64_t px0 = gs_pixels();
    double busy = 0;
    for (size_t f = 0; f < m.size(); ++f) {
        auto t0 = std::chrono::steady_clock::now();
        play(at, m[f]);
        busy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        at = m[f] + 1;
        hs.push_back(hash_vram());
        if (!shots.empty()) {
            uint64_t regs[7];
            std::memcpy(regs, g_recs[m[f]].data.data(), sizeof regs);
            std::vector<uint32_t> pic;
            int w, h;
            if (gs_snapshot(regs, pic, w, h)) {
                char name[64];
                std::snprintf(name, sizeof name, "/field_%04zu.bmp", f);
                save_bmp(shots + name, pic, w, h);
            }
        }
    }
    uint64_t px = gs_pixels() - px0;
    if (print)
        std::printf("%s: %zu fields, %llu pixels written, %.3f s drawing (%.2f ms a field, %.1f Mpx/s)\n",
                    ref ? "reference" : "fast", m.size(), (unsigned long long)px, busy,
                    m.empty() ? 0 : busy * 1000 / (double)m.size(), busy > 0 ? (double)px / busy / 1e6 : 0);
    return hs;
}

// The first field whose memory differs: played again packet by packet with
// both rasterisers, from the end of the field before, to the first packet
// after which they differ, and the differing bytes counted.
void locate(size_t field) {
    std::vector<size_t> m = marks();
    size_t from = field ? m[field - 1] + 1 : 1;
    std::vector<uint8_t> a(4u << 20), b(4u << 20);
    for (size_t k = from; k < m[field]; ++k) {
        gs_set_reference(false); restart(); play(1, k + 1);
        std::memcpy(a.data(), gs_vram(), a.size());
        gs_set_reference(true); restart(); play(1, k + 1);
        std::memcpy(b.data(), gs_vram(), b.size());
        size_t diff = 0, first = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) { if (!diff) first = i; ++diff; }
        if (diff) {
            std::printf("first difference after record %zu (path %u, %zu qw): %zu bytes, the first at %06zX\n", k,
                        g_recs[k].type, g_recs[k].data.size() / 16, diff, first);
            return;
        }
    }
}

#ifdef _WIN32
std::atomic<bool> g_prof_stop;
std::map<uint64_t, uint64_t> g_prof;                // address in the image -> samples

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

void profile_print() {
    std::vector<std::pair<uint64_t, uint64_t>> v(g_prof.begin(), g_prof.end());
    uint64_t total = 0;
    for (auto& e : v) total += e.second;
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::printf("profile: %llu samples; image offsets (add the image base for addr2line)\n", (unsigned long long)total);
    for (size_t k = 0; k < v.size() && k < 3000; ++k)
        std::printf("  %5.1f%%  %llX\n", 100.0 * v[k].second / total, (unsigned long long)v[k].first);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gsreplay FILE [--ref] [--compare] [--shots DIR] [--repeat N]\n");
        return 2;
    }
    bool ref = false, compare = false, profile = false;
    int repeat = 1;
    std::string shots;
    for (int k = 2; k < argc; ++k) {
        if (!std::strcmp(argv[k], "--ref")) ref = true;
        else if (!std::strcmp(argv[k], "--compare")) compare = true;
        else if (!std::strcmp(argv[k], "--profile")) profile = true;
        else if (!std::strcmp(argv[k], "--shots") && k + 1 < argc) shots = argv[++k];
        else if (!std::strcmp(argv[k], "--repeat") && k + 1 < argc) repeat = std::atoi(argv[++k]);
    }
    gs_init();
    if (!load(argv[1])) {
        std::fprintf(stderr, "gsreplay: cannot read %s\n", argv[1]);
        return 1;
    }
    size_t qw = 0;
    for (const Rec& r : g_recs)
        if (r.type == 2 || r.type == 3) qw += r.data.size() / 16;
    std::printf("%s: %zu records, %zu fields, %zu quadwords\n", argv[1], g_recs.size(), marks().size(), qw);
    if (compare) {
        std::vector<uint64_t> f = run(false, "", true), r = run(true, "", true);
        for (size_t k = 0; k < f.size(); ++k)
            if (f[k] != r[k]) {
                std::printf("field %zu: local memory differs\n", k);
                locate(k);
                return 1;
            }
        std::printf("local memory equal after each of the %zu fields\n", f.size());
        return 0;
    }
    std::vector<uint64_t> hs;
#ifdef _WIN32
    std::thread prof;
    if (profile) {
        HANDLE h;
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h, 0, FALSE,
                        DUPLICATE_SAME_ACCESS);
        prof = std::thread(profiler, h);
    }
#endif
    for (int k = 0; k < repeat; ++k) hs = run(ref, k == 0 ? shots : "", true);
#ifdef _WIN32
    if (profile) {
        g_prof_stop = true;
        prof.join();
        profile_print();
    }
#endif
    if (!hs.empty()) std::printf("hash after the last field: %016llX\n", (unsigned long long)hs.back());
    return 0;
}

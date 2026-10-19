// ps2kit runtime — the window: the PCRTC's picture, once per vertical blank,
// in an SDL3 window (4:3), and screenshots of it.
#include "os.h"

#include <SDL3/SDL.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

SDL_Window* g_win;
SDL_Renderer* g_ren;
SDL_Texture* g_tex;
int g_tw, g_th;
uint32_t g_last_vblank = ~0u;
std::vector<uint32_t> g_pic;
int g_pw, g_ph;
std::vector<double> g_shots;
std::string g_shot_dir = ".";
size_t g_next_shot;

void save_bmp(const std::string& path, const std::vector<uint32_t>& px, int w, int h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "ps2kit: cannot write %s\n", path.c_str()); return; }
    int row = (w * 3 + 3) & ~3;
    uint32_t size = 54 + (uint32_t)row * (uint32_t)h;
    uint8_t hdr[54] = {'B', 'M'};
    auto put32 = [&](int at, uint32_t v) { for (int k = 0; k < 4; ++k) hdr[at + k] = (uint8_t)(v >> (8 * k)); };
    put32(2, size); put32(10, 54); put32(14, 40); put32(18, (uint32_t)w); put32(22, (uint32_t)h);
    hdr[26] = 1; hdr[28] = 24; put32(34, (uint32_t)row * (uint32_t)h);
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
    std::fprintf(stderr, "[display] %s: %dx%d\n", path.c_str(), w, h);
}

}  // namespace

void display_init(bool window, const std::vector<double>& shots, const std::string& shot_dir) {
    g_shots = shots;
    g_shot_dir = shot_dir;
    if (window || !shots.empty()) gs_present_enable();
    if (!window) return;
    if (!SDL_Init(SDL_INIT_VIDEO)) rt_die("SDL_Init: %s", SDL_GetError());
    if (!SDL_CreateWindowAndRenderer("ps2kit", 960, 720, SDL_WINDOW_RESIZABLE, &g_win, &g_ren))
        rt_die("SDL_CreateWindowAndRenderer: %s", SDL_GetError());
    SDL_SetRenderLogicalPresentation(g_ren, 640, 480, SDL_LOGICAL_PRESENTATION_LETTERBOX);
    SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 255);
    SDL_RenderClear(g_ren);
    SDL_RenderPresent(g_ren);
}

// Called by the main thread every few milliseconds. False: the window closed.
bool display_pump(double now) {
    if (g_win) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) return false;
            input_event(e);
        }
    }
    input_update(now);
    uint32_t vb = hw_vblank_count();
    bool shot_due = g_next_shot < g_shots.size() && now >= g_shots[g_next_shot];
    if (vb == g_last_vblank && !shot_due) return true;
    g_last_vblank = vb;
    if (!g_win && !shot_due) return true;
    bool have = hw_gs_picture(g_pic, g_pw, g_ph);
    if (shot_due) {
        char name[64];
        std::snprintf(name, sizeof name, "/shot_%07.2f.bmp", g_shots[g_next_shot]);
        if (have) save_bmp(g_shot_dir + name, g_pic, g_pw, g_ph);
        else std::fprintf(stderr, "[display] no picture at %.2f s (PCRTC off)\n", now);
        ++g_next_shot;
    }
    if (!g_win) return true;
    if (have) {
        if (!g_tex || g_tw != g_pw || g_th != g_ph) {
            if (g_tex) SDL_DestroyTexture(g_tex);
            g_tex = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, g_pw, g_ph);
            g_tw = g_pw;
            g_th = g_ph;
        }
        SDL_UpdateTexture(g_tex, nullptr, g_pic.data(), g_pw * 4);
        SDL_RenderClear(g_ren);
        SDL_RenderTexture(g_ren, g_tex, nullptr, nullptr);
    } else {
        SDL_RenderClear(g_ren);
    }
    SDL_RenderPresent(g_ren);
    return true;
}

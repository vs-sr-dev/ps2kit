// ps2kit runtime — the controller in port 0, from the host: the keyboard and
// a gamepad through SDL3, and scripted presses for runs without a window.
//
// The state is kept as the pad's own button word, active high:
//
//   bit  0 L2      4 triangle   8 select   12 up
//        1 R2      5 circle     9 L3       13 right
//        2 L1      6 cross     10 R3       14 down
//        3 R1      7 square    11 start    15 left
//
// which is the inverse of the pad's answer bytes 3 (low) and 2 (high). The
// layout was read in the game's own code: CKzPad::Read builds its word as
// ~(byte2 << 8 | byte3), and its IsPush* tests take 0x80 for square
// (Shikaku), 0x20 circle (Maru), 0x10 triangle (Sankaku), 0x40 cross
// (Batsu), 0x4 L1, 0x1 L2, 0x8 R1, 0x2 R2, 0x800 start, 0x100 select,
// 0x1000 up, 0x2000 right, 0x4000 down, 0x8000 left. The sticks are bytes
// 4-7 of an analog answer: right X, right Y, left X, left Y, 0x80 centred.
//
// The default keyboard: arrows the directions, Enter start, Backspace or
// Right Shift select, X cross, C circle, Z square, S triangle, Q L1, W R1,
// 1 L2, 2 R2. A gamepad's buttons go to the same places by position
// (SDL's south button is cross, east circle, west square, north triangle).
//
// --press T:NAMES[,T:NAMES...] holds the named buttons (joined with +) at
// T seconds of host time for 0.2 s; T-U:NAMES holds them from T to U.
// Names: UP DOWN LEFT RIGHT START SELECT CROSS CIRCLE SQUARE TRIANGLE
// L1 L2 R1 R2 L3 R3.
#include "os.h"

#include <SDL3/SDL.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum : uint16_t {
    B_L2 = 1 << 0, B_R2 = 1 << 1, B_L1 = 1 << 2, B_R1 = 1 << 3, B_TRIANGLE = 1 << 4, B_CIRCLE = 1 << 5,
    B_CROSS = 1 << 6, B_SQUARE = 1 << 7, B_SELECT = 1 << 8, B_L3 = 1 << 9, B_R3 = 1 << 10, B_START = 1 << 11,
    B_UP = 1 << 12, B_RIGHT = 1 << 13, B_DOWN = 1 << 14, B_LEFT = 1 << 15,
};

const struct { const char* name; uint16_t bit; } NAMES[] = {
    {"UP", B_UP}, {"DOWN", B_DOWN}, {"LEFT", B_LEFT}, {"RIGHT", B_RIGHT}, {"START", B_START},
    {"SELECT", B_SELECT}, {"CROSS", B_CROSS}, {"CIRCLE", B_CIRCLE}, {"SQUARE", B_SQUARE},
    {"TRIANGLE", B_TRIANGLE}, {"L1", B_L1}, {"L2", B_L2}, {"R1", B_R1}, {"R2", B_R2}, {"L3", B_L3}, {"R3", B_R3},
};

const struct { SDL_Scancode key; uint16_t bit; } KEYS[] = {
    {SDL_SCANCODE_UP, B_UP}, {SDL_SCANCODE_DOWN, B_DOWN}, {SDL_SCANCODE_LEFT, B_LEFT},
    {SDL_SCANCODE_RIGHT, B_RIGHT}, {SDL_SCANCODE_RETURN, B_START}, {SDL_SCANCODE_BACKSPACE, B_SELECT},
    {SDL_SCANCODE_RSHIFT, B_SELECT}, {SDL_SCANCODE_X, B_CROSS}, {SDL_SCANCODE_C, B_CIRCLE},
    {SDL_SCANCODE_Z, B_SQUARE}, {SDL_SCANCODE_S, B_TRIANGLE}, {SDL_SCANCODE_Q, B_L1},
    {SDL_SCANCODE_W, B_R1}, {SDL_SCANCODE_1, B_L2}, {SDL_SCANCODE_2, B_R2},
};

const struct { SDL_GamepadButton button; uint16_t bit; } BUTTONS[] = {
    {SDL_GAMEPAD_BUTTON_SOUTH, B_CROSS}, {SDL_GAMEPAD_BUTTON_EAST, B_CIRCLE},
    {SDL_GAMEPAD_BUTTON_WEST, B_SQUARE}, {SDL_GAMEPAD_BUTTON_NORTH, B_TRIANGLE},
    {SDL_GAMEPAD_BUTTON_BACK, B_SELECT}, {SDL_GAMEPAD_BUTTON_START, B_START},
    {SDL_GAMEPAD_BUTTON_LEFT_STICK, B_L3}, {SDL_GAMEPAD_BUTTON_RIGHT_STICK, B_R3},
    {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, B_L1}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, B_R1},
    {SDL_GAMEPAD_BUTTON_DPAD_UP, B_UP}, {SDL_GAMEPAD_BUTTON_DPAD_DOWN, B_DOWN},
    {SDL_GAMEPAD_BUTTON_DPAD_LEFT, B_LEFT}, {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, B_RIGHT},
};

struct Press { double from, to; uint16_t bits; };
std::vector<Press> g_script;
bool g_window;
SDL_Gamepad* g_gamepad;

// Read by the IOP's vblank on the guest's thread, written by the main thread.
std::atomic<uint32_t> g_buttons{0};
std::atomic<uint32_t> g_sticks{0x80808080u};         // bytes: right X, right Y, left X, left Y

uint8_t axis_byte(int16_t v) { return (uint8_t)((v + 32768) >> 8); }

}  // namespace

void input_init(bool window, const std::string& script) {
    g_window = window;
    for (const char* s = script.c_str(); *s;) {
        char* end;
        Press p;
        p.from = std::strtod(s, &end);
        if (end == s) rt_die("--press: a time in seconds first, as in 30:START");
        p.to = p.from + 0.2;
        s = end;
        if (*s == '-') {
            p.to = std::strtod(s + 1, &end);
            if (end == s + 1 || p.to <= p.from) rt_die("--press: T-U with U after T");
            s = end;
        }
        if (*s++ != ':') rt_die("--press: T:NAMES, as in 30:START or 30-32:CROSS+UP");
        p.bits = 0;
        while (*s && *s != ',') {
            size_t n = std::strcspn(s, "+,");
            bool found = false;
            for (auto& b : NAMES)
                if (std::strlen(b.name) == n && !std::strncmp(s, b.name, n)) { p.bits |= b.bit; found = true; }
            if (!found) rt_die("--press: no button named %.*s", (int)n, s);
            s += n;
            if (*s == '+') ++s;
        }
        if (*s == ',') ++s;
        g_script.push_back(p);
    }
    if (window && !SDL_InitSubSystem(SDL_INIT_GAMEPAD)) rt_die("SDL_Init (gamepad): %s", SDL_GetError());
}

// From display_pump, on the main thread.
void input_event(const SDL_Event& e) {
    if (e.type == SDL_EVENT_GAMEPAD_ADDED && !g_gamepad) {
        g_gamepad = SDL_OpenGamepad(e.gdevice.which);
        if (g_gamepad) std::fprintf(stderr, "[input] gamepad: %s\n", SDL_GetGamepadName(g_gamepad));
    } else if (e.type == SDL_EVENT_GAMEPAD_REMOVED && g_gamepad && SDL_GetGamepadID(g_gamepad) == e.gdevice.which) {
        SDL_CloseGamepad(g_gamepad);
        g_gamepad = nullptr;
    }
}

void input_update(double now) {
    uint32_t bits = 0, sticks = 0x80808080u;
    for (auto& p : g_script)
        if (now >= p.from && now < p.to) bits |= p.bits;
    if (g_window) {
        const bool* keys = SDL_GetKeyboardState(nullptr);
        for (auto& k : KEYS)
            if (keys[k.key]) bits |= k.bit;
        if (g_gamepad) {
            for (auto& b : BUTTONS)
                if (SDL_GetGamepadButton(g_gamepad, b.button)) bits |= b.bit;
            if (SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16384) bits |= B_L2;
            if (SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16384) bits |= B_R2;
            sticks = (uint32_t)axis_byte(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTX)) |
                     (uint32_t)axis_byte(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTY)) << 8 |
                     (uint32_t)axis_byte(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTX)) << 16 |
                     (uint32_t)axis_byte(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTY)) << 24;
        }
    }
    uint32_t old = g_buttons.exchange(bits, std::memory_order_relaxed);
    if (old != bits) TRACE(1, "[input] %.2f s: buttons %04X\n", now, bits);
    g_sticks.store(sticks, std::memory_order_relaxed);
}

// The pad's answer bytes 2-7 as padman would read them from the controller.
void input_pad(uint8_t out[6]) {
    uint32_t bits = g_buttons.load(std::memory_order_relaxed), sticks = g_sticks.load(std::memory_order_relaxed);
    out[0] = (uint8_t)~(bits >> 8);
    out[1] = (uint8_t)~bits;
    for (int k = 0; k < 4; ++k) out[2 + k] = (uint8_t)(sticks >> (8 * k));
}

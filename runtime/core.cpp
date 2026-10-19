// ps2kit runtime — guest memory and dispatch, shared by every runtime target.
//
// RAM is 32 MB at physical 0; the EE sees it at 0x00000000 (kuseg), through
// the uncached mirrors 0x20000000 and 0x30000000 (uncached accelerated) and in
// kernel segments 0x80000000 / 0xA0000000. The scratchpad is 16 KB at
// 0x70000000. Everything else is hardware, handed to ee_hw_read/ee_hw_write
// (the stub dies there; the hardware layer answers).
#include "rt.h"

#include <algorithm>
#include <cfenv>
#include <cstdio>
#include <cstdlib>

uint8_t* g_ee_ram;
uint8_t* g_ee_spr;
uint8_t* g_vu0_mem;

extern const EEFuncEntry g_ee_funcs[];
extern const size_t g_ee_nfuncs;

void rt_init_memory() {
    if (g_ee_ram) return;
    g_ee_ram = (uint8_t*)std::calloc(1, EE_RAM_SIZE + 16);
    g_ee_spr = (uint8_t*)std::calloc(1, 0x4000 + 16);
    g_vu0_mem = (uint8_t*)std::calloc(1, 0x1000 + 16);
    if (!g_ee_ram || !g_ee_spr || !g_vu0_mem) rt_die("out of host memory");
}

// The PS2's FPU and VUs round toward zero ("chop"), which is also PCSX2's
// default; the host's SSE unit is set to the same mode on every thread that
// runs guest code. Not exact: the PS2's multiplier can differ in the last bit.
void rt_init_fpu() {
    std::fesetround(FE_TOWARDZERO);
}

const EEFuncEntry* rt_find(uint32_t addr) {
    const EEFuncEntry* end = g_ee_funcs + g_ee_nfuncs;
    const EEFuncEntry* f = std::lower_bound(g_ee_funcs, end, addr,
        [](const EEFuncEntry& e, uint32_t a) { return e.addr < a; });
    return (f != end && f->addr == addr) ? f : nullptr;
}

const char* rt_name_of(uint32_t addr) {
    const EEFuncEntry* end = g_ee_funcs + g_ee_nfuncs;
    const EEFuncEntry* f = std::upper_bound(g_ee_funcs, end, addr,
        [](uint32_t a, const EEFuncEntry& e) { return a < e.addr; });
    return f == g_ee_funcs ? "?" : (f - 1)->name;
}

void ee_call(EEContext& c, uint32_t addr) {
    const EEFuncEntry* f = rt_find(addr & 0x1FFFFFFFu);
    if (!f) rt_die("indirect call to %08X (not a function), from near %08X (%s)", addr, c.pc, rt_name_of(c.pc));
    f->fn(c);
}

// The host pointer behind a guest address, or nullptr for hardware.
static uint8_t* host(uint32_t a, uint32_t size) {
    uint32_t seg = a >> 28;
    if (seg == 0x7) {                                    // scratchpad
        uint32_t o = a & 0x3FFF;
        if (o + size <= 0x4000) return g_ee_spr + o;
        return nullptr;
    }
    uint32_t p = a & 0x1FFFFFFFu;
    if ((seg == 0x0 || seg == 0x2 || seg == 0x3 || seg == 0x8 || seg == 0xA) && p < EE_RAM_SIZE)
        return g_ee_ram + p;
    return nullptr;
}

uint64_t ee_read_slow(uint32_t a, int size) {
    if (uint8_t* p = host(a, size)) {
        uint64_t v = 0;
        std::memcpy(&v, p, size);
        return v;
    }
    return ee_hw_read(a, size);
}

void ee_write_slow(uint32_t a, uint64_t v, int size) {
    if (uint8_t* p = host(a, size)) {
        std::memcpy(p, &v, size);
        return;
    }
    ee_hw_write(a, v, size);
}

void ee_read128_slow(uint32_t a, EEReg& out) {
    if (uint8_t* p = host(a, 16)) {
        std::memcpy(&out, p, 16);
        return;
    }
    out.ud[0] = ee_hw_read(a, 8);
    out.ud[1] = ee_hw_read(a + 8, 8);
}

void ee_write128_slow(uint32_t a, const EEReg& v) {
    if (uint8_t* p = host(a, 16)) {
        std::memcpy(p, &v, 16);
        return;
    }
    ee_hw_write128(a, v);
}

[[noreturn]] void rt_die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "ps2kit: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::fflush(stderr);
    std::exit(1);
}

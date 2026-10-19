// Differential test driver: load a machine state, call one recompiled
// function, write the state back. ps2kit/recomp/difftest.py runs the same
// call in ps2kit.eeemu and compares the two.
//
//   difftest IN OUT
//
// IN and OUT share one format (all little-endian):
//   u32 magic 'EEDT', u32 function address
//   registers: r[32] (16 bytes each), hi (16), lo (16), sa (4), f[32] (4 each),
//              acc (4), fcr31 (4), vf[32] (16 each), vacc (16), vi[32] (4 each)
//   memory: RAM (32 MB), scratchpad (16 KB), VU0 data memory (4 KB)
#include "rt.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool io(FILE* f, void* p, size_t n, bool write) {
    return (write ? std::fwrite(p, 1, n, f) : std::fread(p, 1, n, f)) == n;
}

static bool state(FILE* f, EEContext& c, uint32_t& addr, bool write) {
    uint32_t magic = 0x54444545u;   // "EEDT"
    uint32_t m = magic;
    bool ok = io(f, &m, 4, write) && m == magic && io(f, &addr, 4, write);
    for (int k = 0; k < 32 && ok; ++k) ok = io(f, &c.r[k], 16, write);
    ok = ok && io(f, &c.hi, 16, write) && io(f, &c.lo, 16, write) && io(f, &c.sa, 4, write);
    for (int k = 0; k < 32 && ok; ++k) ok = io(f, &c.f[k], 4, write);
    ok = ok && io(f, &c.acc, 4, write) && io(f, &c.fcr31, 4, write);
    for (int k = 0; k < 32 && ok; ++k) ok = io(f, &c.vf[k], 16, write);
    ok = ok && io(f, &c.vacc, 16, write);
    for (int k = 0; k < 32 && ok; ++k) ok = io(f, &c.vi[k], 4, write);
    return ok && io(f, g_ee_ram, EE_RAM_SIZE, write) && io(f, g_ee_spr, 0x4000, write) &&
           io(f, g_vu0_mem, 0x1000, write);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: difftest IN OUT\n");
        return 2;
    }
    rt_init_memory();
    rt_init_fpu();
    static EEContext c;
    uint32_t addr = 0;
    FILE* in = std::fopen(argv[1], "rb");
    if (!in || !state(in, c, addr, false)) rt_die("cannot read %s", argv[1]);
    std::fclose(in);
    const EEFuncEntry* f = rt_find(addr);
    if (!f) rt_die("no recompiled function at %08X", addr);
    f->fn(c);
    FILE* out = std::fopen(argv[2], "wb");
    if (!out || !state(out, c, addr, true)) rt_die("cannot write %s", argv[2]);
    std::fclose(out);
    return 0;
}

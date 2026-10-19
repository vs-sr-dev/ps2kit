// ps2kit runtime — the boot: what the boot ROM does before the game's _start.
//
//   ps2run DISC [--elf FILE] [--trace N] [--seconds S] [--profile] [--headless]
//          [--shot T[,T...]] [--shot-dir DIR]
//
// DISC is the user's disc image (.cue, .iso or a raw .bin). As the boot ROM
// does, SYSTEM.CNF's BOOT2 names the executable (cdrom0:\SLES_500.57;1),
// which is read from the disc and becomes argv[0]; --elf reads it from a
// file instead. Its loadable segments go into RAM at their addresses (the
// part of a segment beyond its file size is zeroed) and the main thread
// starts at the entry point. The code that runs is the recompiled code
// linked into this executable; the ELF's code bytes are in RAM because the
// game may read them.
#include "os.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) rt_die("cannot open %s", path);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> d((size_t)n);
    if (std::fread(d.data(), 1, d.size(), f) != d.size()) rt_die("cannot read %s", path);
    std::fclose(f);
    return d;
}

template <class T> static T get(const std::vector<uint8_t>& d, size_t at) {
    if (at + sizeof(T) > d.size()) rt_die("ELF: read beyond the file at %zX", at);
    T v;
    std::memcpy(&v, d.data() + at, sizeof v);
    return v;
}

static uint32_t load_elf(const std::vector<uint8_t>& d, const char* what) {
    if (d.size() < 52 || std::memcmp(d.data(), "\x7F" "ELF", 4) != 0 || d[4] != 1 || d[5] != 1)
        rt_die("%s is not a 32-bit little-endian ELF", what);
    if (get<uint16_t>(d, 18) != 8) rt_die("%s is not a MIPS executable", what);
    uint32_t entry = get<uint32_t>(d, 24), phoff = get<uint32_t>(d, 28);
    uint16_t phentsize = get<uint16_t>(d, 42), phnum = get<uint16_t>(d, 44);
    for (int k = 0; k < phnum; ++k) {
        size_t ph = phoff + (size_t)k * phentsize;
        if (get<uint32_t>(d, ph) != 1) continue;          // PT_LOAD
        uint32_t off = get<uint32_t>(d, ph + 4), va = get<uint32_t>(d, ph + 8);
        uint32_t filesz = get<uint32_t>(d, ph + 16), memsz = get<uint32_t>(d, ph + 20);
        uint32_t pa = va & 0x1FFFFFFFu;
        if ((uint64_t)pa + memsz > EE_RAM_SIZE || (uint64_t)off + filesz > d.size())
            rt_die("ELF segment %d (%08X, %X bytes) does not fit", k, va, memsz);
        std::memcpy(g_ee_ram + pa, d.data() + off, filesz);
        std::memset(g_ee_ram + pa + filesz, 0, memsz - filesz);
        TRACE(1, "[boot] segment %08X: %X bytes from the file, %X in memory\n", va, filesz, memsz);
    }
    return entry;
}

int main(int argc, char** argv) {
    const char* disc = nullptr;
    const char* elf = nullptr;
    bool profile = false, window = true;
    std::vector<double> shots;
    std::string shot_dir = ".";
    const char* usage = "usage: ps2run DISC [--elf FILE] [--trace N] [--seconds S] [--profile] [--headless]"
                        " [--shot T[,T...]] [--shot-dir DIR]";
    for (int k = 1; k < argc; ++k) {
        if (!std::strcmp(argv[k], "--headless")) window = false;
        else if (!std::strcmp(argv[k], "--shot") && k + 1 < argc) {
            for (const char* s = argv[++k]; *s;) {
                char* end;
                shots.push_back(std::strtod(s, &end));
                s = *end == ',' ? end + 1 : end;
                if (end == s && *s) rt_die("--shot: times in seconds, separated by commas");
            }
        } else if (!std::strcmp(argv[k], "--shot-dir") && k + 1 < argc) shot_dir = argv[++k];
        else if (!std::strcmp(argv[k], "--elf") && k + 1 < argc) elf = argv[++k];
        else if (!std::strcmp(argv[k], "--trace") && k + 1 < argc) g_trace = std::atoi(argv[++k]);
        else if (!std::strcmp(argv[k], "--seconds") && k + 1 < argc) g_run_seconds = std::atof(argv[++k]);
        else if (!std::strcmp(argv[k], "--profile")) profile = true;
        else if (!disc) disc = argv[k];
        else rt_die("%s", usage);
    }
    if (!disc) rt_die("%s", usage);
    if (!disc_open(disc)) rt_die("%s: not a disc image with an ISO 9660 file system", disc);
    std::string boot = disc_boot_path();
    if (boot.empty()) rt_die("%s: no BOOT2 line in SYSTEM.CNF", disc);
    std::vector<uint8_t> image;
    if (elf) {
        image = read_file(elf);
    } else {
        size_t colon = boot.find(':');
        if (colon == std::string::npos || !disc_read_file(boot.substr(colon + 1), image))
            rt_die("%s: cannot read %s", disc, boot.c_str());
    }
    rt_init_memory();
    hw_init();
    iop_init();
    uint32_t entry = load_elf(image, elf ? elf : boot.c_str());
    if (!rt_find(entry)) rt_die("no recompiled function at the entry point %08X", entry);
    TRACE(1, "[boot] %s: entry %08X (%s)\n", boot.c_str(), entry, rt_name_of(entry));
    display_init(window, shots, shot_dir);
    hw_start_clock();
    if (profile) profile_start();
    return kernel_run(entry, boot.c_str());
}

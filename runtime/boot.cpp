// ps2kit runtime — the boot: what the boot ROM does before the game's _start.
//
//   ps2run ELF [--disc DIR] [--argv0 PATH] [--trace N]
//
// DIR holds the disc's files (by default, the ELF's own directory).
// The ELF is the game's own executable, read from the user's disc: its
// loadable segments go into RAM at their addresses (the part of a segment
// beyond its file size is zeroed), and the main thread starts at the entry
// point with the path the boot ROM would pass (SYSTEM.CNF's BOOT2 line)
// as argv[0]. The code that runs is the recompiled code linked into this
// executable; the ELF's code bytes are in RAM because the game may read them.
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

static uint32_t load_elf(const char* path) {
    std::vector<uint8_t> d = read_file(path);
    if (d.size() < 52 || std::memcmp(d.data(), "\x7F" "ELF", 4) != 0 || d[4] != 1 || d[5] != 1)
        rt_die("%s is not a 32-bit little-endian ELF", path);
    if (get<uint16_t>(d, 18) != 8) rt_die("%s is not a MIPS executable", path);
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
    const char* elf = nullptr;
    std::string argv0 = "cdrom0:\\SLES_500.57;1", disc;
    const char* usage = "usage: ps2run ELF [--disc DIR] [--argv0 PATH] [--trace N]";
    for (int k = 1; k < argc; ++k) {
        if (!std::strcmp(argv[k], "--argv0") && k + 1 < argc) argv0 = argv[++k];
        else if (!std::strcmp(argv[k], "--disc") && k + 1 < argc) disc = argv[++k];
        else if (!std::strcmp(argv[k], "--trace") && k + 1 < argc) g_trace = std::atoi(argv[++k]);
        else if (!elf) elf = argv[k];
        else rt_die("%s", usage);
    }
    if (!elf) rt_die("%s", usage);
    rt_init_memory();
    if (disc.empty()) {
        disc = elf;
        size_t cut = disc.find_last_of("\\/");
        disc = cut == std::string::npos ? "." : disc.substr(0, cut);
    }
    hw_init();
    iop_init(disc.c_str());
    uint32_t entry = load_elf(elf);
    if (!rt_find(entry)) rt_die("no recompiled function at the entry point %08X", entry);
    TRACE(1, "[boot] entry %08X (%s), argv[0] %s\n", entry, rt_name_of(entry), argv0.c_str());
    hw_start_clock();
    return kernel_run(entry, argv0.c_str());
}

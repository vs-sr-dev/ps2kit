// ps2kit runtime — vutrans: VU1 microcode translated to C++ (vu.cpp's
// vu1_translate), from what PS2KIT_VU1_DUMP wrote while the game ran.
//
//   vutrans OUT.cpp DIR [DIR...]
//
// For every micro memory (DIR/micro_N.bin) and every address a program
// started from in it (DIR/starts.txt), the program as a function. A program
// found again, the same pairs at the same addresses, is written once. OUT.cpp
// holds the functions and the table the runtime looks them up in
// (g_vu1_trans, vu_ops.h); the game's build compiles it in when it is
// beside the recompiled code as vu1_trans.cpp (runtime.cmake). A program the
// translator leaves (with the reason printed) runs in the interpreter.
#include "vu_ops.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

int g_trace = 0;
double g_run_seconds = 0;

// What vu.cpp calls besides itself: vutrans stands alone, without the game's
// recompiled code (which holds what vutrans writes).
[[noreturn]] void rt_die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "vutrans: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}
VUReg* ee_vu0_vu1reg(uint32_t qaddr) { rt_die("VU0 access to VU1's registers (%04X)", qaddr * 16); }

namespace {

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.assign(0x4000, 0);
    bool ok = std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: vutrans OUT.cpp DIR [DIR...]\n");
        return 2;
    }
    std::string body, table;
    std::set<std::string> done;                      // entry and covered words, of programs written
    std::map<std::string, std::string> left;         // reasons, once each
    int n_fn = 0, n_seen = 0;
    for (int a = 2; a < argc; ++a) {
        std::string dir = argv[a];
        FILE* st = std::fopen((dir + "/starts.txt").c_str(), "r");
        if (!st) {
            std::fprintf(stderr, "vutrans: no %s/starts.txt\n", dir.c_str());
            return 1;
        }
        std::set<std::pair<int, unsigned>> starts;
        int n;
        unsigned pc;
        while (std::fscanf(st, "%d %x", &n, &pc) == 2) starts.insert({n, pc});
        std::fclose(st);
        std::map<int, std::vector<uint8_t>> micro;
        for (auto [mn, entry] : starts) {
            if (!micro.count(mn) &&
                !read_file(dir + "/micro_" + std::to_string(mn) + ".bin", micro[mn])) {
                std::fprintf(stderr, "vutrans: cannot read %s/micro_%d.bin\n", dir.c_str(), mn);
                return 1;
            }
            const std::vector<uint8_t>& m = micro[mn];
            ++n_seen;
            char name[48];
            std::snprintf(name, sizeof name, "vu1_%04X_%d", entry, n_fn);
            std::string code, why;
            std::vector<uint32_t> covered;
            if (!vu1_translate(m.data(), entry, name, code, covered, why)) {
                char k[96];
                std::snprintf(k, sizeof k, "%04X", entry);
                if (!left.count(k)) left[k] = why;
                continue;
            }
            // the words it was translated from: ranges of consecutive pairs
            std::string key = std::to_string(entry);
            std::vector<uint32_t> words;
            for (size_t i = 0; i < covered.size();) {
                size_t j = i;
                while (j + 1 < covered.size() && covered[j + 1] == covered[j] + 8) ++j;
                words.push_back(covered[i]);
                words.push_back((uint32_t)(j - i + 1));
                for (size_t p = i; p <= j; ++p) {
                    uint32_t lo, up;
                    std::memcpy(&lo, m.data() + covered[p], 4);
                    std::memcpy(&up, m.data() + covered[p] + 4, 4);
                    words.push_back(lo);
                    words.push_back(up);
                }
                i = j + 1;
            }
            words.push_back(0);
            words.push_back(0);
            for (uint32_t w : words) key += "," + std::to_string(w);
            if (!done.insert(key).second) continue;
            body += code;
            body += "const uint32_t " + std::string(name) + "_code[] = {";
            for (size_t k = 0; k < words.size(); ++k) {
                char b[32];
                std::snprintf(b, sizeof b, "%s0x%X", k ? (k % 8 ? ", " : ",\n    ") : "\n    ", words[k]);
                body += b;
            }
            body += "};\n\n";
            char row[128];
            std::snprintf(row, sizeof row, "    {0x%04X, %s, %s_code},\n", entry, name, name);
            table += row;
            ++n_fn;
        }
    }
    FILE* out = std::fopen(argv[1], "w");
    if (!out) {
        std::fprintf(stderr, "vutrans: cannot write %s\n", argv[1]);
        return 1;
    }
    std::fprintf(out, "// VU1 microcode translated to C++ by ps2kit's vutrans (do not edit).\n");
    std::fprintf(out, "#include \"vu_ops.h\"\n\nnamespace {\n\n%s}  // namespace\n\n", body.c_str());
    std::fprintf(out, "extern const VUTrans g_vu1_trans[] = {\n%s    {0, nullptr, nullptr}};\n", table.c_str());
    std::fprintf(out, "extern const size_t g_vu1_ntrans = %d;\n", n_fn);
    std::fclose(out);
    std::printf("vutrans: %d programs from %d starts\n", n_fn, n_seen);
    for (auto& [k, why] : left) std::printf("  %s left to the interpreter: %s\n", k.c_str(), why.c_str());
    return 0;
}

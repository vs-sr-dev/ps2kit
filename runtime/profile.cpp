// ps2kit runtime — a sampling profile of one of the runtime's own threads
// (PS2KIT_VIF1_PROFILE, PS2KIT_GS_PROFILE, PS2KIT_GS_WORKER_PROFILE), shared
// by ps2run and gsreplay.
//
// The calling thread is stopped every millisecond and its instruction
// pointer counted, from PS2KIT_PROFILE_FROM seconds after the first call
// on; `file` is written again every second with the counts as offsets into
// the executable (nm names them), or, outside it, with the system library
// and the offset in it.
#include "os.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
void profile_this_thread(const char* file) {
    static const auto t0 = std::chrono::steady_clock::now();
    HANDLE h;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h, 0, FALSE, DUPLICATE_SAME_ACCESS);
    std::thread([h, f = std::string(file)] {
        std::map<uint64_t, uint64_t> counts;
        uint64_t image = (uint64_t)GetModuleHandleA(nullptr);
        auto last = std::chrono::steady_clock::now();
        static const double from = std::getenv("PS2KIT_PROFILE_FROM") ? std::atof(std::getenv("PS2KIT_PROFILE_FROM")) : 0;
        for (;;) {
            Sleep(1);
            double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (now >= from && SuspendThread(h) != (DWORD)-1) {
                CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_CONTROL;
                bool got = GetThreadContext(h, &ctx);
                ResumeThread(h);                     // before counting: the map allocates, and the
                if (got) ++counts[ctx.Rip - image];   // thread stopped may hold the heap's lock
            }
            if (std::chrono::steady_clock::now() - last < std::chrono::seconds(1)) continue;
            last = std::chrono::steady_clock::now();
            FILE* o = std::fopen(f.c_str(), "w");
            if (!o) continue;
            for (auto& [a, n] : counts) {
                HMODULE m = nullptr;
                char name[MAX_PATH] = "";
                uint64_t raw = a + image;
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       (LPCSTR)raw, &m) && (uint64_t)m != image) {
                    GetModuleFileNameA(m, name, sizeof name);
                    const char* base = std::strrchr(name, '\\');
                    std::fprintf(o, "%llX %llu %s+%llX\n", (unsigned long long)a, (unsigned long long)n,
                                 base ? base + 1 : name, (unsigned long long)(raw - (uint64_t)m));
                } else {
                    std::fprintf(o, "%llX %llu\n", (unsigned long long)a, (unsigned long long)n);
                }
            }
            std::fclose(o);
        }
    }).detach();
}
#else
void profile_this_thread(const char*) {}
#endif

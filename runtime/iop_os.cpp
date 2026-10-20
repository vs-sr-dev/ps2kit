// ps2kit runtime — modules run on the IOP's own processor.
//
// The sound modules (Sony's LIBSD, MODHSYN, MODMIDI, MODMSIN; Koei's
// KOEIADS and KESSOUND) run unchanged on an R3000 interpreter (iop_cpu.cpp),
// on a host thread of their own: the IOP. The kernel libraries they import
// are host code, by library and ordinal (the SDK's public numbering, each
// checked against the modules' own calls and arguments); the SPU2 is
// spu2.cpp; everything else of the IOP stays at the SIF boundary (iop.cpp).
//
// What is modelled, as the modules use it:
//   loadcore  modules relocated and linked as the IOP's loader does: import
//             stubs patched to a jump (a library a run module registered)
//             or to a SYSCALL whose code names the host function
//   sysmem    a first-fit allocator over 0x20000-0x100000 (module images,
//             thread stacks, the modules' own allocations)
//   thbase, thsemap, thevent   threads by priority (lower is higher; among
//             equals in the order they became ready, never time-sliced, as
//             the IOP's kernel does), sleep and wake-up, semaphores, event flags
//   intrman   handlers by interrupt line, called between instructions when
//             the running thread has interrupts enabled
//   timrman   the hardware timers by their mode, compare and clock
//             (horizontal lines or the 36.864 MHz system clock)
//   sifman, sifcmd   SIF DMA to the EE; RPC servers whose calls run on the
//             thread in sceSifRpcLoop, replies sent when the function returns
//   cdvdman   sector reads and file searches on the disc, done at once
//   sysclib, stdio   memcpy; printf, to the host's console
#include "os.h"
#include "iop_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#endif

namespace {

using u32 = uint32_t;
using u16 = uint16_t;

// ---- memory ------------------------------------------------------------------------------------
constexpr u32 RAM_MASK = 0x1FFFFF;
uint8_t* ram() { return iop_ram(); }
u32 ld32(u32 a) { u32 v; std::memcpy(&v, ram() + (a & RAM_MASK), 4); return v; }
u16 ld16(u32 a) { u16 v; std::memcpy(&v, ram() + (a & RAM_MASK), 2); return v; }
void st32(u32 a, u32 v) { std::memcpy(ram() + (a & RAM_MASK), &v, 4); }
std::string cstr(u32 a, size_t max = 256) {
    std::string s;
    for (size_t k = 0; k < max; ++k) {
        char ch = (char)ram()[(a + k) & RAM_MASK];
        if (!ch) break;
        s += ch;
    }
    return s;
}

uint8_t g_scratch[1024];
std::vector<u32> g_io(0x1000 / 4);                    // 0x1F801000-0x1F801FFF, as written
std::set<u32> g_io_seen;

// ---- sysmem ------------------------------------------------------------------------------------
// Blocks of 256 bytes between SYS_START and SYS_END (iop.cpp keeps the
// command buffer below and its own servers' data and the EE's heap above).
constexpr u32 SYS_START = 0x00020000, SYS_END = 0x00100000;
std::map<u32, u32> g_used;                             // address -> size

u32 sys_alloc(int type, u32 size, u32 at) {
    size = (size + 255) & ~255u;
    if (!size) return 0;
    std::vector<std::pair<u32, u32>> gaps;            // free spans, low to high
    u32 p = SYS_START;
    for (auto& b : g_used) {
        if (b.first > p) gaps.push_back({p, b.first - p});
        p = b.first + b.second;
    }
    if (SYS_END > p) gaps.push_back({p, SYS_END - p});
    u32 got = 0;
    if (type == 0) {                                   // the lowest that fits
        for (auto& g : gaps) if (g.second >= size) { got = g.first; break; }
    } else if (type == 1) {                            // the highest that fits, at its top
        for (auto it = gaps.rbegin(); it != gaps.rend(); ++it)
            if (it->second >= size) { got = it->first + it->second - size; break; }
    } else if (type == 2) {                            // at an address
        at &= ~255u;
        for (auto& g : gaps) if (at >= g.first && at + size <= g.first + g.second) { got = at; break; }
    }
    if (got) g_used[got] = size;
    return got;
}
bool sys_free(u32 a) { return g_used.erase(a) == 1; }
u32 sys_free_total() {
    u32 used = 0;
    for (auto& b : g_used) used += b.second;
    return SYS_END - SYS_START - used;
}
u32 sys_free_max() {
    u32 p = SYS_START, best = 0;
    for (auto& b : g_used) { best = std::max(best, b.first - p); p = b.first + b.second; }
    return std::max(best, SYS_END - p);
}

// ---- threads -----------------------------------------------------------------------------------
constexpr u32 MAGIC_EXIT = IOP_MAGIC_BASE, MAGIC_INTR = IOP_MAGIC_BASE + 0x100,
              MAGIC_RPC = IOP_MAGIC_BASE + 0x200, MAGIC_START = IOP_MAGIC_BASE + 0x300;

enum class St { DORMANT, READY, WAIT };
enum Wait { W_NONE, W_SLEEP = 1, W_SEMA = 3, W_EVENT = 4, W_RPC = 100 };

struct Thread {
    int id = 0;
    u32 attr = 0, option = 0, entry = 0, stack = 0, stack_size = 0, gp = 0;
    int init_prio = 0, prio = 0;
    St st = St::DORMANT;
    int wait = W_NONE, wait_id = 0;
    u32 wakeups = 0;
    uint64_t seq = 0;                                  // when it became ready
    IopCpu cpu;
    bool intr_off = false;                             // CpuSuspendIntr's state while it runs
    u32 ev_bits = 0, ev_mode = 0, ev_res = 0;          // WaitEventFlag
    u32 rpc_qd = 0, rpc_sp = 0, rpc_gp = 0;            // sceSifRpcLoop
    bool in_call = false;
    IopRpcCall call{};                                 // the call it runs (or, for a loader, loadfile's)
    bool loader = false;                               // runs a module's start function
    std::string module;
};
std::map<int, Thread> g_threads;
int g_thread_ids = 0;
uint64_t g_seq = 0;
Thread* g_cur = nullptr;                               // the thread running (null: none)
IopCpu* g_ctx = nullptr;                               // the context running: a thread's or a handler's
bool g_in_intr = false;

void make_ready(Thread& t) {
    t.st = St::READY;
    t.wait = W_NONE;
    t.seq = ++g_seq;
}
void block(Thread& t, int wait, int id = 0) {
    t.st = St::WAIT;
    t.wait = wait;
    t.wait_id = id;
}
Thread* pick() {
    Thread* best = nullptr;
    for (auto& e : g_threads) {
        Thread& t = e.second;
        if (t.st != St::READY) continue;
        if (!best || t.prio < best->prio || (t.prio == best->prio && t.seq < best->seq)) best = &t;
    }
    return best;
}
Thread* thread(int id) {
    auto it = g_threads.find(id);
    return it == g_threads.end() ? nullptr : &it->second;
}

struct Sema { u32 attr, option; int count, max; std::deque<int> waiters; };
std::map<int, Sema> g_semas;
int g_sema_ids = 0;

struct EventFlag { u32 attr, option, init, bits; std::deque<int> waiters; };
std::map<int, EventFlag> g_events;
int g_event_ids = 0;

// ---- interrupts ------------------------------------------------------------------------------
struct Intr { u32 fn = 0, arg = 0, gp = 0, mode = 0; bool used = false, enabled = false, pending = false; };
Intr g_intr[0x40];
u32 g_intr_stack = 0;                                  // the handlers' stack (top)
uint64_t g_intr_calls = 0;
std::atomic<uint64_t> g_intr_count[0x40];               // handler calls by line (the heartbeat)

// ---- hardware timers -------------------------------------------------------------------------
// The IOP's six counters: 0-2 16 bits (interrupts 4, 5, 6), 3-5 32 bits
// (14, 15, 16). 1 and 3 can count horizontal lines (mode bit 8).
struct HwTimer { bool used = false; u32 mode = 0, compare = 0, flags = 0; double next = 0, period = 0; };
HwTimer g_tim[6];
const int TIMER_INTR[6] = {4, 5, 6, 14, 15, 16};

// ---- RPC ---------------------------------------------------------------------------------------
struct Server { u32 sid, sd, func, buff, qd; };
std::map<u32, Server> g_rpc_servers;                   // by sd (the server's data, as the EE names it)
struct Queue { int thread = 0; std::deque<IopRpcCall> calls; };
std::map<u32, Queue> g_queues;                         // by the queue's address

// ---- libraries ---------------------------------------------------------------------------------
struct Library { std::string name; u16 version; std::vector<u32> funcs; };
std::vector<Library> g_libs;
struct Module { std::string name, path; u32 base = 0, size = 0, entry = 0, gp = 0; u16 version = 0; };
std::vector<Module> g_modules;

// ---- requests from the EE (any thread) --------------------------------------------------------
struct Request {
    enum { LOAD, CALL } kind;
    IopRpcCall call;
    std::string path;
    std::vector<uint8_t> image, args;
};
std::mutex g_req_m;
std::condition_variable g_req_cv;
std::deque<Request> g_reqs;
std::thread g_thread;
std::atomic<uint64_t> g_instructions{0}, g_hle_calls{0};
std::atomic<int> g_threads_n{0};

// ---- printf ------------------------------------------------------------------------------------
std::string format(u32 fmt, const std::function<u32(int)>& arg, int first) {
    std::string out, f = cstr(fmt, 1024);
    int n = first;
    for (size_t k = 0; k < f.size(); ++k) {
        if (f[k] != '%') { out += f[k]; continue; }
        std::string spec = "%";
        ++k;
        while (k < f.size() && std::strchr("-+ #0", f[k])) spec += f[k++];
        while (k < f.size() && (std::isdigit((unsigned char)f[k]) || f[k] == '.')) spec += f[k++];
        while (k < f.size() && (f[k] == 'l' || f[k] == 'h')) ++k;
        if (k >= f.size()) break;
        char c = f[k];
        char buf[512];
        if (c == '%') { out += '%'; continue; }
        if (c == 's') {
            std::string s = cstr(arg(n++), 400);
            std::snprintf(buf, sizeof buf, (spec + "s").c_str(), s.c_str());
        } else if (c == 'c') {
            std::snprintf(buf, sizeof buf, (spec + "c").c_str(), (int)arg(n++));
        } else if (c == 'd' || c == 'i') {
            std::snprintf(buf, sizeof buf, (spec + "d").c_str(), (int)arg(n++));
        } else if (c == 'u' || c == 'x' || c == 'X' || c == 'o') {
            std::snprintf(buf, sizeof buf, (spec + c).c_str(), (unsigned)arg(n++));
        } else if (c == 'p') {
            std::snprintf(buf, sizeof buf, "%08X", (unsigned)arg(n++));
        } else {
            std::snprintf(buf, sizeof buf, "%%%c", c);
        }
        out += buf;
    }
    return out;
}
std::string g_line;
void console(const std::string& s) {
    for (char c : s) {
        if (c == '\n') { std::fprintf(stderr, "[iop] %s\n", g_line.c_str()); g_line.clear(); }
        else if (c != '\r') g_line += c;
    }
}

// ---- the host functions ------------------------------------------------------------------------
enum Res { RET, BLOCKED, JUMPED };
IopCpu& cpu() { return *g_ctx; }
u32 arg(int k) { return k < 4 ? cpu().r[4 + k] : ld32(cpu().r[29] + 16 + 4 * (k - 4)); }
Res ret(u32 v) { cpu().r[2] = v; return RET; }
Thread& need_thread(const char* what) {
    if (!g_cur || g_in_intr) rt_die("IOP: %s outside a thread", what);
    return *g_cur;
}

constexpr int KE_OK = 0, KE_ILLEGAL_THID = -406, KE_UNKNOWN_THID = -407, KE_NOT_DORMANT = -414,
              KE_UNKNOWN_SEMID = -408, KE_UNKNOWN_EVFID = -409, KE_SEMA_OVF = -420,
              KE_CPUDI = -102, KE_NO_MEMORY = -400;

void wake_sleeper(Thread& t, u32 v0) {
    t.cpu.r[2] = v0;
    make_ready(t);
}

bool event_match(const EventFlag& e, u32 bits, u32 mode) {
    return (mode & 1) ? (e.bits & bits) != 0 : (e.bits & bits) == bits;
}
void event_check(EventFlag& e) {
    for (auto it = e.waiters.begin(); it != e.waiters.end();) {
        Thread* t = thread(*it);
        if (t && event_match(e, t->ev_bits, t->ev_mode)) {
            if (t->ev_res) st32(t->ev_res, e.bits);
            if (t->ev_mode & 0x10) e.bits &= ~t->ev_bits;
            if (t->ev_mode & 0x20) e.bits = 0;
            wake_sleeper(*t, KE_OK);
            it = e.waiters.erase(it);
        } else {
            ++it;
        }
    }
}

void call_dispatch(Thread& t);

using HostFn = Res (*)();
struct Host { const char* lib; int ord; const char* name; HostFn fn; };

const Host HOST[] = {
    // loadcore
    {"loadcore", 5, "FlushDcache", [] { return ret(0); }},
    {"loadcore", 6, "RegisterLibraryEntries", [] {
        u32 t = arg(0);
        if (ld32(t) != 0x41C00000u) rt_die("IOP: RegisterLibraryEntries at %06X without its magic", t);
        Library lib;
        lib.version = ld16(t + 8);
        for (int k = 0; k < 8 && ram()[(t + 12 + k) & RAM_MASK]; ++k) lib.name += (char)ram()[(t + 12 + k) & RAM_MASK];
        for (u32 p = t + 20; ld32(p); p += 4) lib.funcs.push_back(ld32(p));
        TRACE(1, "[iop] library %s v%X: %zu entries\n", lib.name.c_str(), lib.version, lib.funcs.size());
        g_libs.push_back(lib);
        return ret(0);
    }},
    {"loadcore", 12, "QueryBootMode", [] { return ret(0); }},
    // sysmem
    {"sysmem", 4, "AllocSysMemory", [] {
        u32 a = sys_alloc((int)arg(0), arg(1), arg(2));
        TRACE(1, "[iop] AllocSysMemory(%u, %X) -> %06X\n", arg(0), arg(1), a);
        return ret(a);
    }},
    {"sysmem", 5, "FreeSysMemory", [] { return ret(sys_free(arg(0)) ? 0 : (u32)-1); }},
    {"sysmem", 6, "QueryMemSize", [] { return ret(0x200000); }},
    {"sysmem", 7, "QueryMaxFreeMemSize", [] { return ret(sys_free_max()); }},
    {"sysmem", 8, "QueryTotalFreeMemSize", [] { return ret(sys_free_total()); }},
    // stdio, sysclib
    {"stdio", 4, "printf", [] {
        console(format(arg(0), arg, 1));
        return ret(0);
    }},
    {"sysclib", 12, "memcpy", [] {
        u32 d = arg(0), s = arg(1), n = arg(2);
        for (u32 k = 0; k < n; ++k) ram()[(d + k) & RAM_MASK] = ram()[(s + k) & RAM_MASK];
        return ret(d);
    }},
    // intrman
    {"intrman", 4, "RegisterIntrHandler", [] {
        u32 n = arg(0);
        if (n >= 0x40) return ret((u32)-100);
        if (g_intr[n].used) return ret((u32)-104);
        // (whether the line is enabled is INTC's mask, kept as it was)
        g_intr[n] = Intr{arg(2), arg(3), cpu().r[28], arg(1), true, g_intr[n].enabled, g_intr[n].pending};
        TRACE(1, "[iop] interrupt %X: handler %06X (%06X)\n", n, arg(2), arg(3));
        return ret(0);
    }},
    {"intrman", 5, "ReleaseIntrHandler", [] {
        u32 n = arg(0);
        if (n < 0x40) g_intr[n] = Intr{0, 0, 0, 0, false, g_intr[n].enabled, g_intr[n].pending};
        return ret(0);
    }},
    {"intrman", 6, "EnableIntr", [] {
        if (arg(0) < 0x40) g_intr[arg(0)].enabled = true;
        TRACE(1, "[iop] EnableIntr(%X)\n", arg(0));
        return ret(0);
    }},
    {"intrman", 7, "DisableIntr", [] {
        u32 n = arg(0);
        if (n < 0x40) g_intr[n].enabled = false;
        if (arg(1)) st32(arg(1), n);
        return ret(0);
    }},
    {"intrman", 8, "CpuDisableIntr", [] {
        Thread& t = need_thread("CpuDisableIntr");
        bool was = t.intr_off;
        t.intr_off = true;
        return ret(was ? (u32)KE_CPUDI : 0);
    }},
    {"intrman", 9, "CpuEnableIntr", [] {
        if (g_cur && !g_in_intr) g_cur->intr_off = false;
        return ret(0);
    }},
    {"intrman", 17, "CpuSuspendIntr", [] {
        if (g_in_intr) { if (arg(0)) st32(arg(0), 0); return ret((u32)KE_CPUDI); }
        Thread& t = need_thread("CpuSuspendIntr");
        bool was = t.intr_off;
        if (arg(0)) st32(arg(0), was ? 0 : 0x404);    // the status bits restored by CpuResumeIntr
        t.intr_off = true;
        return ret(was ? (u32)KE_CPUDI : 0);
    }},
    {"intrman", 18, "CpuResumeIntr", [] {
        if (!g_in_intr && g_cur) g_cur->intr_off = arg(0) == 0;
        return ret(0);
    }},
    // thbase
    {"thbase", 4, "CreateThread", [] {
        u32 p = arg(0);
        Thread t;
        t.id = ++g_thread_ids;
        t.attr = ld32(p);
        t.option = ld32(p + 4);
        t.entry = ld32(p + 8);
        t.stack_size = (ld32(p + 12) + 255) & ~255u;
        t.init_prio = t.prio = (int)ld32(p + 16);
        t.gp = cpu().r[28];
        t.stack = sys_alloc(1, t.stack_size, 0);
        if (!t.stack) return ret((u32)KE_NO_MEMORY);
        TRACE(1, "[iop] thread %d: entry %06X, priority %d, stack %06X + %X\n", t.id, t.entry, t.prio, t.stack,
              t.stack_size);
        g_threads[t.id] = t;
        ++g_threads_n;
        return ret((u32)t.id);
    }},
    {"thbase", 5, "DeleteThread", [] {
        Thread* t = thread((int)arg(0));
        if (!t) return ret((u32)KE_UNKNOWN_THID);
        if (t->st != St::DORMANT) return ret((u32)KE_NOT_DORMANT);
        sys_free(t->stack);
        g_threads.erase(t->id);
        --g_threads_n;
        return ret(0);
    }},
    {"thbase", 6, "StartThread", [] {
        Thread* t = thread((int)arg(0));
        if (!t) return ret((u32)KE_UNKNOWN_THID);
        if (t->st != St::DORMANT) return ret((u32)KE_NOT_DORMANT);
        t->cpu = IopCpu{};
        t->cpu.r[4] = arg(1);
        t->cpu.r[28] = t->gp;
        t->cpu.r[29] = t->stack + t->stack_size - 0x10;
        t->cpu.r[31] = MAGIC_EXIT;
        t->cpu.pc = t->entry;
        t->cpu.npc = t->entry + 4;
        t->prio = t->init_prio;
        t->wakeups = 0;
        t->intr_off = false;
        make_ready(*t);
        return ret(0);
    }},
    {"thbase", 8, "ExitThread", [] {
        Thread& t = need_thread("ExitThread");
        t.st = St::DORMANT;
        return BLOCKED;
    }},
    {"thbase", 16, "RotateThreadReadyQueue", [] {
        Thread& t = need_thread("RotateThreadReadyQueue");
        int prio = (int)arg(0) ? (int)arg(0) : t.prio;
        if (prio == t.prio) t.seq = ++g_seq;
        return ret(0);
    }},
    {"thbase", 20, "GetThreadId", [] { return ret((u32)need_thread("GetThreadId").id); }},
    {"thbase", 22, "ReferThreadStatus", [] {
        Thread* t = arg(0) ? thread((int)arg(0)) : g_cur;
        if (!t) return ret((u32)KE_UNKNOWN_THID);
        u32 p = arg(1);
        int status = t->st == St::DORMANT ? 0x10 : t->st == St::WAIT ? 4 : (t == g_cur ? 1 : 2);
        u32 info[17] = {t->attr, t->option, (u32)status, t->entry, t->stack, t->stack_size, t->gp,
                        (u32)t->init_prio, (u32)t->prio, (u32)(t->st == St::WAIT ? (t->wait == W_RPC ? W_SLEEP : t->wait) : 0),
                        (u32)t->wait_id, t->wakeups, 0, 0, 0, 0, 0};
        for (int k = 0; k < 17; ++k) st32(p + 4 * k, info[k]);
        return ret(0);
    }},
    {"thbase", 24, "SleepThread", [] {
        Thread& t = need_thread("SleepThread");
        if (t.wakeups) { --t.wakeups; return ret(0); }
        block(t, W_SLEEP);
        t.cpu.r[2] = 0;
        return BLOCKED;
    }},
    {"thbase", 25, "WakeupThread", [] {
        Thread* t = thread((int)arg(0));
        if (!t) return ret((u32)KE_UNKNOWN_THID);
        if (t->st == St::WAIT && t->wait == W_SLEEP) wake_sleeper(*t, 0);
        else ++t->wakeups;
        return ret(0);
    }},
    {"thbase", 26, "iWakeupThread", [] {
        Thread* t = thread((int)arg(0));
        if (!t) return ret((u32)KE_UNKNOWN_THID);
        if (t->st == St::WAIT && t->wait == W_SLEEP) wake_sleeper(*t, 0);
        else ++t->wakeups;
        return ret(0);
    }},
    {"thbase", 34, "GetSystemTime", [] {
        uint64_t c = (uint64_t)(hw_now() * 36864000.0);
        st32(arg(0), (u32)c);
        st32(arg(0) + 4, (u32)(c >> 32));
        return ret(0);
    }},
    {"thbase", 40, "SysClock2USec", [] {
        uint64_t c = ld32(arg(0)) | (uint64_t)ld32(arg(0) + 4) << 32;
        uint64_t us = c * 1000000 / 36864000;
        st32(arg(1), (u32)(us / 1000000));
        st32(arg(2), (u32)(us % 1000000));
        return ret(0);
    }},
    // thsemap
    {"thsemap", 4, "CreateSema", [] {
        u32 p = arg(0);
        Sema s{ld32(p), ld32(p + 4), (int)ld32(p + 8), (int)ld32(p + 12), {}};
        g_semas[++g_sema_ids] = s;
        return ret((u32)g_sema_ids);
    }},
    {"thsemap", 5, "DeleteSema", [] {
        auto it = g_semas.find((int)arg(0));
        if (it == g_semas.end()) return ret((u32)KE_UNKNOWN_SEMID);
        for (int id : it->second.waiters)
            if (Thread* t = thread(id)) wake_sleeper(*t, (u32)-425);   // KE_WAIT_DELETE
        g_semas.erase(it);
        return ret(0);
    }},
    {"thsemap", 6, "SignalSema", [] {
        auto it = g_semas.find((int)arg(0));
        if (it == g_semas.end()) return ret((u32)KE_UNKNOWN_SEMID);
        Sema& s = it->second;
        while (!s.waiters.empty()) {
            Thread* t = thread(s.waiters.front());
            s.waiters.pop_front();
            if (t) { wake_sleeper(*t, 0); return ret(0); }
        }
        if (s.count >= s.max) return ret((u32)KE_SEMA_OVF);
        ++s.count;
        return ret(0);
    }},
    {"thsemap", 8, "WaitSema", [] {
        auto it = g_semas.find((int)arg(0));
        if (it == g_semas.end()) return ret((u32)KE_UNKNOWN_SEMID);
        Thread& t = need_thread("WaitSema");
        Sema& s = it->second;
        if (s.count > 0) { --s.count; return ret(0); }
        s.waiters.push_back(t.id);
        block(t, W_SEMA, it->first);
        return BLOCKED;
    }},
    // thevent
    {"thevent", 4, "CreateEventFlag", [] {
        u32 p = arg(0);
        EventFlag e{ld32(p), ld32(p + 4), ld32(p + 8), ld32(p + 8), {}};
        g_events[++g_event_ids] = e;
        return ret((u32)g_event_ids);
    }},
    {"thevent", 6, "SetEventFlag", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        it->second.bits |= arg(1);
        event_check(it->second);
        return ret(0);
    }},
    {"thevent", 7, "iSetEventFlag", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        it->second.bits |= arg(1);
        event_check(it->second);
        return ret(0);
    }},
    {"thevent", 8, "ClearEventFlag", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        it->second.bits &= arg(1);
        return ret(0);
    }},
    {"thevent", 9, "iClearEventFlag", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        it->second.bits &= arg(1);
        return ret(0);
    }},
    {"thevent", 10, "WaitEventFlag", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        Thread& t = need_thread("WaitEventFlag");
        EventFlag& e = it->second;
        u32 bits = arg(1), mode = arg(2), res = arg(3);
        if (event_match(e, bits, mode)) {
            if (res) st32(res, e.bits);
            if (mode & 0x10) e.bits &= ~bits;
            if (mode & 0x20) e.bits = 0;
            return ret(0);
        }
        t.ev_bits = bits;
        t.ev_mode = mode;
        t.ev_res = res;
        e.waiters.push_back(t.id);
        block(t, W_EVENT, it->first);
        return BLOCKED;
    }},
    {"thevent", 13, "ReferEventFlagStatus", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        EventFlag& e = it->second;
        u32 info[7] = {e.attr, e.option, e.init, e.bits, (u32)e.waiters.size(), 0, 0};
        for (int k = 0; k < 7; ++k) st32(arg(1) + 4 * k, info[k]);
        return ret(0);
    }},
    {"thevent", 14, "iReferEventFlagStatus", [] {
        auto it = g_events.find((int)arg(0));
        if (it == g_events.end()) return ret((u32)KE_UNKNOWN_EVFID);
        EventFlag& e = it->second;
        u32 info[7] = {e.attr, e.option, e.init, e.bits, (u32)e.waiters.size(), 0, 0};
        for (int k = 0; k < 7; ++k) st32(arg(1) + 4 * k, info[k]);
        return ret(0);
    }},
    // timrman: the timer's id is its index + 1
    {"timrman", 4, "AllocHardTimer", [] {
        u32 source = arg(0), size = arg(1);
        // horizontal lines: counter 1 (16 bits) or 3 (32 bits); the system clock: any
        const int order16[] = {1, 0, 2}, order32[] = {3, 4, 5};
        for (int k = 0; k < 3; ++k) {
            int i = size == 32 ? order32[k] : order16[k];
            if ((source & 4) && i != 1 && i != 3) continue;
            if (g_tim[i].used) continue;
            g_tim[i] = HwTimer{};
            g_tim[i].used = true;
            TRACE(1, "[iop] AllocHardTimer(source %u, %u bits, prescale %u) -> counter %d\n", source, size, arg(2), i);
            return ret((u32)i + 1);
        }
        return ret((u32)-150);                        // KE_NO_TIMER
    }},
    {"timrman", 7, "SetTimerMode", [] {
        u32 i = arg(0) - 1;
        if (i >= 6) return ret((u32)-151);
        HwTimer& t = g_tim[i];
        t.mode = arg(1) & 0x3FF;
        t.flags = 0;
        double rate = (t.mode & 0x100) && (i == 1 || i == 3) ? hw_line_rate() : 36864000.0;
        t.period = t.compare ? t.compare / rate : 0;
        t.next = hw_now() + t.period;
        TRACE(1, "[iop] SetTimerMode(counter %u, %X): compare %u, %.3f ms\n", i, t.mode, t.compare, t.period * 1e3);
        return ret(0);
    }},
    {"timrman", 8, "GetTimerStatus", [] {
        u32 i = arg(0) - 1;
        if (i >= 6) return ret(0);
        u32 v = g_tim[i].mode | g_tim[i].flags;
        g_tim[i].flags = 0;
        return ret(v);
    }},
    {"timrman", 11, "SetTimerCompare", [] {
        u32 i = arg(0) - 1;
        if (i >= 6) return ret((u32)-151);
        g_tim[i].compare = arg(1);
        return ret(0);
    }},
    {"timrman", 16, "GetHardTimerIntrCode", [] {
        u32 i = arg(0) - 1;
        return ret(i < 6 ? (u32)TIMER_INTR[i] : (u32)-151);
    }},
    // sifman
    {"sifman", 5, "sceSifInit", [] { return ret(0); }},
    {"sifman", 29, "sceSifCheckInit", [] { return ret(1); }},
    {"sifman", 7, "sceSifSetDma", [] {
        // {IOP source, EE destination, size, attribute} per transfer
        for (u32 k = 0, p = arg(0); k < arg(1); ++k, p += 16) {
            u32 src = ld32(p), dst = ld32(p + 4) & 0x01FFFFFFu, n = ld32(p + 8);
            if (dst + n > EE_RAM_SIZE) rt_die("IOP: SIF DMA to EE %08X + %u", dst, n);
            for (u32 b = 0; b < n; ++b) g_ee_ram[dst + b] = ram()[(src + b) & RAM_MASK];
        }
        return ret(1);
    }},
    {"sifman", 8, "sceSifDmaStat", [] { return ret((u32)-1); }},   // done
    // sifcmd
    {"sifcmd", 14, "sceSifInitRpc", [] { return ret(0); }},
    {"sifcmd", 19, "sceSifSetRpcQueue", [] {
        g_queues[arg(0)].thread = (int)arg(1);
        st32(arg(0), arg(1));
        return ret(0);
    }},
    {"sifcmd", 17, "sceSifRegisterRpc", [] {
        u32 sd = arg(0), sid = arg(1), func = arg(2), buff = arg(3), qd = arg(6);
        st32(sd, sid);
        st32(sd + 4, func);
        st32(sd + 8, buff);
        g_rpc_servers[sd] = Server{sid, sd, func, buff, qd};
        TRACE(1, "[iop] RPC server %08X: function %06X, buffer %06X\n", sid, func, buff);
        iop_server_up(sid, sd, buff);
        return ret(0);
    }},
    {"sifcmd", 22, "sceSifRpcLoop", [] {
        Thread& t = need_thread("sceSifRpcLoop");
        t.rpc_qd = arg(0);
        t.rpc_sp = cpu().r[29];
        t.rpc_gp = cpu().r[28];
        cpu().pc = cpu().r[31];                        // (never returned to)
        cpu().npc = cpu().pc + 4;
        block(t, W_RPC);
        call_dispatch(t);
        return JUMPED;
    }},
    // cdvdman: reads complete at once
    {"cdvdman", 4, "sceCdInit", [] { return ret(1); }},
    {"cdvdman", 6, "sceCdRead", [] {
        u32 lsn = arg(0), n = arg(1), buf = arg(2);
        std::vector<uint8_t> d((size_t)n * 2048);
        if (!disc_read(lsn, n, d.data())) rt_die("IOP: sceCdRead of sectors %u + %u", lsn, n);
        for (size_t k = 0; k < d.size(); ++k) ram()[(buf + k) & RAM_MASK] = d[k];
        TRACE(2, "[iop] sceCdRead %u + %u -> %06X\n", lsn, n, buf);
        return ret(1);
    }},
    {"cdvdman", 8, "sceCdGetError", [] { return ret(0); }},
    {"cdvdman", 10, "sceCdSearchFile", [] {
        std::string name = cstr(arg(1));
        DiscFile f;
        if (!disc_find(name, f) || f.is_dir) {
            TRACE(1, "[iop] sceCdSearchFile %s: not found\n", name.c_str());
            return ret(0);
        }
        u32 p = arg(0);
        st32(p, f.lsn);
        st32(p + 4, f.size);
        for (int k = 0; k < 16; ++k) ram()[(p + 8 + k) & RAM_MASK] = (uint8_t)f.name[k];
        for (int k = 0; k < 8; ++k) ram()[(p + 24 + k) & RAM_MASK] = 0;
        TRACE(1, "[iop] sceCdSearchFile %s: sector %u, %u bytes\n", name.c_str(), f.lsn, f.size);
        return ret(1);
    }},
    {"cdvdman", 11, "sceCdSync", [] { return ret(0); }},
    {"cdvdman", 13, "sceCdDiskReady", [] { return ret(2); }},
    {"cdvdman", 39, "sceCdBreak", [] { return ret(1); }},
};
constexpr u32 HOST_CODE = 0x80000;                     // SYSCALL codes from here name HOST[code - HOST_CODE]
std::atomic<uint64_t> g_host_counts[sizeof HOST / sizeof HOST[0]];
std::vector<std::string> g_unresolved;                 // SYSCALL codes from HOST_CODE + 0x1000

const Host* host_fn(const std::string& lib, int ord) {
    for (const Host& h : HOST)
        if (lib == h.lib && ord == h.ord) return &h;
    return nullptr;
}

// ---- calls into IOP code from the host -----------------------------------------------------
Res host_call(u32 code) {
    ++g_hle_calls;
    if (code >= HOST_CODE + 0x1000 && code < HOST_CODE + 0x1000 + g_unresolved.size())
        rt_die("IOP: %s is not modelled (called from %06X)", g_unresolved[code - HOST_CODE - 0x1000].c_str(),
               cpu().r[31] - 8);
    if (code < HOST_CODE || code >= HOST_CODE + sizeof HOST / sizeof HOST[0])
        rt_die("IOP: SYSCALL %X at %06X", code, cpu().pc - 4);
    const Host& h = HOST[code - HOST_CODE];
    ++g_host_counts[code - HOST_CODE];
    TRACE(3, "[iop] %s.%s(%X, %X, %X, %X)\n", h.lib, h.name, cpu().r[4], cpu().r[5], cpu().r[6], cpu().r[7]);
    u32 ra = cpu().r[31];
    Res r = h.fn();
    if (r != JUMPED) {
        cpu().pc = ra;
        cpu().npc = ra + 4;
    }
    return r;
}

// An interrupt handler, run to its return on the handlers' stack.
void run_handler(Intr& in, int n) {
    IopCpu c;
    c.r[4] = in.arg;
    c.r[28] = in.gp;
    c.r[29] = g_intr_stack - 0x10;
    c.r[31] = MAGIC_INTR;
    c.pc = in.fn;
    c.npc = in.fn + 4;
    IopCpu* outer = g_ctx;
    g_ctx = &c;
    g_in_intr = true;
    ++g_intr_calls;
    ++g_intr_count[n];
    for (int guard = 0;; ++guard) {
        u32 code;
        int ran;
        IopStop s = iop_cpu_run(c, 100000, code, ran);
        g_instructions += ran;
        if (s == IOP_MAGIC) break;
        if (s == IOP_SYSCALL) {
            if (host_call(code) == BLOCKED) rt_die("IOP: interrupt %X's handler blocks", n);
            continue;
        }
        if (guard > 1000) rt_die("IOP: interrupt %X's handler %06X runs on (at %08X)", n, in.fn, c.pc);
    }
    g_in_intr = false;
    g_ctx = outer;
}

void deliver_interrupts() {
    if (g_cur && g_cur->intr_off) return;
    for (int n = 0; n < 0x40; ++n) {
        Intr& in = g_intr[n];
        if (!in.pending || !in.used || !in.enabled) continue;
        in.pending = false;
        run_handler(in, n);
    }
}

// The next call waiting for a thread in sceSifRpcLoop: the server's
// function runs on it and returns to MAGIC_RPC.
void call_dispatch(Thread& t) {
    if (t.st != St::WAIT || t.wait != W_RPC) return;
    Queue& q = g_queues[t.rpc_qd];
    if (q.calls.empty()) return;
    IopRpcCall c = q.calls.front();
    q.calls.pop_front();
    auto it = g_rpc_servers.find(c.sd);
    if (it == g_rpc_servers.end()) rt_die("IOP: a call to server data %06X, which no module registered", c.sd);
    t.call = c;
    t.in_call = true;
    IopCpu& r = t.cpu;
    r.r[4] = c.fno;
    r.r[5] = it->second.buff;
    r.r[6] = c.ssize;
    r.r[28] = t.rpc_gp;
    r.r[29] = t.rpc_sp;
    r.r[31] = MAGIC_RPC;
    r.pc = it->second.func;
    r.npc = r.pc + 4;
    r.ld_reg = 0;
    make_ready(t);
}

void magic(Thread& t, u32 pc) {
    if (pc == MAGIC_EXIT) {
        TRACE(1, "[iop] thread %d returns\n", t.id);
        t.st = St::DORMANT;
    } else if (pc == MAGIC_RPC) {
        u32 data = t.cpu.r[2];
        std::vector<uint8_t> back(t.call.rsize);
        for (u32 k = 0; k < t.call.rsize; ++k) back[k] = ram()[(data + k) & RAM_MASK];
        t.in_call = false;
        iop_rpc_end(t.call, back.data(), back.size());
        block(t, W_RPC);
        call_dispatch(t);
    } else if (pc == MAGIC_START) {
        int res = (int)t.cpu.r[2];
        TRACE(1, "[iop] %s started: %d\n", t.module.c_str(), res);
        u32 id = (u32)iop_module_id();
        std::vector<uint8_t> back(t.call.rsize, 0);
        if (back.size() >= 4) std::memcpy(back.data(), &id, 4);
        iop_rpc_end(t.call, back.data(), back.size());
        t.st = St::DORMANT;
        sys_free(t.stack);
        g_threads.erase(t.id);
        --g_threads_n;
        g_cur = nullptr;
    } else {
        rt_die("IOP: thread %d at %08X", t.id, pc);
    }
}

// ---- modules ------------------------------------------------------------------------------------
void relocate(std::vector<uint8_t>& img, const std::vector<std::pair<u32, int>>& rel, u32 base) {
    auto rd = [&](u32 o) { u32 v; std::memcpy(&v, &img[o], 4); return v; };
    auto wr = [&](u32 o, u32 v) { std::memcpy(&img[o], &v, 4); };
    std::vector<u32> hi;
    for (auto& r : rel) {
        u32 off = r.first, w = rd(off);
        switch (r.second) {
        case 2: w += base; break;                                       // R_MIPS_32
        case 4: w = (w & 0xFC000000u) | ((((w & 0x03FFFFFFu) << 2) + base) >> 2 & 0x03FFFFFFu); break;   // R_MIPS_26
        case 5: hi.push_back(off); continue;                            // R_MIPS_HI16 (paired with the next LO16)
        case 6: {                                                       // R_MIPS_LO16
            int32_t lo = (int16_t)(w & 0xFFFF);
            for (u32 h : hi) {
                u32 hw = rd(h);
                u32 v = ((hw & 0xFFFF) << 16) + (u32)lo + base;
                wr(h, (hw & 0xFFFF0000u) | (((v + 0x8000) >> 16) & 0xFFFF));
            }
            hi.clear();
            w = (w & 0xFFFF0000u) | ((u32)(lo + (int32_t)base) & 0xFFFF);
            break;
        }
        default: continue;
        }
        wr(off, w);
    }
}

// Loaded, linked and ready to start; returns the module's entry and gp.
bool load_module(const std::string& path, const std::vector<uint8_t>& d, Module& m) {
    auto u16at = [&](size_t o) { u16 v = 0; if (o + 2 <= d.size()) std::memcpy(&v, &d[o], 2); return v; };
    auto u32at = [&](size_t o) { u32 v = 0; if (o + 4 <= d.size()) std::memcpy(&v, &d[o], 4); return v; };
    if (d.size() < 52 || std::memcmp(d.data(), "\x7F" "ELF", 4) != 0 || u16at(16) != 0xFF80) return false;
    u32 shoff = u32at(0x20);
    u16 shnum = u16at(0x30), shstrndx = u16at(0x32);
    u32 strs = u32at(shoff + 40u * shstrndx + 16);
    struct Sec { std::string name; u32 type, addr, off, size; };
    std::vector<Sec> secs;
    for (u16 k = 0; k < shnum; ++k) {
        size_t sh = shoff + 40u * k;
        Sec s;
        for (size_t c = strs + u32at(sh); c < d.size() && d[c]; ++c) s.name += (char)d[c];
        s.type = u32at(sh + 4);
        s.addr = u32at(sh + 12);
        s.off = u32at(sh + 16);
        s.size = u32at(sh + 20);
        secs.push_back(s);
    }
    u32 size = 0, entry = 0, gp = 0;
    for (auto& s : secs) {
        if (s.type == 1 || s.type == 8) size = std::max(size, s.addr + s.size);
        if (s.name == ".iopmod") {
            entry = u32at(s.off + 4);
            gp = u32at(s.off + 8);
            m.version = u16at(s.off + 24);
            for (size_t c = s.off + 26; c < d.size() && d[c]; ++c) m.name += (char)d[c];
        }
    }
    std::vector<uint8_t> img(size, 0);
    std::vector<std::pair<u32, int>> rel;
    for (auto& s : secs) {
        if (s.type == 1 && s.name != ".iopmod" && s.size) std::memcpy(&img[s.addr], &d[s.off], s.size);
        if (s.type == 9)
            for (u32 k = 0; k < s.size; k += 8) rel.push_back({u32at(s.off + k), (int)(u32at(s.off + k + 4) & 0xFF)});
    }
    u32 base = sys_alloc(0, size, 0);
    if (!base) rt_die("IOP: no memory for %s (%u bytes)", path.c_str(), size);
    relocate(img, rel, base);
    // Imports: `jr ra` + `addiu zero, zero, ordinal` per stub after the
    // table's header (magic, 0, version, 0, name[8]).
    for (u32 p = 0; p + 20 <= size; p += 4) {
        u32 w0, w1;
        std::memcpy(&w0, &img[p], 4);
        std::memcpy(&w1, &img[p + 4], 4);
        if (w0 != 0x41E00000u || w1 != 0) continue;
        std::string lib;
        for (int k = 0; k < 8 && img[p + 12 + k]; ++k) lib += (char)img[p + 12 + k];
        for (u32 q = p + 20; q + 8 <= size; q += 8) {
            u32 a, b;
            std::memcpy(&a, &img[q], 4);
            std::memcpy(&b, &img[q + 4], 4);
            if (a != 0x03E00008u || (b >> 16) != 0x2400) break;
            int ord = (int)(b & 0xFFFF);
            u32 patch0 = 0;
            auto L = std::find_if(g_libs.begin(), g_libs.end(), [&](const Library& l) { return l.name == lib; });
            if (L != g_libs.end() && ord < (int)L->funcs.size()) {
                patch0 = 0x08000000u | ((L->funcs[ord] >> 2) & 0x03FFFFFFu);      // j
            } else if (const Host* h = host_fn(lib, ord)) {
                patch0 = 0x0000000Cu | ((HOST_CODE + (u32)(h - HOST)) << 6);   // syscall
            } else {
                g_unresolved.push_back(lib + " #" + std::to_string(ord));
                patch0 = 0x0000000Cu | ((HOST_CODE + 0x1000 + (u32)g_unresolved.size() - 1) << 6);
            }
            std::memcpy(&img[q], &patch0, 4);
            u32 nop = 0;
            std::memcpy(&img[q + 4], &nop, 4);
        }
    }
    std::memcpy(ram() + base, img.data(), size);
    m.path = path;
    m.base = base;
    m.size = size;
    m.entry = base + entry;
    m.gp = base + gp;
    return true;
}

void start_module(const std::string& path, const std::vector<uint8_t>& image, const std::vector<uint8_t>& args,
                  const IopRpcCall& call) {
    Module m;
    if (!load_module(path, image, m)) rt_die("IOP: %s is not an IOP module", path.c_str());
    g_modules.push_back(m);
    TRACE(1, "[iop] %s v%X.%02X at %06X-%06X, entry %06X, gp %06X\n", m.name.c_str(), m.version >> 8,
          m.version & 0xFF, m.base, m.base + m.size, m.entry, m.gp);
    // The start function's thread: argv[0] the path, then the arguments
    // (each ending in 0), on its own stack.
    Thread t;
    t.id = ++g_thread_ids;
    t.entry = m.entry;
    t.gp = m.gp;
    t.init_prio = t.prio = 8;
    t.stack_size = 0x1000;
    t.stack = sys_alloc(1, t.stack_size, 0);
    t.loader = true;
    t.module = m.name;
    t.call = call;
    u32 sp = t.stack + t.stack_size - 0x200;
    std::vector<std::string> argv{path};
    for (size_t k = 0; k < args.size();) {
        std::string s;
        while (k < args.size() && args[k]) s += (char)args[k++];
        ++k;
        argv.push_back(s);
    }
    u32 strings = sp + 0x40;
    for (size_t k = 0; k < argv.size() && k < 15; ++k) {
        st32(sp + 4 * (u32)k, strings);
        for (char c : argv[k]) ram()[strings++ & RAM_MASK] = (uint8_t)c;
        ram()[strings++ & RAM_MASK] = 0;
    }
    st32(sp + 4 * (u32)std::min<size_t>(argv.size(), 15), 0);
    t.cpu.r[4] = (u32)std::min<size_t>(argv.size(), 15);
    t.cpu.r[5] = sp;
    t.cpu.r[28] = m.gp;
    t.cpu.r[29] = sp - 0x20;
    t.cpu.r[31] = MAGIC_START;
    t.cpu.pc = m.entry;
    t.cpu.npc = m.entry + 4;
    g_threads[t.id] = t;
    ++g_threads_n;
    make_ready(g_threads[t.id]);
}

// ---- time: the timers' interrupts, the SPU2 ----------------------------------------------------
double next_event(double now) {
    double next = now + 0.001;
    for (int i = 0; i < 6; ++i)
        if (g_tim[i].used && g_tim[i].period > 0 && (g_tim[i].mode & 0x30)) next = std::min(next, g_tim[i].next);
    return next;
}

void advance() {
    double now = hw_now();
    for (int i = 0; i < 6; ++i) {
        HwTimer& t = g_tim[i];
        if (!t.used || t.period <= 0) continue;
        if (now - t.next > 0.25) t.next = now;        // the host fell far behind: no burst of catch-up ticks
        // One tick a pass: when the host made the IOP late, the ticks owed
        // come one at a time, each handler (and the thread it wakes) run
        // before the next, not merged into one interrupt.
        if (now >= t.next) {
            t.flags |= 0x800;                          // the compare value reached
            if (t.mode & 0x10) iopos_raise(TIMER_INTR[i]);
            t.next += t.period;
            if (!(t.mode & 0x08)) t.next = 1e300;      // no reset at the compare value: once
        }
    }
    spu2_advance(now);
}

void handle_requests() {
    std::deque<Request> reqs;
    {
        std::lock_guard<std::mutex> lk(g_req_m);
        reqs.swap(g_reqs);
    }
    for (auto& r : reqs) {
        if (r.kind == Request::LOAD) {
            start_module(r.path, r.image, r.args, r.call);
        } else {
            auto it = g_rpc_servers.find(r.call.sd);
            if (it == g_rpc_servers.end()) rt_die("IOP: a call to server data %06X, which no module registered", r.call.sd);
            Queue& q = g_queues[it->second.qd];
            q.calls.push_back(r.call);
            if (Thread* t = thread(q.thread)) call_dispatch(*t);
        }
    }
}

// The IOP runs at its own speed: 36.864 MHz, taken as one instruction a
// cycle, by the host's clock. Threads that poll (KOEIADS's streaming
// thread, at the lowest priority, spins whenever it has work in hand) then
// take the IOP's time, not a host core.
constexpr double IOP_HZ = 36864000.0;
double g_clock0 = -1;

bool over_budget(double now, double& wait) {
    if (g_clock0 < 0) g_clock0 = now;
    double allowed = (now - g_clock0) * IOP_HZ, done = (double)g_instructions.load();
    if (allowed - done > IOP_HZ * 0.05) g_clock0 = now - (done + IOP_HZ * 0.05) / IOP_HZ;   // a host stall: no burst
    wait = (done - allowed) / IOP_HZ;
    return wait > 0;
}

void iop_main() {
#ifdef _WIN32
    // As the clock's thread (hw.cpp): a condition variable's wait overslept
    // by up to 15 ms here, which merged KESSOUND's 4.2 ms timer ticks (its
    // music played at a third of its speed); Sleep(1), with the timer
    // resolution at 1 ms, and a spin through the last millisecond. Above
    // the GS's drawing threads, so that the IOP keeps its time.
    timeBeginPeriod(1);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    g_intr_stack = sys_alloc(1, 0x2000, 0) + 0x2000;
    for (;;) {
        handle_requests();
        advance();
        deliver_interrupts();
        Thread* t = pick();
        double now = hw_now(), ahead = 0;
        if (!t || over_budget(now, ahead)) {
            double wait = std::min(next_event(now) - now, t ? ahead : 1.0);
            bool requests;
            {
                std::lock_guard<std::mutex> lk(g_req_m);
                requests = !g_reqs.empty();
            }
            if (requests || wait <= 0 || spu2_behind(now)) continue;
#ifdef _WIN32
            if (wait > 0.0015) Sleep(1);
            else std::this_thread::yield();
#else
            std::this_thread::sleep_for(std::chrono::duration<double>(std::min(wait, 0.001)));
#endif
            continue;
        }
        g_cur = t;
        g_ctx = &t->cpu;
        // Run until it blocks, a slice ends, or a host function changed who should run.
        for (int slice = 0; slice < 64; ++slice) {
            u32 code;
            int ran;
            IopStop s = iop_cpu_run(t->cpu, 2000, code, ran);
            g_instructions += ran;
            if (s == IOP_BUDGET) {
                if (slice % 8 == 7) break;             // time moves on: interrupts, requests
                continue;
            }
            if (s == IOP_MAGIC) { magic(*t, t->cpu.pc); break; }
            host_call(code);
            break;
        }
        g_cur = nullptr;
        g_ctx = nullptr;
    }
}

bool enabled_from_env() {
    const char* e = std::getenv("PS2KIT_IOP_SOUND");
    return !(e && *e == '0');
}

}  // namespace

// ---- the IOP's address space ---------------------------------------------------------------------
uint32_t iop_read(uint32_t a, int size) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x1F800000 && p < 0x1F800400) {
        uint32_t v = 0;
        std::memcpy(&v, g_scratch + (p & 0x3FF), size);
        return v;
    }
    if (p >= 0x1F900000 && p < 0x1F900C00) {
        uint32_t o = p - 0x1F900000;
        if (size == 4) return spu2_read(o) | (uint32_t)spu2_read(o + 2) << 16;
        return spu2_read(o & ~1u) >> ((p & 1) * 8);
    }
    if (p >= 0x1F801000 && p < 0x1F802000) {
        uint32_t w = g_io[(p - 0x1F801000) >> 2];
        return w >> ((p & 3) * 8);
    }
    if (g_io_seen.insert(p).second) std::fprintf(stderr, "ps2kit: IOP: read of %08X (%d bytes): 0\n", a, size);
    return 0;
}

void iop_write(uint32_t a, uint32_t v, int size) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x1F800000 && p < 0x1F800400) { std::memcpy(g_scratch + (p & 0x3FF), &v, size); return; }
    if (p >= 0x1F900000 && p < 0x1F900C00) {
        uint32_t o = p - 0x1F900000;
        if (size == 1) rt_die("IOP: a byte write to the SPU2 (%08X)", a);
        spu2_write(o, (uint16_t)v);
        if (size == 4) spu2_write(o + 2, (uint16_t)(v >> 16));
        return;
    }
    if (p >= 0x1F801000 && p < 0x1F802000) {
        uint32_t& w = g_io[(p - 0x1F801000) >> 2];
        uint32_t sh = (p & 3) * 8, m = size == 4 ? 0xFFFFFFFFu : size == 2 ? 0xFFFFu : 0xFFu;
        w = (w & ~(m << sh)) | ((v & m) << sh);
        // DMA channel 4 (SPU2 core 0) at 0x10C0 and 7 (core 1) at 0x1500:
        // MADR, BCR (block size in words, blocks), CHCR (bit 0 from memory,
        // bit 24 start). The transfer is done at once, then its interrupt.
        uint32_t base = (p & ~0xFu);
        if ((base == 0x1F8010C0 || base == 0x1F801500) && (p & 0xF) >= 8 && (w & 0x01000000)) {
            int core = base == 0x1F8010C0 ? 0 : 1;
            uint32_t madr = g_io[(base - 0x1F801000) >> 2] & 0xFFFFFF, bcr = g_io[(base + 4 - 0x1F801000) >> 2];
            uint32_t words = (bcr & 0xFFFF) * std::max<uint32_t>(1, bcr >> 16);
            spu2_dma(core, madr, words, w & 1);
            w &= ~0x01000000u;
            iopos_raise(core ? 0x28 : 0x24);
        }
        return;
    }
    if (g_io_seen.insert(p).second) std::fprintf(stderr, "ps2kit: IOP: write of %08X (%d bytes) = %X: ignored\n", a, size, v);
}

// ---- the interface ---------------------------------------------------------------------------
bool iopos_enabled() {
    static const bool on = enabled_from_env();
    return on;
}

bool iopos_runs(const std::string& module) {
    if (!iopos_enabled()) return false;
    static const char* RUN[] = {"LIBSD", "MODHSYN", "MODMIDI", "MODMSIN", "KOEIADS", "KESSOUND"};
    for (const char* r : RUN)
        if (module == r) return true;
    return false;
}

void iopos_init() {
    spu2_init();
    g_thread = std::thread(iop_main);
    g_thread.detach();
}

void iopos_load(const std::string& path, const std::vector<uint8_t>& image, const std::vector<uint8_t>& args,
                const IopRpcCall& call) {
    {
        std::lock_guard<std::mutex> lk(g_req_m);
        g_reqs.push_back(Request{Request::LOAD, call, path, image, args});
    }
    g_req_cv.notify_one();
}

void iopos_call(const IopRpcCall& call) {
    {
        std::lock_guard<std::mutex> lk(g_req_m);
        g_reqs.push_back(Request{Request::CALL, call, {}, {}, {}});
    }
    g_req_cv.notify_one();
}

void iopos_raise(int intr) {
    if (intr >= 0 && intr < 0x40) g_intr[intr].pending = true;
}

std::string iopos_status() {
    // the three host functions called most
    std::vector<std::pair<uint64_t, int>> top;
    for (size_t k = 0; k < sizeof HOST / sizeof HOST[0]; ++k)
        if (g_host_counts[k]) top.push_back({g_host_counts[k].load(), (int)k});
    std::sort(top.rbegin(), top.rend());
    std::string most;
    for (size_t k = 0; k < top.size() && k < 3; ++k)
        most += std::string(k ? ", " : " (") + HOST[top[k].second].name + " " + std::to_string(top[k].first);
    if (!most.empty()) most += ")";
    std::string intr;
    for (int n = 0; n < 0x40; ++n)
        if (g_intr_count[n]) intr += (intr.empty() ? "" : " ") + std::to_string(n) + ":" + std::to_string(g_intr_count[n].load());
    return "IOP " + std::to_string(g_instructions.load() / 1000000) + "M instructions, " +
           std::to_string(g_hle_calls.load()) + " kernel calls" + most + ", interrupts by line " + intr + ", " +
           std::to_string(g_threads_n.load()) + " threads; " + spu2_status();
}

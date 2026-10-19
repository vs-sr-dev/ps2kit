// ps2kit runtime — the EE kernel at the syscall boundary.
//
// The game's SDK libraries run recompiled and unchanged; the kernel below
// them (the BIOS's, reached through `syscall` with the number in v1) is
// replaced here, as far as the game asks. The numbers and names are the
// SDK's own (klib.s); a negative number is the interrupt-context form of the
// same call (iSignalSema is -67: the kernel's table entry 67 again, and
// 26-31 are the interrupt-context forms of 20-25), so the kernel works on
// the absolute value and only notes that it must not switch threads.
//
// Threads. Every guest thread runs on a host thread of its own, with its own
// registers (EEContext) and its own host stack for the recompiled call
// chain; one runs at a time. The thread that runs holds the baton (g_cur);
// a switch hands it on and parks the host thread until the baton comes
// back. Scheduling is the EE kernel's: strict priority (0 highest), first in
// first out within a priority, no time slices; a thread that is preempted
// goes back to the head of its queue. VU0's registers belong to the unit,
// not to a thread, so they travel with the baton.
//
// Interrupts. The hardware raises bits in INTC_STAT and D_STAT (hw.cpp) and
// alarms fall due on the host clock; recompiled code polls at loop
// back-edges (EE_POLL) and when it sets Status.EIE. Delivery does what the
// kernel's dispatcher does: take the pending causes, clear them, call each
// registered handler with the cause (or the DMA channel) in a0 on a stack of
// the kernel's, with the gp the handler was registered under; handlers do
// not nest. When no thread can run, the baton holder waits for interrupts
// itself, as the kernel's idle loop does.
//
// What is not modelled is refused with a message, never guessed silently.
#include "os.h"

#include <algorithm>
#include <condition_variable>
#include <cstdarg>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef STACK_SIZE_PARAM_IS_A_STACK_SIZE
#define STACK_SIZE_PARAM_IS_A_STACK_SIZE 0x00010000
#endif
#else
#include <pthread.h>
#endif

std::atomic<uint32_t> g_ee_pending{0};
int g_trace = 0;
static const char* g_kernel_argv0 = "";

namespace {

// ---- guest memory as the kernel reads it --------------------------------------------------
uint32_t ld32(uint32_t a) { return ee_ld32(a); }
void st32(uint32_t a, uint32_t v) { ee_st32(a, v); }
std::string cstr(uint32_t a, size_t max = 4096) {
    std::string s;
    for (size_t k = 0; k < max; ++k) {
        char ch = (char)ee_ld8(a + (uint32_t)k);
        if (!ch) break;
        s += ch;
    }
    return s;
}

void ret(EEContext& c, int32_t v) { c.r[2].ud[0] = S32(v); }
uint32_t arg(EEContext& c, int k) { return c.r[4 + k].uw[0]; }
int32_t sarg(EEContext& c, int k) { return c.r[4 + k].sw[0]; }

constexpr uint32_t ST_IE = 0x1, ST_EIE = 0x10000;
constexpr uint32_t INTR_STACK_TOP = 0x00080000;   // in the kernel's own 512 KB, which nothing loads
constexpr uint32_t THREAD_CONTEXT_SIZE = 0x2A0;   // a saved context, kept at the top of a thread's stack

// ---- threads --------------------------------------------------------------------------------
enum { THS_RUN = 1, THS_READY = 2, THS_WAIT = 4, THS_SUSPEND = 8, THS_WAITSUSPEND = 12, THS_DORMANT = 16 };
enum { WAIT_NONE = 0, WAIT_SLEEP = 1, WAIT_SEMA = 2 };
constexpr int MAX_THREADS = 256, MAX_SEMAS = 256;

struct ThreadExit {};

struct Host {                          // a host thread carrying a guest thread
    std::condition_variable cv;
    bool zombie = false;               // its guest thread was terminated: unwind and end
};

struct Thread {
    int id = 0;
    bool used = false;
    int status = THS_DORMANT;
    uint32_t func = 0, stack = 0, stack_size = 0, gp = 0, attr = 0, option = 0;
    int init_prio = 0, prio = 0;
    int wait_type = WAIT_NONE, wait_id = 0, wakeup = 0;
    int32_t wait_result = 0;
    int64_t seq = 0;                   // place in the ready queue of its priority
    EEContext c{};
    Host* host = nullptr;
};

struct Sema {
    bool used = false;
    int count = 0, max = 0, init = 0;
    uint32_t attr = 0, option = 0;
    std::deque<int> waiters;
};

struct Handler { int id; uint32_t fn, gp, arg; bool enabled; };
struct Alarm { int id; double due; uint32_t fn, gp, arg, ticks; };

std::mutex g_k;                        // the kernel's tables and the baton
std::condition_variable g_idle_cv;
Thread g_th[MAX_THREADS];
Sema g_sema[MAX_SEMAS];
Thread* g_cur = nullptr;               // holds the baton
int64_t g_tail = 0, g_head = 0;
bool g_in_intr = false;
std::vector<Handler> g_intc_h[15], g_dmac_h[10];
int g_handler_ids = 0;
std::vector<Alarm> g_alarms;
int g_alarm_ids = 0;
uint32_t g_heap_end = 0;
uint32_t g_vsync_flag = 0, g_vsync_csr = 0;
uint32_t g_sreg[32];                   // the SIF software registers
double g_idle_limit = 10.0;            // seconds with no runnable thread before giving up

thread_local Thread* t_thread = nullptr;
thread_local Host* t_host = nullptr;

Thread* thread_of(int id) {
    if (id == 0) return t_thread;      // TH_SELF
    if (id < 0 || id >= MAX_THREADS || !g_th[id].used) return nullptr;
    return &g_th[id];
}

Thread* best_ready() {
    Thread* b = nullptr;
    for (auto& t : g_th)
        if (t.used && t.status == THS_READY && (!b || t.prio < b->prio || (t.prio == b->prio && t.seq < b->seq)))
            b = &t;
    return b;
}

void make_ready(Thread* t) {
    t->status = THS_READY;
    t->wait_type = WAIT_NONE;
    t->wait_id = 0;
    t->seq = ++g_tail;
}

void carry_vu0(const EEContext& from, EEContext& to) {
    if (&from == &to) return;
    std::memcpy(to.vf, from.vf, sizeof to.vf);
    to.vacc = from.vacc;
    std::memcpy(to.vi, from.vi, sizeof to.vi);
}

void host_main(Thread* t, Host* h);

struct Start { Thread* t; Host* h; };
#ifdef _WIN32
DWORD WINAPI host_entry(LPVOID p) {
    Start s = *static_cast<Start*>(p);
    delete static_cast<Start*>(p);
    host_main(s.t, s.h);
    return 0;
}
void spawn(Thread* t) {
    t->host = new Host;
    // recompiled code nests C++ calls as deeply as the guest nests its own
    HANDLE th = CreateThread(nullptr, 64u << 20, host_entry, new Start{t, t->host},
                             STACK_SIZE_PARAM_IS_A_STACK_SIZE, nullptr);
    if (!th) rt_die("cannot create a host thread");
    CloseHandle(th);
}
#else
void* host_entry(void* p) {
    Start s = *static_cast<Start*>(p);
    delete static_cast<Start*>(p);
    host_main(s.t, s.h);
    return nullptr;
}
void spawn(Thread* t) {
    t->host = new Host;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, 64u << 20);
    pthread_t th;
    if (pthread_create(&th, &a, host_entry, new Start{t, t->host})) rt_die("cannot create a host thread");
    pthread_detach(th);
}
#endif

void wait_baton(std::unique_lock<std::mutex>& lk) {
    for (;;) {
        if (t_host->zombie) throw ThreadExit{};
        if (g_cur == t_thread && t_thread->host == t_host) return;
        t_host->cv.wait(lk);
    }
}

// The baton to `to`; then, unless this host thread is ending, wait to get it back.
void pass(std::unique_lock<std::mutex>& lk, Thread* to, bool wait) {
    if (g_cur) carry_vu0(g_cur->c, to->c);
    g_cur = to;
    to->status = THS_RUN;
    if (!to->host) spawn(to);
    else to->host->cv.notify_one();
    if (wait) wait_baton(lk);
}

void deliver();

void dump_threads() {
    std::fprintf(stderr, "threads:\n");
    for (auto& t : g_th) {
        if (!t.used) continue;
        std::fprintf(stderr, "  %3d prio %3d status %2d wait %d/%d wakeups %d func %08X (%s)%s\n", t.id, t.prio,
                     t.status, t.wait_type, t.wait_id, t.wakeup, t.func, rt_name_of(t.func),
                     &t == g_cur ? "  <- baton" : "");
    }
    for (int k = 0; k < MAX_SEMAS; ++k)
        if (g_sema[k].used && !g_sema[k].waiters.empty())
            std::fprintf(stderr, "  sema %d count %d: %zu waiting\n", k, g_sema[k].count, g_sema[k].waiters.size());
}

// Nothing can run: wait for an interrupt and deliver it on this host thread.
void idle(std::unique_lock<std::mutex>& lk) {
    double since = hw_now();
    while (!g_ee_pending.load(std::memory_order_relaxed)) {
        g_idle_cv.wait_for(lk, std::chrono::milliseconds(100));
        if (hw_now() - since > g_idle_limit) {
            std::fprintf(stderr, "ps2kit: no thread has been able to run for %.0f s\n", g_idle_limit);
            dump_threads();
            std::fflush(stdout);
            std::_Exit(3);
        }
    }
    lk.unlock();
    deliver();
    lk.lock();
}

// After a change of state: let the best thread run (the caller, if it still may).
void schedule(std::unique_lock<std::mutex>& lk) {
    Thread* me = t_thread;
    for (;;) {
        Thread* b = best_ready();
        if (me->status == THS_RUN && (!b || b->prio >= me->prio)) return;
        if (b == me) { me->status = THS_RUN; return; }
        if (b) {
            if (me->status == THS_RUN) {
                me->status = THS_READY;
                me->seq = --g_head;
            }
            pass(lk, b, true);
            return;
        }
        idle(lk);
    }
}

// The host thread's guest thread has ended (ExitThread, or its function returned).
void leave(std::unique_lock<std::mutex>& lk) {
    for (;;) {
        Thread* b = best_ready();
        if (b) { pass(lk, b, false); return; }
        idle(lk);
    }
}

void host_main(Thread* t, Host* h) {
    rt_init_fpu();
    t_thread = t;
    t_host = h;
    try {
        {
            std::unique_lock<std::mutex> lk(g_k);
            wait_baton(lk);
        }
        ee_call(t->c, t->func);
        TRACE(1, "[sys] thread %d returned from %08X\n", t->id, t->func);
        std::unique_lock<std::mutex> lk(g_k);
        t->status = THS_DORMANT;
    } catch (ThreadExit&) {
    }
    std::unique_lock<std::mutex> lk(g_k);
    if (!h->zombie) {
        t->host = nullptr;
        leave(lk);
    }
    delete h;
}

// ---- interrupts -------------------------------------------------------------------------------
uint32_t call_guest(uint32_t fn, uint32_t gp, uint32_t sp, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                    EEContext& vu) {
    EEContext h{};
    h.r[4].ud[0] = S32(a0);
    h.r[5].ud[0] = S32(a1);
    h.r[6].ud[0] = S32(a2);
    h.r[7].ud[0] = S32(a3);
    h.r[28].ud[0] = S32(gp);
    h.r[29].ud[0] = S32(sp);
    h.cop0[12] = 0;                    // interrupts off inside handlers
    carry_vu0(vu, h);
    ee_call(h, fn);
    carry_vu0(h, vu);
    return h.r[2].uw[0];
}

void deliver() {
    g_ee_pending.store(0, std::memory_order_relaxed);
    if (g_in_intr) return;
    EEContext& vu = g_cur->c;
    for (int round = 0; round < 16; ++round) {
        uint32_t intc = hw_intc_take();
        uint32_t dmac = hw_dmac_take();
        std::vector<Alarm> due;
        std::vector<Handler> hs;
        {
            std::lock_guard<std::mutex> lk(g_k);
            double now = hw_now(), next = 1e300;
            for (size_t k = 0; k < g_alarms.size();) {
                if (g_alarms[k].due <= now) {
                    due.push_back(g_alarms[k]);
                    g_alarms.erase(g_alarms.begin() + (long)k);
                } else {
                    next = std::min(next, g_alarms[k].due);
                    ++k;
                }
            }
            if (next < 1e300) hw_set_deadline(next);
            if ((intc & (1u << INTC_VBON)) && g_vsync_flag) {
                st32(g_vsync_flag, 1);
                uint64_t csr = hw_gs_csr();
                ee_st64(g_vsync_csr, csr);
            }
        }
        if (!intc && !dmac && due.empty()) return;
        g_in_intr = true;
        for (int cause = 0; cause < 15; ++cause) {
            if (!(intc & (1u << cause))) continue;
            {
                std::lock_guard<std::mutex> lk(g_k);
                hs = g_intc_h[cause];
            }
            for (auto& h : hs)
                if (h.enabled) call_guest(h.fn, h.gp, INTR_STACK_TOP - 0x10, (uint32_t)cause, h.arg, 0, 0, vu);
        }
        for (int ch = 0; ch < 10; ++ch) {
            if (!(dmac & (1u << ch))) continue;
            {
                std::lock_guard<std::mutex> lk(g_k);
                hs = g_dmac_h[ch];
            }
            for (auto& h : hs)
                if (h.enabled) call_guest(h.fn, h.gp, INTR_STACK_TOP - 0x10, (uint32_t)ch, h.arg, 0, 0, vu);
        }
        for (auto& a : due) {
            TRACE(1, "[irq] alarm %d -> %08X (%s)\n", a.id, a.fn, rt_name_of(a.fn));
            call_guest(a.fn, a.gp, INTR_STACK_TOP - 0x10, (uint32_t)a.id, a.ticks & 0xFFFF, a.arg, 0, vu);
        }
        g_in_intr = false;
    }
}

int add_handler(std::vector<Handler>& list, uint32_t fn, int next, uint32_t gp, uint32_t a) {
    Handler h{++g_handler_ids, fn, gp, a, true};
    // `next` places the handler in the chain; 0 is taken as "last". The
    // order only matters with two handlers on one cause, which is said.
    if (!list.empty())
        std::fprintf(stderr, "ps2kit: handler %08X joins %zu other(s) on one cause (next = %d)\n", fn, list.size(), next);
    if (next == 0) list.push_back(h);
    else list.insert(list.begin(), h);
    return h.id;
}

int remove_handler(std::vector<Handler>& list, int id) {
    for (size_t k = 0; k < list.size(); ++k)
        if (list[k].id == id) {
            list.erase(list.begin() + (long)k);
            return 0;
        }
    return -1;
}

// ---- Deci2: the devkit's debug channel, as the SDK's TTY uses it ------------------------------
// sceTtyInit opens a socket of protocol 0x210 with a handler; sceTtyWrite
// builds a packet (a 12-byte header, then the text with \n as \r\n), asks
// for it to be sent (ReqSend) and polls until its handler has been called
// with event 3 (it hands the bytes over with ExSend) and event 4 (done).
// The host is the far end: the text goes to the console.
struct Deci2Socket { bool used; uint32_t protocol, opt, handler, gp; bool send_pending; uint32_t skip; };
Deci2Socket g_deci2[4];

void console_write(const char* p, size_t n) {
    for (size_t k = 0; k < n; ++k)
        if (p[k] != '\r') std::fputc(p[k], stdout);
    std::fflush(stdout);
}

int32_t deci2(EEContext& c, int32_t fn, uint32_t param) {
    switch (fn) {
    case 1: {                                        // open (protocol, opt, handler)
        for (int s = 1; s < 4; ++s) {
            if (g_deci2[s].used) continue;
            g_deci2[s] = {true, ld32(param) & 0xFFFF, ld32(param + 4), ld32(param + 8), c.r[28].uw[0], false, 0};
            TRACE(1, "[deci2] open protocol %X handler %08X -> %d\n", g_deci2[s].protocol, g_deci2[s].handler, s);
            return s;
        }
        return -1;
    }
    case 2: {                                        // close
        uint32_t s = ld32(param);
        if (s < 4) g_deci2[s].used = false;
        return 0;
    }
    case 3: {                                        // reqsend (socket, destination)
        uint32_t s = ld32(param);
        if (s >= 4 || !g_deci2[s].used) return -1;
        g_deci2[s].send_pending = true;
        g_deci2[s].skip = 12;                        // the packet's header
        return 0;
    }
    case 4: {                                        // poll: the far end is always ready
        uint32_t s = ld32(param);
        if (s >= 4 || !g_deci2[s].used || !g_deci2[s].send_pending) return 0;
        Deci2Socket& d = g_deci2[s];
        d.send_pending = false;
        uint32_t sp = (c.r[29].uw[0] - 0x400) & ~15u;
        call_guest(d.handler, d.gp, sp, 3, 0, d.opt, 0, c);       // write
        call_guest(d.handler, d.gp, sp, 4, 0, d.opt, 0, c);       // write done
        return 0;
    }
    case 5: case -5:                                 // exrecv: nothing comes from the host
        return 0;
    case 6: case -6: {                               // exsend (socket, buffer, length)
        uint32_t s = ld32(param), buf = ld32(param + 4), len = ld32(param + 8) & 0xFFFF;
        if (s >= 4 || !g_deci2[s].used) return -1;
        std::string out;
        for (uint32_t k = 0; k < len; ++k) {
            if (g_deci2[s].skip) { --g_deci2[s].skip; continue; }
            out += (char)ee_ld8(buf + k);
        }
        console_write(out.data(), out.size());
        return (int32_t)len;
    }
    case 0x10: {                                     // kputs (string)
        std::string s = cstr(ld32(param));
        console_write(s.data(), s.size());
        return 0;
    }
    }
    std::fprintf(stderr, "ps2kit: Deci2Call %d not modelled: answered -1\n", fn);
    return -1;
}

// ---- the IOP side, until phase 3 -------------------------------------------------------------
// A loop that reads the same SIF register and gets the same value is the EE
// waiting on the IOP. Until the IOP side exists, that is where the run ends.
struct { int reg = 0; uint32_t value = 0, ra = 0; double since = 0; long n = 0; } g_sifwait;

void sif_wait_check(EEContext& c, int reg, uint32_t v) {
    uint32_t ra = c.r[31].uw[0];
    if (g_sifwait.reg == reg && g_sifwait.value == v && g_sifwait.ra == ra) {
        if (++g_sifwait.n > 1000 && hw_now() - g_sifwait.since > 2.0) {
            std::fflush(stdout);
            std::fprintf(stderr, "ps2kit: the EE waits on the IOP: SIF register %X stays %08X, read from %08X (%s)\n",
                         (unsigned)reg, v, ra, rt_name_of(ra));
            std::_Exit(4);
        }
        return;
    }
    g_sifwait.reg = reg;
    g_sifwait.value = v;
    g_sifwait.ra = ra;
    g_sifwait.since = hw_now();
    g_sifwait.n = 0;
}

int g_sif_dma_id = 0;

// ---- names, for traces --------------------------------------------------------------------------
const char* sys_name(int n) {
    static const struct { int n; const char* s; } names[] = {
        {2, "SetGsCrt"}, {4, "Exit"}, {16, "AddIntcHandler"}, {17, "RemoveIntcHandler"},
        {18, "AddDmacHandler"}, {19, "RemoveDmacHandler"}, {20, "_EnableIntc"}, {21, "_DisableIntc"},
        {22, "_EnableDmac"}, {23, "_DisableDmac"}, {24, "_SetAlarm"}, {25, "_ReleaseAlarm"},
        {26, "_iEnableIntc"}, {27, "_iDisableIntc"}, {28, "_iEnableDmac"}, {29, "_iDisableDmac"},
        {30, "_iSetAlarm"}, {31, "_iReleaseAlarm"}, {32, "CreateThread"}, {33, "DeleteThread"},
        {34, "StartThread"}, {35, "ExitThread"}, {36, "ExitDeleteThread"}, {37, "TerminateThread"},
        {38, "iTerminateThread"}, {41, "ChangeThreadPriority"}, {42, "iChangeThreadPriority"},
        {43, "RotateThreadReadyQueue"}, {44, "iRotateThreadReadyQueue"}, {47, "GetThreadId"},
        {48, "ReferThreadStatus"}, {49, "iReferThreadStatus"}, {50, "SleepThread"}, {51, "WakeupThread"},
        {52, "iWakeupThread"}, {53, "CancelWakeupThread"}, {54, "iCancelWakeupThread"},
        {55, "SuspendThread"}, {56, "iSuspendThread"}, {57, "ResumeThread"}, {58, "iResumeThread"},
        {60, "SetupThread"}, {61, "SetupHeap"}, {62, "EndOfHeap"}, {64, "CreateSema"}, {65, "DeleteSema"},
        {66, "SignalSema"}, {67, "iSignalSema"}, {68, "WaitSema"}, {69, "PollSema"}, {70, "iPollSema"},
        {71, "ReferSemaStatus"}, {72, "iReferSemaStatus"}, {100, "FlushCache"}, {104, "iFlushCache"},
        {107, "sceSifStopDma"}, {112, "GsGetIMR"}, {113, "GsPutIMR"}, {115, "SetVSyncFlag"},
        {118, "sceSifDmaStat"}, {119, "sceSifSetDma"}, {120, "sceSifSetDChain"}, {121, "sceSifSetReg"},
        {122, "sceSifGetReg"}, {124, "Deci2Call"},
    };
    for (auto& e : names)
        if (e.n == n) return e.s;
    return "?";
}

// Consecutive identical trace lines are counted, not repeated.
std::string g_last_line;
long g_last_repeats = 0;
void trace_line(const std::string& s) {
    if (s == g_last_line) { ++g_last_repeats; return; }
    if (g_last_repeats) std::fprintf(stderr, "      (the line above %ld more times)\n", g_last_repeats);
    g_last_line = s;
    g_last_repeats = 0;
    std::fprintf(stderr, "%s\n", s.c_str());
}

}  // namespace

// ---- the system calls ------------------------------------------------------------------------------
void ee_syscall(EEContext& c, uint32_t pc) {
    c.pc = pc;
    int32_t raw = c.r[3].sw[0];
    int n = raw < 0 ? -raw : raw;
    bool intr = raw < 0 || g_in_intr;
    char line[256];
    if (g_trace >= 1) {
        uint32_t ra = c.r[31].uw[0];
        std::snprintf(line, sizeof line, "[sys] %-22s %08X %08X %08X %08X  from %08X %s", sys_name(n),
                      arg(c, 0), arg(c, 1), arg(c, 2), arg(c, 3), ra, rt_name_of(ra));
    }
    if (n != 122) g_sifwait.reg = 0;

    std::unique_lock<std::mutex> lk(g_k);
    Thread* me = t_thread;
    switch (n) {
    // -- the boot ---------------------------------------------------------------------------
    case 60: {                                       // SetupThread (gp, stack, stack size, args, root)
        uint32_t gp = arg(c, 0), stack = arg(c, 1), size = arg(c, 2), args = arg(c, 3);
        if (stack == 0xFFFFFFFFu) stack = EE_RAM_SIZE - size;   // the SDK's GNU crt0 asks for the top of RAM
        me->gp = gp;
        me->stack = stack;
        me->stack_size = size;
        if (args) {
            // argc, argv[16], then the strings; the boot ROM passes the
            // executable's own path as argv[0].
            const char* argv0 = g_kernel_argv0;
            uint32_t strings = args + 4 + 16 * 4;
            st32(args, 1);
            st32(args + 4, strings);
            size_t len = std::strlen(argv0);
            for (size_t k = 0; k <= len; ++k) ee_st8(strings + (uint32_t)k, (uint8_t)argv0[k]);
        }
        ret(c, (int32_t)(stack + size - THREAD_CONTEXT_SIZE));
        break;
    }
    case 61: {                                       // SetupHeap (start, size)
        uint32_t start = arg(c, 0);
        int32_t size = sarg(c, 1);
        g_heap_end = size < 0 ? me->stack : start + (uint32_t)size;
        ret(c, (int32_t)g_heap_end);
        break;
    }
    case 62: ret(c, (int32_t)g_heap_end); break;     // EndOfHeap
    case 4:                                          // Exit
        lk.unlock();
        std::fflush(stdout);
        std::fprintf(stderr, "ps2kit: Exit(%d) from %08X (%s)\n", sarg(c, 0), c.r[31].uw[0], rt_name_of(c.r[31].uw[0]));
        std::_Exit(sarg(c, 0) & 0xFF);

    // -- threads ---------------------------------------------------------------------------------
    case 32: {                                       // CreateThread (param)
        uint32_t p = arg(c, 0);
        int id = -1;
        for (int k = 1; k < MAX_THREADS; ++k)
            if (!g_th[k].used) { id = k; break; }
        if (id < 0) { ret(c, -1); break; }
        Thread& t = g_th[id];
        t = Thread{};
        t.id = id;
        t.used = true;
        t.status = THS_DORMANT;
        t.func = ld32(p + 0x04);
        t.stack = ld32(p + 0x08);
        t.stack_size = ld32(p + 0x0C);
        t.gp = ld32(p + 0x10);
        t.init_prio = t.prio = (int)ld32(p + 0x14);
        t.attr = ld32(p + 0x1C);
        t.option = ld32(p + 0x20);
        ret(c, id);
        break;
    }
    case 33: {                                       // DeleteThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t == me || t->status != THS_DORMANT) { ret(c, -1); break; }
        t->used = false;
        ret(c, t->id);
        break;
    }
    case 34: {                                       // StartThread (id, arg)
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t->status != THS_DORMANT) { ret(c, -1); break; }
        t->c = EEContext{};
        t->c.r[4].ud[0] = S32(arg(c, 1));
        t->c.r[28].ud[0] = S32(t->gp);
        t->c.r[29].ud[0] = S32(t->stack + t->stack_size - THREAD_CONTEXT_SIZE);
        t->c.cop0[12] = ST_IE | ST_EIE;
        t->prio = t->init_prio;
        t->wakeup = 0;
        make_ready(t);
        ret(c, t->id);
        if (!intr) schedule(lk);
        break;
    }
    case 35:                                         // ExitThread
        me->status = THS_DORMANT;
        lk.unlock();
        if (g_trace >= 1) trace_line(line);
        throw ThreadExit{};
    case 37: case 38: {                              // TerminateThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t == me) { ret(c, -1); break; }
        if (t->wait_type == WAIT_SEMA) {
            auto& w = g_sema[t->wait_id].waiters;
            w.erase(std::remove(w.begin(), w.end(), t->id), w.end());
        }
        t->status = THS_DORMANT;
        t->wait_type = WAIT_NONE;
        if (t->host) {
            t->host->zombie = true;
            t->host->cv.notify_one();
            t->host = nullptr;
        }
        ret(c, t->id);
        break;
    }
    case 41: case 42: {                              // ChangeThreadPriority (id, priority)
        Thread* t = thread_of(sarg(c, 0));
        int p = sarg(c, 1);
        if (!t || p < 0 || p > 127) { ret(c, -1); break; }
        int old = t->prio;
        t->prio = p;
        if (t->status == THS_READY) t->seq = ++g_tail;
        ret(c, old);
        if (!intr) schedule(lk);
        break;
    }
    case 43: case 44: {                              // RotateThreadReadyQueue (priority)
        int p = sarg(c, 0);
        if (!intr && me->status == THS_RUN && me->prio == p) {
            me->status = THS_READY;
            me->seq = ++g_tail;
            ret(c, p);
            schedule(lk);
            break;
        }
        Thread* first = nullptr;
        for (auto& t : g_th)
            if (t.used && t.status == THS_READY && t.prio == p && (!first || t.seq < first->seq)) first = &t;
        if (first) first->seq = ++g_tail;
        ret(c, p);
        break;
    }
    case 47: ret(c, me->id); break;                  // GetThreadId
    case 48: case 49: {                              // ReferThreadStatus (id, info)
        Thread* t = thread_of(sarg(c, 0));
        uint32_t p = arg(c, 1);
        if (!t) { ret(c, -1); break; }
        if (p) {
            st32(p + 0x00, (uint32_t)t->status);
            st32(p + 0x04, t->func);
            st32(p + 0x08, t->stack);
            st32(p + 0x0C, t->stack_size);
            st32(p + 0x10, t->gp);
            st32(p + 0x14, (uint32_t)t->init_prio);
            st32(p + 0x18, (uint32_t)t->prio);
            st32(p + 0x1C, t->attr);
            st32(p + 0x20, t->option);
            st32(p + 0x24, (uint32_t)t->wait_type);
            st32(p + 0x28, (uint32_t)t->wait_id);
            st32(p + 0x2C, (uint32_t)t->wakeup);
        }
        ret(c, t->status);
        break;
    }
    case 50:                                         // SleepThread
        if (me->wakeup > 0) {
            --me->wakeup;
        } else {
            me->status = THS_WAIT;
            me->wait_type = WAIT_SLEEP;
            schedule(lk);
        }
        ret(c, me->id);
        break;
    case 51: case 52: {                              // WakeupThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t == me) { ret(c, -1); break; }
        if ((t->status & THS_WAIT) && t->wait_type == WAIT_SLEEP) {
            if (t->status == THS_WAITSUSPEND) { t->status = THS_SUSPEND; t->wait_type = WAIT_NONE; }
            else make_ready(t);
        } else {
            ++t->wakeup;
        }
        ret(c, t->id);
        if (!intr) schedule(lk);
        break;
    }
    case 53: case 54: {                              // CancelWakeupThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t) { ret(c, -1); break; }
        int w = t->wakeup;
        t->wakeup = 0;
        ret(c, w);
        break;
    }
    case 55: case 56: {                              // SuspendThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t == me || t->status == THS_DORMANT) { ret(c, -1); break; }
        if (t->status == THS_READY) t->status = THS_SUSPEND;
        else if (t->status == THS_WAIT) t->status = THS_WAITSUSPEND;
        ret(c, t->id);
        break;
    }
    case 57: case 58: {                              // ResumeThread
        Thread* t = thread_of(sarg(c, 0));
        if (!t || t == me) { ret(c, -1); break; }
        if (t->status == THS_SUSPEND) make_ready(t);
        else if (t->status == THS_WAITSUSPEND) t->status = THS_WAIT;
        ret(c, t->id);
        if (!intr) schedule(lk);
        break;
    }

    // -- semaphores -----------------------------------------------------------------------------
    case 64: {                                       // CreateSema (param)
        uint32_t p = arg(c, 0);
        int id = -1;
        for (int k = 1; k < MAX_SEMAS; ++k)
            if (!g_sema[k].used) { id = k; break; }
        if (id < 0) { ret(c, -1); break; }
        Sema& s = g_sema[id];
        s = Sema{};
        s.used = true;
        s.max = (int)ld32(p + 0x04);
        s.init = s.count = (int)ld32(p + 0x08);
        s.attr = ld32(p + 0x10);
        s.option = ld32(p + 0x14);
        ret(c, id);
        break;
    }
    case 65: {                                       // DeleteSema
        int id = sarg(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { ret(c, -1); break; }
        for (int w : g_sema[id].waiters) {
            g_th[w].wait_result = -1;
            if (g_th[w].status == THS_WAITSUSPEND) g_th[w].status = THS_SUSPEND;
            else make_ready(&g_th[w]);
        }
        g_sema[id] = Sema{};
        ret(c, id);
        if (!intr) schedule(lk);
        break;
    }
    case 66: case 67: {                              // SignalSema
        int id = sarg(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { ret(c, -1); break; }
        Sema& s = g_sema[id];
        if (!s.waiters.empty()) {
            Thread* t = &g_th[s.waiters.front()];
            s.waiters.pop_front();
            t->wait_result = id;
            if (t->status == THS_WAITSUSPEND) { t->status = THS_SUSPEND; t->wait_type = WAIT_NONE; }
            else make_ready(t);
        } else {
            ++s.count;
        }
        ret(c, id);
        if (!intr) schedule(lk);
        break;
    }
    case 68: {                                       // WaitSema
        int id = sarg(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { ret(c, -1); break; }
        Sema& s = g_sema[id];
        if (s.count > 0) {
            --s.count;
            ret(c, id);
            break;
        }
        if (intr) rt_die("WaitSema(%d) in an interrupt handler, from %08X", id, c.r[31].uw[0]);
        s.waiters.push_back(me->id);
        me->status = THS_WAIT;
        me->wait_type = WAIT_SEMA;
        me->wait_id = id;
        schedule(lk);
        ret(c, me->wait_result);
        break;
    }
    case 69: case 70: {                              // PollSema
        int id = sarg(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { ret(c, -1); break; }
        if (g_sema[id].count > 0) { --g_sema[id].count; ret(c, id); }
        else ret(c, -1);
        break;
    }
    case 71: case 72: {                              // ReferSemaStatus (id, info)
        int id = sarg(c, 0);
        uint32_t p = arg(c, 1);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { ret(c, -1); break; }
        Sema& s = g_sema[id];
        st32(p + 0x00, (uint32_t)s.count);
        st32(p + 0x04, (uint32_t)s.max);
        st32(p + 0x08, (uint32_t)s.init);
        st32(p + 0x0C, (uint32_t)s.waiters.size());
        st32(p + 0x10, s.attr);
        st32(p + 0x14, s.option);
        ret(c, id);
        break;
    }

    // -- interrupts, alarms ------------------------------------------------------------------------
    case 16:                                         // AddIntcHandler (cause, handler, next)
        if (arg(c, 0) >= 15) { ret(c, -1); break; }
        ret(c, add_handler(g_intc_h[arg(c, 0)], arg(c, 1), sarg(c, 2), c.r[28].uw[0], arg(c, 3)));
        break;
    case 17:                                         // RemoveIntcHandler (cause, id)
        ret(c, arg(c, 0) < 15 ? remove_handler(g_intc_h[arg(c, 0)], sarg(c, 1)) : -1);
        break;
    case 18:                                         // AddDmacHandler (channel, handler, next)
        if (arg(c, 0) >= 10) { ret(c, -1); break; }
        ret(c, add_handler(g_dmac_h[arg(c, 0)], arg(c, 1), sarg(c, 2), c.r[28].uw[0], arg(c, 3)));
        break;
    case 19:                                         // RemoveDmacHandler (channel, id)
        ret(c, arg(c, 0) < 10 ? remove_handler(g_dmac_h[arg(c, 0)], sarg(c, 1)) : -1);
        break;
    case 20: case 26: case 21: case 27: {            // _EnableIntc, _DisableIntc (cause)
        uint32_t cause = arg(c, 0);
        if (cause >= 15) { ret(c, -1); break; }
        bool on = n == 20 || n == 26;
        bool was = hw_intc_mask() & (1u << cause);
        hw_intc_mask_set((int)cause, on);
        ret(c, was != on ? 1 : 0);
        break;
    }
    case 22: case 28: case 23: case 29: {            // _EnableDmac, _DisableDmac (channel)
        uint32_t ch = arg(c, 0);
        if (ch >= 10) { ret(c, -1); break; }
        bool on = n == 22 || n == 28;
        bool was = hw_dmac_mask() & (1u << ch);
        hw_dmac_mask_set((int)ch, on);
        ret(c, was != on ? 1 : 0);
        break;
    }
    case 24: case 30: {                              // SetAlarm (time in horizontal blanks, handler, arg)
        uint32_t ticks = arg(c, 0) & 0xFFFF;
        Alarm a{++g_alarm_ids, hw_now() + ticks / 15625.0, arg(c, 1), c.r[28].uw[0], arg(c, 2), ticks};
        g_alarms.push_back(a);
        hw_set_deadline(a.due);
        ret(c, a.id);
        break;
    }
    case 25: case 31: {                              // ReleaseAlarm (id)
        int id = sarg(c, 0);
        auto it = std::find_if(g_alarms.begin(), g_alarms.end(), [&](const Alarm& a) { return a.id == id; });
        if (it == g_alarms.end()) { ret(c, -1); break; }
        g_alarms.erase(it);
        ret(c, id);
        break;
    }
    case 100: case 104: ret(c, 0); break;            // FlushCache: no caches here

    // -- the GS -----------------------------------------------------------------------------------
    case 2:                                          // SetGsCrt (interlace, mode, field/frame)
        TRACE(1, "[gs] SetGsCrt interlace %u mode %u ffmd %u\n", arg(c, 0), arg(c, 1), arg(c, 2));
        ret(c, 0);
        break;
    case 112: ret(c, (int32_t)hw_gs_imr()); break;   // GsGetIMR
    case 113: {                                      // GsPutIMR
        uint32_t old = hw_gs_imr();
        hw_gs_set_imr(arg(c, 0));
        ret(c, (int32_t)old);
        break;
    }
    case 115:                                        // SetVSyncFlag (flag, csr)
        g_vsync_flag = arg(c, 0);
        g_vsync_csr = arg(c, 1);
        ret(c, 0);
        break;

    // -- the SIF ----------------------------------------------------------------------------------
    case 121: {                                      // sceSifSetReg (register, value)
        uint32_t r = arg(c, 0), v = arg(c, 1);
        uint32_t old;
        if (r & 0x80000000u) { old = g_sreg[r & 31]; g_sreg[r & 31] = v; }
        else if (r >= 1 && r <= 4) { old = hw_sif_get((int)r); hw_sif_set((int)r, v); }
        else { std::fprintf(stderr, "ps2kit: sceSifSetReg(%X): no such register\n", r); old = 0; }
        ret(c, (int32_t)old);
        break;
    }
    case 122: {                                      // sceSifGetReg (register)
        uint32_t r = arg(c, 0), v;
        if (!(r & 0x80000000u)) {
            lk.unlock();
            iop_update();
            lk.lock();
        }
        if (r & 0x80000000u) v = g_sreg[r & 31];
        else if (r >= 1 && r <= 4) v = hw_sif_get((int)r);
        else { std::fprintf(stderr, "ps2kit: sceSifGetReg(%X): no such register\n", r); v = 0; }
        ret(c, (int32_t)v);
        if (!(r & 0x80000000u)) sif_wait_check(c, (int)r, v);
        break;
    }
    case 119: {                                      // sceSifSetDma (transfers, count)
        uint32_t p = arg(c, 0), count = arg(c, 1);
        for (uint32_t k = 0; k < count; ++k, p += 16) {
            uint32_t src = ld32(p), dst = ld32(p + 4), size = ld32(p + 8), attr = ld32(p + 12);
            TRACE(2, "[sif] EE %08X -> IOP %06X, %u bytes, attr %X\n", src, dst, size, attr);
            lk.unlock();
            iop_sif1(src, dst, size, attr);
            lk.lock();
        }
        ret(c, ++g_sif_dma_id);
        break;
    }
    case 118: ret(c, -1); break;                     // sceSifDmaStat: every transfer has ended
    case 120:                                        // sceSifSetDChain: SIF0 armed for the IOP
        lk.unlock();
        ee_hw_write(0x1000C020, 0, 4);
        ee_hw_write(0x1000C000, 0x184, 4);
        lk.lock();
        ret(c, 0);
        break;
    case 107:                                        // sceSifStopDma
        lk.unlock();
        ee_hw_write(0x1000C000, ee_hw_read(0x1000C000, 4) & ~0x100u, 4);
        lk.lock();
        ret(c, 0);
        break;

    case 124: {                                      // Deci2Call (function, parameters)
        lk.unlock();
        int32_t r = deci2(c, sarg(c, 0), arg(c, 1));
        lk.lock();
        ret(c, r);
        break;
    }

    default:
        lk.unlock();
        std::fflush(stdout);
        rt_die("syscall %d (%s) at %08X, from %08X (%s): not modelled", raw, sys_name(n), pc, c.r[31].uw[0],
               rt_name_of(c.r[31].uw[0]));
    }
    lk.unlock();
    if (g_trace >= 1) {
        char r[32];
        std::snprintf(r, sizeof r, " = %d", c.r[2].sw[0]);
        trace_line(std::string(line) + r);
    }
}

void ee_poll(EEContext& c) {
    if (g_in_intr || (c.cop0[12] & (ST_IE | ST_EIE)) != (ST_IE | ST_EIE)) return;
    deliver();
    std::unique_lock<std::mutex> lk(g_k);
    schedule(lk);
}

void ee_ei(EEContext& c) {
    c.cop0[12] |= ST_EIE;
    if (g_ee_pending.load(std::memory_order_relaxed)) ee_poll(c);
}
void ee_di(EEContext& c) { c.cop0[12] &= ~ST_EIE; }
uint32_t ee_mfc0(EEContext& c, int reg) { return c.cop0[reg]; }
void ee_mtc0(EEContext& c, int reg, uint32_t v) { c.cop0[reg] = v; }

void ee_break(EEContext&, uint32_t pc, uint32_t code) {
    std::fflush(stdout);
    rt_die("break 0x%X at %08X (%s)", code, pc, rt_name_of(pc));
}
void ee_unimplemented(EEContext&, uint32_t pc, const char* what) {
    std::fflush(stdout);
    rt_die("%s at %08X (%s)", what, pc, rt_name_of(pc));
}
uint32_t ee_cfc2(EEContext& c, int reg) { return c.vi[reg]; }
void ee_ctc2(EEContext& c, int reg, uint32_t v) { c.vi[reg] = v; }
void ee_vu0_callms(EEContext& c, uint32_t addr) {
    std::fflush(stdout);
    rt_die("VU0 micro program at %04X, from near %08X (%s): not modelled yet", addr, c.pc, rt_name_of(c.pc));
}

// ---- starting the guest ---------------------------------------------------------------------------
void kernel_wake() {
    { std::lock_guard<std::mutex> lk(g_k); }
    g_idle_cv.notify_all();
}

int kernel_run(uint32_t entry, const char* argv0) {
    g_kernel_argv0 = argv0;
    if (const char* s = std::getenv("PS2KIT_IDLE_LIMIT")) g_idle_limit = std::atof(s);
    std::unique_lock<std::mutex> lk(g_k);
    Thread& m = g_th[1];
    m = Thread{};
    m.id = 1;
    m.used = true;
    m.func = entry;
    m.prio = m.init_prio = 0;          // the main thread starts at priority 0
    m.c.cop0[12] = ST_IE;              // crt0 sets EIE itself (its `ei` before main)
    g_cur = &m;
    m.status = THS_RUN;
    spawn(&m);
    lk.unlock();
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

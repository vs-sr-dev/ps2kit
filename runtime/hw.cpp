// ps2kit runtime — the EE's hardware registers, as far as booting needs them.
//
// Every register is a stored value; the ones with behaviour are below:
//
//   INTC     INTC_STAT bits clear when 1 is written, INTC_MASK bits toggle
//            when 1 is written (EE User's Manual, INTC).
//   DMAC     D_STAT: CIS bits clear on 1, CIM bits toggle on 1. A channel
//            runs when CHCR.STR is set, and finishes at once: the data is
//            walked (normal mode, source chain with QWC > 0 sent first, call
//            and ret with the two ASR slots) and handed to the channel's
//            destination; then STR clears and the channel's CIS bit is set.
//            Transfer time is not modelled. Scratchpad transfers (8, 9) copy;
//            VIF0 and VIF1 data goes to vu0.cpp and vif1.cpp (with the
//            tag's upper half first when CHCR.TTE is set), GIF data to the
//            GS, IPU data to the IPU; SIF0 stays running until the IOP side
//            sends something.
//   Timers   Tn_COUNT counts from the host clock at the selected rate
//            (bus clock 147.456 MHz, /16, /256, or the horizontal blank).
//            Timer interrupts are not modelled (said once if asked for).
//   GS       CSR: SIGNAL, FINISH, HSINT, VSINT, EDWINT clear on 1; FIELD
//            toggles every vertical blank. IMR is stored.
//   SIO      a byte written to the transmit FIFO goes to the host console.
//   SIF      MSCOM, SMCOM, MSFLG, SMFLG: the EE sets MSFLG bits by writing
//            them and clears SMFLG bits the same way; the IOP side owns the rest.
//
// Anything else is stored and read back, and its first access is reported,
// so that the boot shows what it touches.
#include "os.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <timeapi.h>
#endif

namespace {

std::mutex g_hw;
std::unordered_map<uint32_t, uint64_t> g_reg;
std::set<uint32_t> g_reported;

std::atomic<uint32_t> g_intc_stat{0};          // read without the lock (polled in tight loops)
uint32_t g_intc_mask;
uint32_t g_dstat;                       // CIS 0-9, SIS 13, MEIS 14, BEIS 15, CIM 16-25, SIM 29, MEIM 30
std::atomic<uint64_t> g_gs_csr{0};             // read without the lock (polled in tight loops)
uint32_t g_gs_imr = 0x7F00;             // all GS interrupts masked
uint32_t g_sif[5];                      // [1] MSCOM [2] SMCOM [3] MSFLG [4] SMFLG
bool g_pal = true;

using Clock = std::chrono::steady_clock;
Clock::time_point g_t0 = Clock::now();
std::condition_variable g_clock_cv;
double g_deadline = 1e300;

constexpr double BUS_HZ = 147456000.0;

struct Timer {
    uint32_t mode = 0, comp = 0, hold = 0;
    uint32_t c0 = 0;                    // the count at t0
    double t0 = 0;
    double rate() const {
        switch (mode & 3) {
        case 0: return BUS_HZ;
        case 1: return BUS_HZ / 16;
        case 2: return BUS_HZ / 256;
        default: return g_pal ? 15625.0 : 15734.0;
        }
    }
    uint32_t count(double now) const {
        if (!(mode & 0x80)) return c0;
        return (uint32_t)(c0 + (uint64_t)((now - t0) * rate())) & 0xFFFF;
    }
} g_timer[4];

const uint32_t DMA_BASE[10] = {0x10008000, 0x10009000, 0x1000A000, 0x1000B000, 0x1000B400,
                               0x1000C000, 0x1000C400, 0x1000C800, 0x1000D000, 0x1000D400};
const char* const DMA_NAME[10] = {"VIF0", "VIF1", "GIF", "fromIPU", "toIPU",
                                  "SIF0", "SIF1", "SIF2", "fromSPR", "toSPR"};
uint64_t g_dma_qw[10];                  // quadwords each channel has carried
bool (*g_sif0_source)() = nullptr;
std::atomic<uint32_t> g_vblanks{0};

void settle_flips_locked();
void vbon_taken_locked();

void update_pending() {
    if ((g_intc_stat & g_intc_mask) || (g_dstat & (g_dstat >> 16) & 0x3FF))
        g_ee_pending.store(1, std::memory_order_relaxed);
}

const char* reg_name(uint32_t a) {
    static const struct { uint32_t a; const char* n; } names[] = {
        {0x10003000, "GIF_CTRL"}, {0x10003010, "GIF_MODE"}, {0x10003020, "GIF_STAT"},
        {0x10003800, "VIF0_STAT"}, {0x10003810, "VIF0_FBRST"}, {0x10003820, "VIF0_ERR"},
        {0x10003C00, "VIF1_STAT"}, {0x10003C10, "VIF1_FBRST"}, {0x10003C20, "VIF1_ERR"}, {0x1000E000, "D_CTRL"}, {0x1000E010, "D_STAT"},
        {0x1000E020, "D_PCR"}, {0x1000E030, "D_SQWC"}, {0x1000E040, "D_RBSR"}, {0x1000E050, "D_RBOR"},
        {0x1000E060, "D_STADR"}, {0x1000F000, "INTC_STAT"}, {0x1000F010, "INTC_MASK"},
        {0x1000F130, "SIO_ISR"}, {0x1000F180, "SIO_TXFIFO"}, {0x1000F200, "SB_MSCOM"},
        {0x1000F210, "SB_SMCOM"}, {0x1000F220, "SB_MSFLG"}, {0x1000F230, "SB_SMFLG"},
        {0x1000F240, "SB_CTRL"}, {0x1000F260, "SB_BD6"}, {0x1000F520, "D_ENABLER"},
        {0x1000F590, "D_ENABLEW"}, {0x12000000, "PMODE"}, {0x12000010, "SMODE1"},
        {0x12000020, "SMODE2"}, {0x12000030, "SRFSH"}, {0x12000040, "SYNCH1"}, {0x12000050, "SYNCH2"},
        {0x12000060, "SYNCV"}, {0x12000070, "DISPFB1"}, {0x12000080, "DISPLAY1"},
        {0x12000090, "DISPFB2"}, {0x120000A0, "DISPLAY2"}, {0x120000B0, "EXTBUF"},
        {0x120000C0, "EXTDATA"}, {0x120000D0, "EXTWRITE"}, {0x120000E0, "BGCOLOR"},
        {0x12001000, "GS_CSR"}, {0x12001010, "GS_IMR"}, {0x12001040, "GS_BUSDIR"},
        {0x12001080, "GS_SIGLBLID"},
    };
    for (auto& n : names)
        if (n.a == a) return n.n;
    if (a < 0x10002000) return "timer";
    if (a >= 0x10008000 && a < 0x1000E000) return "DMA channel";
    return "?";
}

void report(const char* what, uint32_t a, uint64_t v) {
    if (g_trace >= 2 || (g_reported.insert(a).second && std::strcmp(reg_name(a), "?") == 0))
        std::fprintf(stderr, "hw: %s %08X (%s) = %llX\n", what, a, reg_name(a), (unsigned long long)v);
}

// GS events from the GIF: SIGNAL and FINISH set their CSR bits, and raise
// the GS interrupt unless IMR masks them (SIGMSK bit 8, FINISHMSK bit 9).
void gs_events_locked() {
    uint64_t e = gs_take_events();
    if (!e) return;
    g_gs_csr |= e & 3;
    if (((e & 1) && !(g_gs_imr & 0x100)) || ((e & 2) && !(g_gs_imr & 0x200))) {
        g_intc_stat |= 1u << INTC_GS;
        update_pending();
    }
}

// PS2KIT_TIMELINE=1: on stderr, with the time in seconds, each vertical
// blank (and whether channels 1 and 2 are busy, since when), each start and
// end of channels 1 and 2, the PCRTC's registers as they change, and the
// writes to VIF1_FBRST and GIF_CTRL: to see when a game's lists run.
const bool g_timeline = [] {
    const char* e = std::getenv("PS2KIT_TIMELINE");
    return e && *e && *e != '0';
}();

// ---- DMA ---------------------------------------------------------------------------------
int g_dma_ch = -1;                                       // the channel dma_run is walking (for messages)
uint32_t g_dma_token[10];                                // the transfer a deferred end belongs to
uint32_t& chreg(int ch, uint32_t off);
struct TagSeen { uint32_t tadr; uint64_t tag0, tag1; uint64_t run; };
TagSeen g_tags[10][32];                                  // per channel, the last tags walked
unsigned g_ntags[10];
uint64_t g_dma_runs;

[[noreturn]] void dma_die(const char* what, uint32_t a, uint32_t qwc) {
    if (g_dma_ch < 0) rt_die("%s: %08X + %u qw", what, a, qwc);
    unsigned n = g_ntags[g_dma_ch];
    std::fprintf(stderr, "ps2kit: the last tags of %s (transfer number, TADR, tag):\n", DMA_NAME[g_dma_ch]);
    for (unsigned k = n > 32 ? n - 32 : 0; k < n; ++k) {
        const TagSeen& t = g_tags[g_dma_ch][k % 32];
        std::fprintf(stderr, "  %llu  %08X  %016llX %016llX  (id %u, qwc %u, addr %08X)\n", (unsigned long long)t.run,
                     t.tadr, (unsigned long long)t.tag1, (unsigned long long)t.tag0, (unsigned)(t.tag0 >> 28 & 7),
                     (unsigned)(t.tag0 & 0xFFFF), (unsigned)(t.tag0 >> 32));
    }
    rt_die("%s: %08X + %u qw, channel %s (CHCR %08X MADR %08X QWC %X TADR %08X; D_CTRL %08X D_STADR %08X)", what, a,
           qwc, DMA_NAME[g_dma_ch], chreg(g_dma_ch, 0x00), chreg(g_dma_ch, 0x10), chreg(g_dma_ch, 0x20),
           chreg(g_dma_ch, 0x30), (uint32_t)g_reg[0x1000E000], (uint32_t)g_reg[0x1000E060]);
}

// VIF1's and the GIF's channels: from a start to the end of the transfer
// (when VIF1 and the GS have taken all of it), since the last heartbeat. A
// game that sends its lists at each vertical blank needs them done within a
// field.
struct Latency {
    double t0 = 0, sum = 0, max = 0;
    uint64_t n = 0, over = 0;
    void end(double now) {
        double d = now - t0;
        sum += d; ++n;
        max = std::max(max, d);
        if (d > 0.02) ++over;
    }
};
Latency g_lat[3];

uint8_t* dma_ptr(uint32_t a, uint32_t qwc) {
    if (a & 0x80000000u) {                               // scratchpad
        uint32_t o = a & 0x3FF0;
        if (o + qwc * 16 > 0x4000) dma_die("DMA beyond the scratchpad", a, qwc);
        return g_ee_spr + o;
    }
    uint32_t p = a & 0x01FFFFF0u;
    if (p + qwc * 16 > EE_RAM_SIZE) dma_die("DMA beyond RAM", a, qwc);
    return g_ee_ram + p;
}

uint32_t& chreg(int ch, uint32_t off) {
    auto it = g_reg.find(DMA_BASE[ch] + off);
    if (it == g_reg.end()) it = g_reg.emplace(DMA_BASE[ch] + off, 0).first;
    return *reinterpret_cast<uint32_t*>(&it->second);
}

// The data of one source transfer, in order, to the channel's destination.
void dma_out(int ch, uint32_t addr, uint32_t qwc, const uint64_t* tag) {
    if (ch == 0 && tag && (chreg(0, 0x00) & 0x40)) vif0_data((const uint8_t*)&tag[1], 8);   // TTE
    if (ch == 1 && tag && (chreg(1, 0x00) & 0x40)) vif1_data((const uint8_t*)&tag[1], 8, true);
    if (!qwc) return;
    const uint8_t* p = dma_ptr(addr, qwc);
    g_dma_qw[ch] += qwc;
    if (ch == 2) {                                       // GIF, path 3
        gs_gif(3, p, qwc);
        gs_events_locked();
        return;
    }
    if (ch == 0) {                                       // VIF0
        vif0_data(p, qwc * 16);
        return;
    }
    if (ch == 1) {                                       // VIF1
        vif1_data(p, qwc * 16, false);
        gs_events_locked();
        return;
    }
    if (ch == 9) {                                       // toSPR
        uint32_t& sadr = chreg(9, 0x80);
        if ((sadr & 0x3FF0) + qwc * 16 > 0x4000) rt_die("toSPR beyond the scratchpad");
        std::memcpy(g_ee_spr + (sadr & 0x3FF0), p, qwc * 16);
        sadr = (sadr + qwc * 16) & 0x3FF0;
        return;
    }
    TRACE(2, "dma: %s %u qw from %08X\n", DMA_NAME[ch], qwc, addr);
}

// SIF0 is armed: if the IOP side has a packet, it lands and the channel ends.
void sif0_try_locked() {
    uint32_t& chcr = chreg(5, 0x00);
    if (!(chcr & 0x100) || !g_sif0_source || !g_sif0_source()) return;
    chcr &= ~0x100u;
    g_dstat |= 1u << 5;
    update_pending();
}

void dma_run(int ch) {
    struct Cur { Cur(int c) { g_dma_ch = c; } ~Cur() { g_dma_ch = -1; } } cur(ch);
    ++g_dma_runs;
    uint32_t& chcr = chreg(ch, 0x00);
    uint32_t& madr = chreg(ch, 0x10);
    uint32_t& qwc = chreg(ch, 0x20);
    uint32_t& tadr = chreg(ch, 0x30);
    int mode = (chcr >> 2) & 3;
    bool from_mem = chcr & 1;                            // DIR: 1 = memory to peripheral
    if (ch == 5) {                                       // SIF0, IOP to EE: waits for the IOP
        TRACE(2, "dma: SIF0 armed (CHCR %08X, TADR %08X)\n", chcr, tadr);
        sif0_try_locked();
        return;
    }
    if (ch == 8) {                                       // fromSPR
        if (mode != 0) rt_die("fromSPR in mode %d is not modelled", mode);
        uint32_t& sadr = chreg(8, 0x80);
        uint8_t* d = dma_ptr(madr, qwc);
        if ((sadr & 0x3FF0) + qwc * 16 > 0x4000) rt_die("fromSPR beyond the scratchpad");
        std::memcpy(d, g_ee_spr + (sadr & 0x3FF0), qwc * 16);
        g_dma_qw[8] += qwc;
        madr += qwc * 16;
        sadr = (sadr + qwc * 16) & 0x3FF0;
        qwc = 0;
    } else if (!from_mem && ch == 1 && mode == 0) {      // VIF1 from the GS: a local-to-host transfer
        vif1_drain();
        uint8_t* d = dma_ptr(madr, qwc);
        g_dma_qw[1] += qwc;
        uint32_t n = qwc;
        madr += qwc * 16;
        qwc = 0;
        if (!gs_readback(d, n, ++g_dma_token[1])) return;   // ends when the GS has sent it
        chcr &= ~0x100u;
        g_dstat |= 1u << ch;
        update_pending();
        return;
    } else if (!from_mem && ch != 9 && ch != 6) {
        rt_die("DMA %s towards memory (CHCR %08X) is not modelled", DMA_NAME[ch], chcr);
    } else if (mode == 0) {
        dma_out(ch, madr, qwc, nullptr);
        madr += qwc * 16;
        qwc = 0;
    } else if (mode == 1) {
        // Source chain. A channel started with QWC > 0 sends those quadwords
        // from MADR first, then reads its first tag at TADR.
        if (qwc) {
            dma_out(ch, madr, qwc, nullptr);
            madr += qwc * 16;
            qwc = 0;
        }
        int asp = (chcr >> 4) & 3;
        uint32_t& asr0 = chreg(ch, 0x40);
        uint32_t& asr1 = chreg(ch, 0x50);
        bool tie = chcr & 0x80;
        for (int guard = 0;; ++guard) {
            if (guard > 1000000) rt_die("DMA %s: a chain of more than a million tags", DMA_NAME[ch]);
            uint64_t tag[2];
            std::memcpy(tag, dma_ptr(tadr, 1), 16);
            g_tags[ch][g_ntags[ch]++ % 32] = TagSeen{tadr, tag[0], tag[1], g_dma_runs};
            uint32_t n = tag[0] & 0xFFFF, id = (tag[0] >> 28) & 7, irq = (tag[0] >> 31) & 1;
            uint32_t addr = (uint32_t)(tag[0] >> 32) & 0xFFFFFFF0u;
            chcr = (chcr & 0xFFFF) | (uint32_t)((tag[0] >> 16) & 0xFFFF0000u);
            uint32_t next = tadr + 16 + n * 16;
            bool end = false;
            switch (id) {
            case 0: dma_out(ch, addr, n, tag); end = true; break;                           // refe
            case 1: dma_out(ch, tadr + 16, n, tag); tadr = next; break;                     // cnt
            case 2: dma_out(ch, tadr + 16, n, tag); tadr = addr; break;                     // next
            case 3: case 4: dma_out(ch, addr, n, tag); tadr += 16; break;                   // ref, refs
            case 5:                                                                         // call
                dma_out(ch, tadr + 16, n, tag);
                if (asp >= 2) rt_die("DMA %s: call with the address stack full", DMA_NAME[ch]);
                (asp == 0 ? asr0 : asr1) = next;
                ++asp;
                tadr = addr;
                break;
            case 6:                                                                         // ret
                dma_out(ch, tadr + 16, n, tag);
                if (asp == 0) { end = true; tadr = next; break; }
                --asp;
                tadr = asp == 0 ? asr0 : asr1;
                break;
            case 7: dma_out(ch, tadr + 16, n, tag); tadr = next; end = true; break;         // end
            }
            if (end || (tie && irq)) break;
        }
        chcr = (chcr & ~0x30u) | (uint32_t)asp << 4;
    } else {
        rt_die("DMA %s in interleave mode is not modelled", DMA_NAME[ch]);
    }
    if (ch == 1 && !vif1_channel_end(++g_dma_token[1])) return;   // ends when VIF1 has done it and the GS drawn it
    if (ch == 2 && !gs_channel_end(ch, ++g_dma_token[ch])) return;   // ends when the GS has drawn it
    chcr &= ~0x100u;
    g_dstat |= 1u << ch;
    update_pending();
}

void hw_write_locked(uint32_t a, uint64_t v) {
    if (a == 0x1000F000) {
        if ((v & (1u << INTC_VBON)) && (g_intc_stat & (1u << INTC_VBON))) vbon_taken_locked();   // see the clock
        g_intc_stat &= ~(uint32_t)v;
        return;
    }
    if (a == 0x1000F010) { g_intc_mask ^= (uint32_t)v & 0x7FFF; settle_flips_locked(); update_pending(); return; }
    if (a == 0x1000E010) {
        g_dstat &= ~((uint32_t)v & 0xE3FF);
        g_dstat ^= (uint32_t)v & 0x63FF0000u;
        update_pending();
        return;
    }
    if (a == 0x1000F180) { std::fputc((int)(v & 0xFF), stdout); std::fflush(stdout); return; }
    if (a == 0x12001000) {
        g_gs_csr &= ~(v & 0x1F);
        if (v & 0x200) TRACE(1, "hw: GS reset\n");
        return;
    }
    if (a == 0x12001010) { g_gs_imr = (uint32_t)v; return; }
    if (a >= 0x1000F200 && a <= 0x1000F230) {
        int r = (int)((a - 0x1000F200) >> 4) + 1;
        if (r == 3) g_sif[3] |= (uint32_t)v;
        else if (r == 4) g_sif[4] &= ~(uint32_t)v;
        else g_sif[r] = (uint32_t)v;
        return;
    }
    if (a < 0x10002000) {                                // timers
        Timer& t = g_timer[(a >> 11) & 3];
        double now = hw_now();
        switch ((a >> 4) & 0xF) {
        case 0: t.c0 = (uint32_t)v & 0xFFFF; t.t0 = now; return;
        case 1:
            t.c0 = t.count(now); t.t0 = now;
            t.mode = (t.mode & 0xC00 & ~((uint32_t)v & 0xC00)) | ((uint32_t)v & 0x3FF);   // EQUF, OVFF clear on 1
            if ((v & 0x300) && g_reported.insert(a).second)
                std::fprintf(stderr, "hw: timer %u interrupts asked for (mode %X): not modelled\n", (a >> 11) & 3, (unsigned)v);
            return;
        case 2: t.comp = (uint32_t)v & 0xFFFF; return;
        case 3: t.hold = (uint32_t)v & 0xFFFF; return;
        }
    }
    if ((a >= 0x10002000 && a <= 0x10002030)) { ipu_write_locked(a, v); return; }
    for (int ch = 0; ch < 10; ++ch) {
        if (a == DMA_BASE[ch]) {
            // VIF1 and the GIF stay busy until the GS has drawn what they sent
            // (gs_channel_end). A transfer starts when STR goes from 0 to 1, as
            // on the console: a CHCR written back with STR still set while
            // they are busy changes the other bits, it starts nothing (DW2
            // does that; taken as a start, the walk went on from the last tag).
            bool busy = (ch == 1 || ch == 2) && (chreg(ch, 0) & 0x100);
            chreg(ch, 0) = (uint32_t)v;
            if (busy && (v & 0x100)) return;
            // stopped while busy (sceDmaPause, before a game's reset): the end
            // VIF1 or the GS would still report is no longer this channel's
            if (busy) ++g_dma_token[ch];
            if ((ch == 3 || ch == 4) && (v & 0x100)) {       // fromIPU, toIPU: run by the IPU
                TRACE(2, "dma: %s start CHCR %08X MADR %08X QWC %X TADR %08X\n", DMA_NAME[ch], (uint32_t)v,
                      chreg(ch, 0x10), chreg(ch, 0x20), chreg(ch, 0x30));
                ipu_dma_locked(ch);
            } else if (v & 0x100) {
                TRACE(2, "dma: %s start CHCR %08X MADR %08X QWC %X TADR %08X\n", DMA_NAME[ch], (uint32_t)v,
                      chreg(ch, 0x10), chreg(ch, 0x20), chreg(ch, 0x30));
                if (ch == 1 || ch == 2) g_lat[ch].t0 = hw_now();
                if (g_timeline && (ch == 1 || ch == 2)) {
                    bool h;
                    uint32_t ra = kernel_guest_ra(h);
                    std::fprintf(stderr, "start %d %.4f %s %s (TADR %08X)\n", ch, hw_now(), h ? "handler" : "thread",
                                 rt_name_of(ra), chreg(ch, 0x30));
                }
                uint64_t qw0 = g_dma_qw[ch];
                dma_run(ch);
                if (g_timeline && (ch == 1 || ch == 2))
                    std::fprintf(stderr, "sent %d %.4f %llu qw\n", ch, hw_now(), (unsigned long long)(g_dma_qw[ch] - qw0));
            }
            return;
        }
    }
    if (g_timeline && ((a >= 0x12000000 && a <= 0x120000E0 && g_reg[a] != v) || a == 0x10003C10 || a == 0x10003000))
        std::fprintf(stderr, "%s %.4f %s = %llX\n", a < 0x12000000 ? "reset" : "pcrtc", hw_now(), reg_name(a),
                     (unsigned long long)v);
    // Resets after a timeout (DW2's vblank handler when the lists are still
    // running): VIF1's (FBRST RST) and the GIF's (GIF_CTRL RST) drop the work
    // they were given and have not done, as on the console. Only with
    // PS2KIT_RESETS=1 for now: DW2's battle lists take VIF1 and VU1 longer
    // than a field here, and dropped they would lose nearly every frame;
    // ignored, a list finishes late and the next waits behind it.
    static const bool resets = [] {
        const char* e = std::getenv("PS2KIT_RESETS");
        return e && *e && *e != '0';
    }();
    if (resets && a == 0x10003C10 && (v & 1)) vif1_reset();
    if (resets && a == 0x10003000 && (v & 1)) gs_gif_reset();
    g_reg[a] = v;
}

uint64_t hw_read_locked(uint32_t a) {
    if (a == 0x1000F000) return g_intc_stat;
    if (a == 0x1000F010) return g_intc_mask;
    if (a == 0x1000E010) return g_dstat;
    if (a == 0x1000F130) return 0;                       // SIO: the transmit FIFO is never full
    if (a == 0x12001000) return g_gs_csr;
    if (a == 0x12001010) return g_gs_imr;
    if (a == 0x12001080) return gs_siglblid();
    if (a >= 0x1000F200 && a <= 0x1000F230) return g_sif[((a - 0x1000F200) >> 4) + 1];
    if (a == 0x10003020 || a == 0x10003800 || a == 0x10003C00) return 0;   // GIF, VIF0, VIF1 idle
    if (a >= 0x10002000 && a <= 0x10002030) return ipu_read_locked(a);
    if (a == 0x10007000 || a == 0x10007008) return ipu_read_locked(a);
    if (a < 0x10002000) {
        const Timer& t = g_timer[(a >> 11) & 3];
        switch ((a >> 4) & 0xF) {
        case 0: return t.count(hw_now());
        case 1: return t.mode;
        case 2: return t.comp;
        case 3: return t.hold;
        }
    }
    auto it = g_reg.find(a);
    return it == g_reg.end() ? 0 : it->second;
}

// ---- the clock: vertical blanks and the kernel's deadlines -------------------------------------
// Events must come on time: a vblank handler that runs late can see FIELD
// already changed (below). Waiting on a condition variable, or the C++
// library's sleep, overslept by up to 15 ms on Windows; the clock sleeps
// with Sleep(1) (timer resolution raised to 1 ms) and spins through the
// last millisecond before an event.
//
// FIELD changes some lines into the blanking interval (with the vertical
// sync pulse, after the front porch), not at its start, and DW2 depends on
// it: set_main_callback waits for a vblank (the SDK's VSync) and samples
// FIELD at once; its handler (main_callback) then busy-waits, at each
// vblank, for FIELD to leave that phase, which takes a few lines only if
// FIELD has not changed yet when the handler starts. On the console the
// interrupt is taken within cycles, far inside those lines; here it is taken
// at the guest's next loop back-edge or after a syscall, which can be much
// later (a disc read). So when the VBON interrupt is enabled, FIELD changes
// three lines after the kernel takes it, however late; blanks that pass
// before it is taken change FIELD then, all but the last at once. With the
// interrupt masked, FIELD changes three lines after the blank begins.
// Three lines is a choice. "Taken" includes the guest acknowledging VBON
// itself in INTC_STAT: the SDK's VSync clears the bit and polls it, and
// Koei's movie player (kpmStartDisplay, from readMpeg) calls it in a loop
// until FIELD changes, with the interrupt enabled and its own vblank handler
// installed. Here the poll often sees the bit before the next back-edge
// delivers it, and then the kernel never takes that blank; FIELD must still
// change, or the loop never ends. (How the console's kernel leaves the bit
// for VSync to see with the interrupt enabled is not known yet.)
double g_line;
double g_next_flip = 1e300;
int g_flips_owed;                       // blanks whose FIELD change waits for the kernel

// The VBON interrupt is taken, by the kernel or by the guest's own
// acknowledgement: FIELD changes owed for earlier blanks happen now, the
// last blank's three lines later (lock held).
void vbon_taken_locked() {
    if (g_flips_owed <= 0) return;
    if (g_next_flip < 1e300) g_gs_csr ^= 0x2000;     // one still due from before
    for (; g_flips_owed > 1; --g_flips_owed) g_gs_csr ^= 0x2000;
    g_flips_owed = 0;
    g_next_flip = hw_now() + 3 * g_line;
}

// VBON masked while changes are owed: they happen now (lock held).
void settle_flips_locked() {
    if (g_intc_mask & (1u << INTC_VBON)) return;
    for (; g_flips_owed > 0; --g_flips_owed) g_gs_csr ^= 0x2000;
}

// PMODE, SMODE2, DISPFB1, DISPLAY1, DISPFB2, DISPLAY2, BGCOLOR (lock held).
void pcrtc_regs_locked(uint64_t regs[7]) {
    static const uint32_t at[7] = {0x12000000, 0x12000020, 0x12000070, 0x12000080, 0x12000090, 0x120000A0,
                                   0x120000E0};
    for (int k = 0; k < 7; ++k) {
        auto it = g_reg.find(at[k]);
        regs[k] = it == g_reg.end() ? 0 : it->second;
    }
}

void clock_thread() {
#ifdef _WIN32
    timeBeginPeriod(1);
    // Above the others: with the GS's drawing threads, VIF1's and the GS's
    // busy, the clock was sometimes not run for 10 ms, a vertical blank came
    // that late and the field after it was short (DW2 then found its list
    // unfinished). It sleeps nearly all the time.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
    double field = g_pal ? 1.0 / 50.0 : 1001.0 / 60000.0;
    g_line = 1.0 / (g_pal ? 15625.0 : 15734.0);
    double next_on = field, next_off = 1e300;
    for (;;) {
        double due, now;
        {
            std::lock_guard<std::mutex> lk(g_hw);
            due = std::min(std::min(std::min(next_on, next_off), g_next_flip), g_deadline);
        }
        now = hw_now();
        if (due - now > 0.0015) {
#ifdef _WIN32
            Sleep(1);
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
            continue;
        }
        if (due > now) {
            std::this_thread::yield();
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(g_hw);
            now = hw_now();
            if (now >= next_on) {
                // Vertical blank start. The blanking interval is taken as about
                // 8% of the field, the order of the real one (not measured).
                g_intc_stat |= 1u << INTC_VBON;
                g_vblanks.fetch_add(1, std::memory_order_relaxed);
                if (g_timeline)
                    std::fprintf(stderr, "vbon %.4f busy1 %d (%.4f) busy2 %d (%.4f) gs lag %.1f ms\n", now,
                                 (chreg(1, 0) >> 8) & 1, now - g_lat[1].t0, (chreg(2, 0) >> 8) & 1, now - g_lat[2].t0,
                                 gs_lag() * 1000);
                uint64_t pcrtc[7];
                pcrtc_regs_locked(pcrtc);
                // the picture is for the field this blank begins: FIELD after its change
                gs_vblank(now, pcrtc, (int)((g_gs_csr.load() >> 13 & 1) ^ 1));
                g_gs_csr |= 0x8;                             // VSINT
                if (!(g_gs_imr & 0x800)) g_intc_stat |= 1u << INTC_GS;
                if (g_intc_mask & (1u << INTC_VBON)) ++g_flips_owed;   // see hw_intc_take
                else g_next_flip = now + 3 * g_line;
                next_off = next_on + field * 0.08;
                next_on += field;
                if (next_on < now) next_on = now + field;    // the host fell behind: no burst
            }
            if (now >= g_next_flip) {
                g_gs_csr ^= 0x2000;                          // FIELD
                g_next_flip = 1e300;
            }
            if (now >= next_off) {
                g_intc_stat |= 1u << INTC_VBOF;
                next_off = 1e300;
            }
            if (now >= g_deadline) g_deadline = 1e300;
            g_ee_pending.store(1, std::memory_order_relaxed);
        }
        kernel_wake();
    }
}

}  // namespace

// ---- the interface ---------------------------------------------------------------------------
// From the GS's thread (gs_start_thread).
void gs_events_from_thread(uint64_t e) {
    {
        std::lock_guard<std::mutex> lk(g_hw);
        g_gs_csr |= e & 3;
        if (((e & 1) && !(g_gs_imr & 0x100)) || ((e & 2) && !(g_gs_imr & 0x200))) {
            g_intc_stat |= 1u << INTC_GS;
            update_pending();
        }
    }
    kernel_wake();
}

void gs_channel_end_from_thread(int ch, uint32_t token) {
    {
        std::lock_guard<std::mutex> lk(g_hw);
        if (token == g_dma_token[ch]) hw_dma_end_locked(ch);   // not a transfer stopped and started again since
    }
    kernel_wake();
}

void hw_init() {
    {
        std::lock_guard<std::mutex> lk(g_hw);
        g_reg[0x1000F520] = 0x1201;                      // D_ENABLER as the BIOS leaves it
        gs_init();
    }
    gs_start_thread(gs_events_from_thread, gs_channel_end_from_thread);
    vif1_start_thread(gs_channel_end_from_thread);
    ipu_init(g_hw);
}

// The picture taken at the last vertical blank (gs_present_enable).
bool hw_gs_picture(std::vector<uint32_t>& out, int& w, int& h) { return gs_picture(out, w, h); }

uint32_t& hw_dma_reg(int ch, uint32_t off) { return chreg(ch, off); }
uint8_t* hw_dma_ptr(uint32_t a, uint32_t qwc) { return dma_ptr(a, qwc); }
void hw_dma_end_locked(int ch) {
    if (ch == 1 || ch == 2) g_lat[ch].end(hw_now());
    if (g_timeline && (ch == 1 || ch == 2)) std::fprintf(stderr, "end %d %.4f\n", ch, hw_now());
    chreg(ch, 0x00) &= ~0x100u;
    g_dstat |= 1u << ch;
    update_pending();
}

void hw_start_clock() {
    g_t0 = Clock::now();
    std::thread(clock_thread).detach();
}

double hw_now() { return std::chrono::duration<double>(Clock::now() - g_t0).count(); }

void hw_set_deadline(double due) {
    std::lock_guard<std::mutex> lk(g_hw);
    if (due < g_deadline) {
        g_deadline = due;
        g_clock_cv.notify_one();
    }
}

uint32_t hw_intc_take() {
    std::lock_guard<std::mutex> lk(g_hw);
    uint32_t p = g_intc_stat & g_intc_mask;
    g_intc_stat &= ~p;
    if (p & (1u << INTC_VBON)) vbon_taken_locked();             // see the clock
    return p;
}

uint32_t hw_dmac_take() {
    std::lock_guard<std::mutex> lk(g_hw);
    uint32_t p = g_dstat & (g_dstat >> 16) & 0x3FF;
    g_dstat &= ~p;
    return p;
}

uint32_t hw_intc_mask() { std::lock_guard<std::mutex> lk(g_hw); return g_intc_mask; }
void hw_intc_mask_set(int cause, bool on) {
    std::lock_guard<std::mutex> lk(g_hw);
    if (on) g_intc_mask |= 1u << cause; else g_intc_mask &= ~(1u << cause);
    settle_flips_locked();
    update_pending();
}
uint32_t hw_dmac_mask() { std::lock_guard<std::mutex> lk(g_hw); return (g_dstat >> 16) & 0x3FF; }
void hw_dmac_mask_set(int ch, bool on) {
    std::lock_guard<std::mutex> lk(g_hw);
    if (on) g_dstat |= 1u << (16 + ch); else g_dstat &= ~(1u << (16 + ch));
    update_pending();
}
void hw_raise_intc(int cause) {
    { std::lock_guard<std::mutex> lk(g_hw); g_intc_stat |= 1u << cause; update_pending(); }
    kernel_wake();
}
void hw_raise_dmac(int ch) {
    { std::lock_guard<std::mutex> lk(g_hw); g_dstat |= 1u << ch; update_pending(); }
    kernel_wake();
}

uint32_t hw_vblank_count() { return g_vblanks.load(std::memory_order_relaxed); }

std::string hw_status() {
    std::lock_guard<std::mutex> lk(g_hw);
    std::string s = std::to_string(g_vblanks.load()) + " vblanks; DMA qw";
    for (int ch = 0; ch < 10; ++ch)
        if (g_dma_qw[ch]) s += std::string(" ") + DMA_NAME[ch] + " " + std::to_string(g_dma_qw[ch]);
    std::string ipu = ipu_status();
    if (!ipu.empty()) s += "; IPU " + ipu;
    std::string vu0 = vu0_status();
    if (!vu0.empty()) s += "; VU0 " + vu0;
    std::string vif1 = vif1_status();
    if (!vif1.empty()) s += "; VIF1" + vif1;
    s += "; " + gs_status();
    for (int ch = 1; ch <= 2; ++ch) {
        Latency& l = g_lat[ch];
        if (!l.n) continue;
        char b[128];
        std::snprintf(b, sizeof b, "; %s start to end %.1f ms mean, %.1f max, %llu over a field", DMA_NAME[ch],
                      1000 * l.sum / (double)l.n, 1000 * l.max, (unsigned long long)l.over);
        s += b;
        double t0 = l.t0;
        l = Latency{};
        l.t0 = t0;
    }
    return s;
}
uint64_t hw_gs_csr() { return g_gs_csr.load(); }
uint32_t hw_gs_imr() { std::lock_guard<std::mutex> lk(g_hw); return g_gs_imr; }
void hw_gs_set_imr(uint32_t v) { std::lock_guard<std::mutex> lk(g_hw); g_gs_imr = v; }

uint32_t hw_sif_get(int reg) {
    std::lock_guard<std::mutex> lk(g_hw);
    return reg >= 1 && reg <= 4 ? g_sif[reg] : 0;
}
void hw_sif_set(int reg, uint32_t v) {
    std::lock_guard<std::mutex> lk(g_hw);
    hw_write_locked(0x1000F200 + (uint32_t)(reg - 1) * 0x10, v);
}

void hw_sif_iop(int reg, uint32_t v, bool set) {
    std::lock_guard<std::mutex> lk(g_hw);
    if (reg < 1 || reg > 4) return;
    if (set) g_sif[reg] = v;
    else g_sif[reg] |= v;
}

void hw_set_sif0_source(bool (*source)()) {
    std::lock_guard<std::mutex> lk(g_hw);
    g_sif0_source = source;
}

void hw_sif0_kick() {
    {
        std::lock_guard<std::mutex> lk(g_hw);
        sif0_try_locked();
    }
    kernel_wake();
}

uint64_t ee_hw_read(uint32_t a, int size) {
    uint32_t p = a & 0x1FFFFFFFu;
    // INTC_STAT and the GS CSR are polled in tight loops (the SDK's VSync,
    // DW2's vblank handler): read without the lock, so that the clock thread,
    // which changes them, is never starved of it.
    if (g_trace < 2 && (size == 4 || size == 8)) {
        if (p == 0x1000F000) return g_intc_stat.load();
        if (p == 0x12001000) return size == 4 ? (uint32_t)g_gs_csr.load() : g_gs_csr.load();
    }
    std::lock_guard<std::mutex> lk(g_hw);
    uint64_t v = hw_read_locked(p & ~3u);
    if (size < 8 && (p & 3)) v >>= (p & 3) * 8;
    if (size < 8) v &= size == 4 ? 0xFFFFFFFFull : size == 2 ? 0xFFFFull : 0xFFull;
    report("read", p, v);
    return v;
}

void ee_hw_write(uint32_t a, uint64_t v, int size) {
    uint32_t p = a & 0x1FFFFFFFu;
    std::lock_guard<std::mutex> lk(g_hw);
    if (size < 4 && p != 0x1000F180) rt_die("hardware write%d %08X: sub-word writes are not modelled", size * 8, a);
    report("write", p, v);
    hw_write_locked(p, v);
}

void ee_hw_write128(uint32_t a, const EEReg& v) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p == 0x10006000) {                               // the GIF FIFO: path 3 from the CPU
        std::lock_guard<std::mutex> lk(g_hw);
        uint8_t q[16];
        std::memcpy(q, &v, 16);
        g_dma_qw[2] += 1;
        gs_gif(3, q, 1);
        gs_events_locked();
        return;
    }
    if (p == 0x10007010) {                               // the IPU's input FIFO
        std::lock_guard<std::mutex> lk(g_hw);
        uint8_t q[16];
        std::memcpy(q, &v, 16);
        ipu_fifo_write_locked(q);
        return;
    }
    if (p >= 0x10004000 && p < 0x10008000) {             // the VIF0, VIF1 and GIF FIFOs
        static const int ch_of[4] = {0, 1, 2, 4};
        int ch = ch_of[((p >> 12) & 0xF) - 4];
        std::lock_guard<std::mutex> lk(g_hw);
        g_dma_qw[ch] += 1;
        if (ch == 0) {
            uint8_t q[16];
            std::memcpy(q, &v, 16);
            vif0_data(q, 16);
            return;
        }
        TRACE(2, "hw: %s FIFO write %016llX%016llX\n", DMA_NAME[ch], (unsigned long long)v.ud[1],
              (unsigned long long)v.ud[0]);
        return;
    }
    ee_hw_write(a, v.ud[0], 8);
    ee_hw_write(a + 8, v.ud[1], 8);
}

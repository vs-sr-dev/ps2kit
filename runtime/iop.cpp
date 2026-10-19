// ps2kit runtime — the IOP, at the SIF boundary.
//
// The IOP's modules are not run. The EE side of the SIF (the SDK's sifcmd
// and sifrpc, recompiled) talks to this file the way it would talk to the
// IOP's own sifcmd and sifrpc modules, through the same registers, DMA
// channels and packets; the RPC servers behind them are host code.
//
// Packets, as the EE's code builds and reads them: a 16-byte header (byte 0
// the packet size, bits 8-31 of word 0 the size of a data part sent ahead
// of the packet, word 1 that data's destination, word 2 the command id,
// word 3 an option), then the command's fields.
//
//   EE -> IOP   sceSifSetDma into the IOP's command buffer (SMCOM), with
//               the "interrupt the IOP" attribute; a data part goes first,
//               to its own destination.
//   IOP -> EE   through SIF0 into the EE's receive buffer (the address the
//               EE gives with command 0x80000002 or 0x80000000), when the
//               EE has armed the channel; the EE's DMAC channel 5 handler
//               copies the packet out, re-arms, and dispatches on the id.
//
// System commands handled here: 0x80000000 (the EE's receive buffer moved),
// 0x80000001 (set a software register), 0x80000002 (init: option 0 gives
// the receive buffer; option 1 is sceSifInitRpc, which waits for its
// software register 0 to be set from the IOP), 0x80000003 (reset the IOP
// with a reboot string), 0x80000009 (bind an RPC server), 0x8000000A
// (call one). Replies to binds and calls are 0x80000008 packets: the
// client's address, the id of the command they end, the server, its
// receive buffers. SMFLG bits: 0x10000 SIF up, 0x20000 commands up,
// 0x40000 boot finished.
#include "os.h"

#include <algorithm>
#include <ctime>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr uint32_t IOP_RAM = 2u << 20;
constexpr uint32_t IOP_CMDBUF = 0x00010000;           // the IOP's command buffer (our choice)
constexpr uint32_t SMFLG_SIF = 0x10000, SMFLG_CMD = 0x20000, SMFLG_BOOTEND = 0x40000;

std::mutex g_iop;                                     // never held while taking the hardware's lock
std::vector<uint8_t> g_ram(IOP_RAM);
uint32_t g_ee_cmdbuf;                                 // the EE's receive buffer (0: not told yet)
uint32_t g_sreg[32];
uint32_t g_alloc;                                     // IOP RAM for server data, bump-allocated
double g_reboot_done = -1;                            // host time the reboot completes (<0: none)

struct Out {
    uint32_t dest = 0;                                // EE address of the data part
    std::vector<uint8_t> data, pkt;
};
std::deque<Out> g_out;

uint32_t ld32(uint32_t a) { uint32_t v; std::memcpy(&v, &g_ram[a & (IOP_RAM - 1)], 4); return v; }
void st32(uint32_t a, uint32_t v) { std::memcpy(&g_ram[a & (IOP_RAM - 1)], &v, 4); }
std::string cstr(uint32_t a, size_t max) {
    std::string s;
    for (size_t k = 0; k < max; ++k) {
        char ch = (char)g_ram[(a + k) & (IOP_RAM - 1)];
        if (!ch) break;
        s += ch;
    }
    return s;
}
// IOP RAM: the command buffer at 0x10000; the servers' own data from
// 0x100000 (bump-allocated, never freed); the heap the EE allocates from
// (sceSifAllocIopHeap) from 0x140000 to the end.
constexpr uint32_t SERVER_END = 0x00140000, HEAP_START = 0x00140000, HEAP_END = IOP_RAM - 0x1000;
uint32_t alloc(uint32_t size) {
    uint32_t a = g_alloc;
    g_alloc = (g_alloc + size + 63) & ~63u;
    if (g_alloc > SERVER_END) rt_die("IOP RAM for server data exhausted");
    return a;
}

// The heap: first fit over a list of blocks, 64-byte aligned.
std::map<uint32_t, uint32_t> g_heap_used;            // address -> size
uint32_t heap_alloc(uint32_t size) {
    size = (size + 63) & ~63u;
    uint32_t at = HEAP_START;
    for (auto& b : g_heap_used) {
        if (b.first - at >= size) break;
        at = b.first + b.second;
    }
    if (at + size > HEAP_END) return 0;
    g_heap_used[at] = size;
    return at;
}
bool heap_free(uint32_t addr) { return g_heap_used.erase(addr) == 1; }

// ---- RPC servers -------------------------------------------------------------------------------
// A call runs fn(function number, the server's receive buffer, bytes sent)
// and returns the IOP address of the data to send back.
struct Server {
    const char* name;
    uint32_t (*fn)(int fno, uint32_t buff, uint32_t size);
    uint32_t sd = 0, buff = 0;                        // the server's data and receive buffer, in IOP RAM
    bool up = false;
    int failed_binds = 0;
};
std::map<uint32_t, Server> g_servers;

uint32_t sv_loadfile(int fno, uint32_t buff, uint32_t size);
uint32_t sv_cd_init(int fno, uint32_t buff, uint32_t size);
uint32_t sv_mcserv(int fno, uint32_t buff, uint32_t size);
uint32_t sv_padman(int fno, uint32_t buff, uint32_t size);
uint32_t sv_cd_search(int fno, uint32_t buff, uint32_t size);
uint32_t sv_cd_ready(int fno, uint32_t buff, uint32_t size);
uint32_t sv_cd_ncmd(int fno, uint32_t buff, uint32_t size);
uint32_t sv_cd_scmd(int fno, uint32_t buff, uint32_t size);
uint32_t sv_koei_sound(int fno, uint32_t buff, uint32_t size);
uint32_t sv_koeiads(int fno, uint32_t buff, uint32_t size);
uint32_t sv_iopheap(int fno, uint32_t buff, uint32_t size);
extern uint32_t g_zero_reply, g_ads_reply;
void ee_write(uint32_t addr, const void* p, size_t n);

void server(uint32_t sid, const char* name, uint32_t (*fn)(int, uint32_t, uint32_t) = nullptr) {
    Server& s = g_servers[sid];
    s.name = name;
    s.fn = fn;
    if (!s.sd) {
        s.sd = alloc(0x40);
        s.buff = alloc(0x4000);
    }
}

// The modules each server comes from, read from the modules' own code (the
// constant each passes to sceSifRegisterRpc): CDVDFSV 0x80000592-0x8000059A,
// PADMAN 0x8000010F and 0x8000011F, MCSERV 0x80000400, KESSOUND 0x4000009A,
// KOEIADS 0x40000099. FILEIO (0x80000001, and 0x80000003 for the IOP heap)
// and CDVDFSV are in the game's IOPRP images too; LOADFILE (0x80000006) is
// the boot ROM's.
const struct { const char* module; uint32_t sid; } MODULE_SERVERS[] = {
    {"CDVDFSV", 0x80000592}, {"CDVDFSV", 0x80000593}, {"CDVDFSV", 0x80000595}, {"CDVDFSV", 0x80000597},
    {"CDVDFSV", 0x8000059A}, {"PADMAN", 0x8000010F}, {"PADMAN", 0x8000011F}, {"MCSERV", 0x80000400},
    {"KESSOUND", 0x4000009A}, {"KOEIADS", 0x40000099}, {"FILEIO", 0x80000001}, {"FILEIO", 0x80000003},
    {"LOADFILE", 0x80000006},
};

void module_up(const std::string& name) {
    for (auto& m : MODULE_SERVERS)
        if (name == m.module) g_servers[m.sid].up = true;
}

void boot_modules() {
    for (auto& s : g_servers) s.second.up = false;
    module_up("LOADFILE");
    module_up("FILEIO");
    module_up("CDVDFSV");
}

void init_servers() {
    g_alloc = 0x00100000;
    server(0x80000001, "fileio");
    server(0x80000003, "iopheap", sv_iopheap);
    server(0x80000006, "loadfile", sv_loadfile);
    server(0x80000592, "cdvdfsv init", sv_cd_init);
    server(0x80000593, "cdvdfsv s-command", sv_cd_scmd);
    server(0x80000595, "cdvdfsv n-command", sv_cd_ncmd);
    server(0x80000597, "cdvdfsv search file", sv_cd_search);
    server(0x8000059A, "cdvdfsv disk ready", sv_cd_ready);
    server(0x8000010F, "padman", sv_padman);
    server(0x8000011F, "padman ext");
    server(0x80000400, "mcserv", sv_mcserv);
    server(0x4000009A, "kessound", sv_koei_sound);
    server(0x40000099, "koeiads", sv_koeiads);
    g_zero_reply = alloc(64);
    g_ads_reply = alloc(16);
    boot_modules();
}

// ---- the modules on the disc -------------------------------------------------------------------
// cdrom0:\DIR\NAME.EXT;1 -> \DIR\NAME.EXT;1 on the disc ("" for another device)
std::string disc_path(const std::string& p) {
    size_t colon = p.find(':');
    std::string dev = p.substr(0, colon);
    if (colon == std::string::npos || (dev != "cdrom0" && dev != "cdrom")) return "";
    return p.substr(colon + 1);
}

// A loaded module, as its own header (.iopmod) names it.
struct Module { std::string file, name; uint16_t version; };
std::vector<Module> g_modules;

bool read_iopmod(const std::string& path, Module& m) {
    std::vector<uint8_t> d;
    if (!disc_read_file(path, d)) return false;
    auto u16 = [&](size_t o) { uint16_t v = 0; if (o + 2 <= d.size()) std::memcpy(&v, &d[o], 2); return v; };
    auto u32 = [&](size_t o) { uint32_t v = 0; if (o + 4 <= d.size()) std::memcpy(&v, &d[o], 4); return v; };
    if (d.size() < 52 || std::memcmp(d.data(), "\x7F" "ELF", 4) != 0) return false;
    uint32_t shoff = u32(0x20);
    uint16_t shnum = u16(0x30), shstrndx = u16(0x32);
    uint32_t strs = u32(shoff + 40u * shstrndx + 16);
    for (uint16_t k = 0; k < shnum; ++k) {
        size_t sh = shoff + 40u * k;
        size_t name = strs + u32(sh);
        if (name >= d.size() || std::strcmp((const char*)&d[name], ".iopmod") != 0) continue;
        size_t at = u32(sh + 16);
        m.version = u16(at + 24);
        for (size_t c = at + 26; c < d.size() && d[c]; ++c) m.name += (char)d[c];
        return true;
    }
    return false;
}

const Module* module_named(const char* name) {
    for (auto& m : g_modules)
        if (m.name == name) return &m;
    return nullptr;
}

// loadfile: function 0 loads a module. The EE sends {argument length, -,
// path[252], arguments[252]} and reads one word back: the module's id, or
// a negative error. The module is read from the disc for its header; its
// code does not run, the servers it would register come up.
int g_module_ids = 1;
uint32_t sv_loadfile(int fno, uint32_t buff, uint32_t) {
    if (fno != 0) rt_die("IOP: loadfile function %d is not modelled yet", fno);
    std::string path = cstr(buff + 8, 252);
    std::string file = disc_path(path);
    Module m{file, "", 0};
    if (file.empty() || !read_iopmod(file, m)) rt_die("IOP: cannot load module %s (%s)", path.c_str(), file.c_str());
    std::string base = path.substr(path.find_last_of("\\/:") + 1);
    base = base.substr(0, base.find_first_of(".;"));
    module_up(base);
    g_modules.push_back(m);
    TRACE(1, "[iop] load module %s: %s v%X.%02X\n", path.c_str(), m.name.c_str(), m.version >> 8, m.version & 0xFF);
    st32(buff, (uint32_t)++g_module_ids);
    return buff;
}

// mcserv. Function 0xFE (sceMcInit) answers three words: mcman's export 37
// (which returns 0), mcserv's own version and mcman's, both as the
// modules' headers carry them (read in MCSERV.IRX and MCMAN.IRX v2.06/v2.07).
uint32_t sv_mcserv(int fno, uint32_t buff, uint32_t) {
    if (fno == 0xFE) {
        const Module* serv = module_named("mcserv");
        const Module* man = module_named("mcman_tool");
        if (!serv || !man) rt_die("IOP: sceMcInit without mcserv and mcman loaded");
        st32(buff, 0);
        st32(buff + 4, serv->version);
        st32(buff + 8, man->version);
        return buff;
    }
    if (fno == 1) {
        // sceMcGetInfo. The request: port +4, slot +8, whether to answer the
        // format +0xC, the free space +0x10, the type +0x14, and at +0x1C the
        // EE address of a 0xC0-byte area that mcserv sends with its own SIF
        // DMA: type at +0, free clusters at +4, formatted (0/1) at +0x90
        // (mcserv 0xA24; libmc's mceGetInfoApdx reads the same offsets).
        // The slots are empty. With no card, mcman's detection (export 21)
        // probes for a PS2 card (0x2A48: the SIO2 status lacks 0x1000, so
        // -11 from 0x2C30), then a PS1 card (0x8A94: five tries, -11), and
        // clears the slot's type and format; mcserv then skips the free
        // space and the format and answers -11, with a type of 0.
        uint32_t port = ld32(buff + 4), slot = ld32(buff + 8), dest = ld32(buff + 0x1C);
        TRACE(1, "[iop] sceMcGetInfo port %u slot %u: no card\n", port, slot);
        std::vector<uint8_t> info(0xC0, 0);
        ee_write(dest, info.data(), info.size());
        st32(buff, (uint32_t)-11);
        return buff;
    }
    rt_die("IOP: mcserv function %d is not modelled yet", fno);
}

// cdvdfsv's init server: function 0 is sceCdInit's mode (0 init, 1 init
// without a disc check, 5 exit); nothing comes back.
uint32_t sv_cd_init(int fno, uint32_t buff, uint32_t) {
    if (fno != 0) rt_die("IOP: cdvdfsv init function %d is not modelled yet", fno);
    TRACE(1, "[iop] sceCdInit mode %u\n", ld32(buff));
    return buff;
}

// padman (the library's v2.01 protocol, read from libpad in the game). Every
// request is function 1 with a 128-byte block: command at +0, port +4,
// slot +8, arguments after; the answer is read at +0xC (init, open) or at
// +0x14 (the info and mode commands, whose arguments fill +0xC and +0x10).
// An open port gets two 128-byte frames at the address the EE gives;
// padman writes the older one each vertical blank: +0 a frame counter (the
// higher is current), +4 the pad's state (6 stable), +5 the state of the
// last request (0 complete), +8 the pad's 32-byte answer, +0x28 its length.
//
// Port 0 holds a DualShock 2 driven from the host (input.cpp); it starts
// digital (ID 4, answer 0x41) as the real one does and turns analog (ID 7,
// 0x73) when the game sets that mode. Port 1 is empty.
struct Pad {
    bool open = false, present = false;
    uint32_t frames = 0;
    int mode = 0;
    uint8_t act_align[6] = {}, act_direct[6] = {};   // the motors (not driven yet)
};
Pad g_pad[2];

int pad_id(const Pad& p) { return p.mode ? 7 : 4; }

uint32_t sv_padman(int fno, uint32_t buff, uint32_t) {
    if (fno != 1) rt_die("IOP: padman function %d is not modelled", fno);
    uint32_t cmd = ld32(buff), port = ld32(buff + 4), slot = ld32(buff + 8);
    Pad* p = port < 2 && slot == 0 ? &g_pad[port] : nullptr;
    int32_t r;
    switch (cmd) {
    case 0x100:                                       // init
        g_pad[0].present = true;
        st32(buff + 0xC, 1);
        return buff;
    case 0x80000100:                                  // open (port, slot, frames)
        if (!p) { st32(buff + 0xC, 0); return buff; }
        p->open = true;
        p->frames = ld32(buff + 0x10);
        st32(buff + 0xC, 1);
        TRACE(1, "[iop] pad port %u open, frames at %08X\n", port, p->frames);
        return buff;
    case 0x80000102: {                                // info act (actuator, term)
        // A DualShock 2 has two motors; its answers to the actuator query
        // (command 0x46) are 01 02 00 0A for the small one and 01 01 01 14
        // for the large one: function, sub-function, data size, current
        // (public descriptions of the pad protocol; to check against PCSX2).
        static const int ACT[2][4] = {{1, 2, 0, 10}, {1, 1, 1, 20}};
        int32_t act = (int32_t)ld32(buff + 0xC), term = (int32_t)ld32(buff + 0x10);
        if (!p || !p->present || !p->mode) r = 0;
        else if (act == -1) r = 2;
        else if (act >= 0 && act < 2 && term >= 1 && term <= 4) r = ACT[act][term - 1];
        else r = -1;
        break;
    }
    case 0x80000104: {                                // info mode (term, offset)
        int term = (int)ld32(buff + 0xC), offs = (int)ld32(buff + 0x10);
        if (!p || !p->present) r = 0;
        else if (term == 1) r = pad_id(*p);           // current ID
        else if (term == 2) r = 0;                    // extended ID: none until pressure mode
        else if (term == 3) r = p->mode;              // current mode's place in the table
        else if (term == 4) r = offs == -1 ? 2 : offs == 0 ? 4 : offs == 1 ? 7 : 0;
        else r = 0;
        break;
    }
    case 0x80000106:                                  // actuators, direct: six bytes (the motors' values)
    case 0x80000107:                                  // actuators, alignment: six bytes
        if (!p || !p->present || !p->mode) { r = 0; break; }
        for (int b = 0; b < 6; ++b) (cmd == 0x80000106 ? p->act_direct : p->act_align)[b] = g_ram[buff + 0xC + b];
        r = 1;
        break;
    case 0x80000105: {                                // set main mode (mode, lock)
        if (!p || !p->present) { r = 0; break; }
        p->mode = ld32(buff + 0xC) ? 1 : 0;
        TRACE(1, "[iop] pad port %u mode %s (lock %u)\n", port, p->mode ? "analog" : "digital", ld32(buff + 0x10));
        r = 1;
        break;
    }
    default:
        rt_die("IOP: padman command %08X is not modelled yet", cmd);
    }
    st32(buff + 0x14, (uint32_t)r);
    return buff;
}

void pad_vblank() {
    for (int k = 0; k < 2; ++k) {
        Pad& p = g_pad[k];
        if (!p.open) continue;
        uint8_t* f0 = g_ee_ram + (p.frames & 0x01FFFFFFu);
        uint8_t* f1 = f0 + 0x80;
        uint32_t c0, c1;
        std::memcpy(&c0, f0, 4);
        std::memcpy(&c1, f1, 4);
        uint8_t* f = c0 <= c1 ? f0 : f1;
        uint32_t next = (c0 > c1 ? c0 : c1) + 1;
        std::memcpy(f, &next, 4);
        f[4] = p.present ? 6 : 0;
        f[5] = 0;
        uint8_t data[32];
        std::memset(data, 0xFF, sizeof data);
        uint32_t len = 0;
        if (p.present) {
            data[0] = 0;
            data[1] = p.mode ? 0x73 : 0x41;
            if (k == 0) input_pad(data + 2);              // buttons (active low), sticks
            else data[2] = data[3] = 0xFF;
            len = 2 + 2 * (data[1] & 0xF);
        }
        std::memcpy(f + 8, data, 32);
        std::memcpy(f + 0x28, &len, 4);
    }
}

// Writes into EE memory, as the IOP's modules do with their own SIF DMA.
// Called inside the EE's syscall, between its instructions.
void ee_write(uint32_t addr, const void* p, size_t n) {
    uint32_t a = addr & 0x01FFFFFFu;
    if (a + n > EE_RAM_SIZE) rt_die("IOP: write to EE %08X + %zu beyond RAM", addr, n);
    std::memcpy(g_ee_ram + a, p, n);
}

// cdvdfsv's file search (sceCdSearchFile): the EE sends a 0x124-byte block,
// 32 bytes for the answer (sector, size, name[16], date[8]), the path
// (256), and at +0x120 the block's own EE address, where the answer is
// written before the call ends. One word comes back: 1 found, 0 not.
// The date is left zero (DW2's InitFilePosInfo keeps only the sector).
uint32_t sv_cd_search(int fno, uint32_t buff, uint32_t) {
    if (fno != 0) rt_die("IOP: cdvdfsv search function %d is not modelled yet", fno);
    std::string path = cstr(buff + 0x20, 256);
    uint32_t ee = ld32(buff + 0x120);
    DiscFile f;
    uint32_t found = disc_find(path, f) && !f.is_dir;
    if (found) {
        uint8_t out[32] = {};
        std::memcpy(out, &f.lsn, 4);
        std::memcpy(out + 4, &f.size, 4);
        std::memcpy(out + 8, f.name, 16);
        ee_write(ee, out, sizeof out);
    }
    TRACE(1, "[iop] sceCdSearchFile %s: %s sector %u, %u bytes\n", path.c_str(), found ? "found," : "not found",
          found ? f.lsn : 0, found ? f.size : 0);
    st32(buff, found);
    return buff;
}

// cdvdfsv's disk-ready server (sceCdDiskReady): function 0, the mode in;
// one word out, 2 when the drive is ready (the game's readers wait for
// exactly 2; libcdvd itself answers 6, not ready, when the call fails).
uint32_t sv_cd_ready(int fno, uint32_t buff, uint32_t) {
    if (fno != 0) rt_die("IOP: cdvdfsv disk-ready function %d is not modelled yet", fno);
    st32(buff, 2);
    return buff;
}

uint64_t g_cd_reads, g_cd_sectors;

// Streaming (sceCdStream, n-command 9): {sector, count, EE buffer, command,
// mode (tries, spindle, data pattern) at +0x10}; one word back. CDVDFSV
// v2.08 (in IOPRP153.IMG, 0x20D4) keeps the stream itself: a ring in the
// IOP buffer the EE gives at init, cut into banks, filled from the drive
// ahead of the reader. Init (5: total sectors, banks, IOP buffer) answers 1;
// start (1: sector, mode) and seek (4: sector) empty the ring, move to the
// sector and answer 1; stop (3) empties it and answers 1; read (2: count,
// EE buffer) copies what the ring holds, at most the count, by SIF DMA,
// and answers the sectors copied in the low half, an error in the high one.
// Here the drive is infinitely fast: the ring is always full, so a read
// gets all it asks for, straight from the disc at the stream's position.
// Stat (6) is not modelled: with a full ring CDVDFSV's count stops at once
// (the fill index has caught up with the read index) and answers 0.
struct CdStream {
    bool init = false, started = false;
    uint32_t lsn = 0, sectors = 0, banks = 0;
};
CdStream g_st;

uint32_t cd_stream(uint32_t buff) {
    uint32_t a0 = ld32(buff), a1 = ld32(buff + 4), a2 = ld32(buff + 8), cmd = ld32(buff + 12);
    uint32_t r = 0;
    switch (cmd) {
    case 5:                                           // init: total sectors, banks, IOP buffer
        if (a1 == 0 || a0 / a1 == 0) rt_die("IOP: sceCdStInit with %u sectors in %u banks", a0, a1);
        g_st = CdStream{true, false, 0, a0, a1};
        TRACE(1, "[iop] sceCdStInit %u sectors, %u banks, buffer %06X\n", a0, a1, a2);
        r = 1;
        break;
    case 1:                                           // start: sector, mode
    case 4: {                                         // seek: sector
        if (!g_st.init) rt_die("IOP: sceCdStream command %u before sceCdStInit", cmd);
        uint32_t pattern = g_ram[(buff + 0x12) & (IOP_RAM - 1)];
        if (cmd == 1 && pattern != 0)
            rt_die("IOP: sceCdStStart with data pattern %u is not modelled yet", pattern);
        g_st.lsn = a0;
        g_st.started = true;
        TRACE(1, "[iop] sceCdSt%s sector %u\n", cmd == 1 ? "Start" : "Seek", a0);
        r = 1;
        break;
    }
    case 3:                                           // stop
        g_st.started = false;
        TRACE(1, "[iop] sceCdStStop\n");
        r = 1;
        break;
    case 2: {                                         // read: count, EE buffer
        if (!g_st.started) rt_die("IOP: sceCdStRead with no stream started");
        std::vector<uint8_t> data((size_t)a1 * 2048);
        if (!disc_read(g_st.lsn, a1, data.data()))
            rt_die("IOP: sceCdStRead of sectors %u + %u beyond the disc", g_st.lsn, a1);
        ee_write(a2, data.data(), data.size());
        TRACE(2, "[iop] sceCdStRead sector %u + %u -> %08X\n", g_st.lsn, a1, a2);
        g_st.lsn += a1;
        ++g_cd_reads;
        g_cd_sectors += a1;
        r = a1 & 0xFFFF;
        break;
    }
    default:
        rt_die("IOP: sceCdStream command %u is not modelled yet", cmd);
    }
    st32(buff, r);
    return buff;
}

// cdvdfsv's n-command server. Function 14 (sceCdNcmdDiskReady) answers one
// word: 2, ready (libcdvd's sceCdRead gives up on 6). Function 1 is
// sceCdRead, sent without waiting: {sector, count, EE buffer, mode (tries,
// spindle, data pattern), an EE area of 0x90 bytes, an EE result word}.
// The IOP sends the 64-byte-aligned part of the data by DMA and leaves the
// unaligned head and tail in the area ({head size, tail size, head address,
// tail address, head bytes at +0x10, tail bytes at +0x50}), which the
// EE's end-of-call function (cd_read_intr) copies into place. Here all the
// data goes straight into the buffer and both sizes are zero.
uint32_t sv_cd_ncmd(int fno, uint32_t buff, uint32_t) {
    if (fno == 14) {
        st32(buff, 2);
        return buff;
    }
    if (fno == 1) {
        uint32_t lsn = ld32(buff), count = ld32(buff + 4), dst = ld32(buff + 8);
        uint32_t pattern = g_ram[(buff + 0xE) & (IOP_RAM - 1)], area = ld32(buff + 0x10);
        if (pattern != 0) rt_die("IOP: sceCdRead with data pattern %u (sectors of %s bytes) is not modelled yet",
                                 pattern, pattern == 1 ? "2328" : "2340");
        std::vector<uint8_t> data((size_t)count * 2048);
        if (!disc_read(lsn, count, data.data())) rt_die("IOP: sceCdRead of sectors %u + %u beyond the disc", lsn, count);
        ee_write(dst, data.data(), data.size());
        uint32_t zero[2] = {0, 0};
        ee_write(area, zero, sizeof zero);
        TRACE(1, "[iop] sceCdRead sector %u + %u -> %08X\n", lsn, count, dst);
        ++g_cd_reads;
        g_cd_sectors += count;
        return buff;
    }
    if (fno == 9) return cd_stream(buff);
    rt_die("IOP: cdvdfsv n-command function %d is not modelled yet", fno);
}

// cdvdfsv's s-command server (the drive's own commands, synchronous).
// Function 4 is sceCdGetError: 0, no error (reads here do not fail).
// Function 1 is sceCdReadClock: CDVDFSV v2.08 (0x30C0) answers the word
// cdvdman's sceCdReadClock returns, then the 8-byte clock. CDVDMAN v2.09
// (0x28C8) sends the mechanism's S-command 8 for those 8 bytes, clears byte
// 4 and bit 7 of byte 6, and returns 1 when the command completed (0xE48).
// The bytes: the command's status (0, done), then second, minute, hour,
// 0, day, month, year, in BCD. The console's clock keeps Japan time
// (UTC+9), which the SDK's libraries convert from; here it is the host's
// UTC plus nine hours.
uint8_t bcd(int v) { return (uint8_t)(((v / 10) << 4) | (v % 10)); }
uint32_t sv_cd_scmd(int fno, uint32_t buff, uint32_t) {
    if (fno == 4) {
        st32(buff, 0);
        return buff;
    }
    if (fno == 1) {
        std::time_t now = std::time(nullptr) + 9 * 3600;
        std::tm t = *std::gmtime(&now);
        uint8_t clock[8] = {0, bcd(t.tm_sec), bcd(t.tm_min), bcd(t.tm_hour), 0, bcd(t.tm_mday),
                            bcd(t.tm_mon + 1), bcd(t.tm_year % 100)};
        st32(buff, 1);
        std::memcpy(&g_ram[buff + 4], clock, 8);
        TRACE(1, "[iop] sceCdReadClock: 20%02X-%02X-%02X %02X:%02X:%02X (Japan time)\n", clock[7], clock[6],
              clock[5], clock[3], clock[2], clock[1]);
        return buff;
    }
    rt_die("IOP: cdvdfsv s-command function %d is not modelled yet", fno);
}

// The IOP heap server (sceSifAllocIopHeap, sceSifFreeIopHeap): function 1
// takes a size and answers an IOP address (0 when there is no room);
// function 2 takes an address and answers 0 (or -1 if it was not allocated).
uint32_t sv_iopheap(int fno, uint32_t buff, uint32_t) {
    if (fno == 1) {
        uint32_t size = ld32(buff), a = heap_alloc(size);
        TRACE(1, "[iop] heap alloc %u bytes -> %06X\n", size, a);
        st32(buff, a);
        return buff;
    }
    if (fno == 2) {
        uint32_t a = ld32(buff);
        st32(buff, heap_free(a) ? 0 : (uint32_t)-1);
        return buff;
    }
    rt_die("IOP: iopheap function %d is not modelled yet", fno);
}

// Koei's two sound drivers (KESSOUND.IRX, "kessen_sound_driver", and
// KOEIADS.IRX), a silent stand-in until their own logic is ported (phase 5).
// Their EE libraries send each command with the command code as the RPC
// function number (kessound 0x9A00-0x9A1A, koeiads 0x99xx) and read a reply
// of 4, 12 or 20 bytes whose first word is the result. Every reply here is
// zeros: success, with nothing playing. For kessound's init (0x9A00) the
// driver's own code confirms it: 0 on success, -3, -9 or -99 on failure.
uint32_t g_zero_reply;                                // 64 zero bytes in IOP RAM
uint32_t g_ads_reply;                                 // koeiads host transfer: {result, status, value}
uint32_t sv_koei_sound(int fno, uint32_t, uint32_t size) {
    static std::vector<int> seen;
    if (std::find(seen.begin(), seen.end(), fno) == seen.end()) {
        seen.push_back(fno);
        TRACE(1, "[iop] Koei sound command %04X (%u bytes): silent stand-in, answered 0\n", (unsigned)fno, size);
    }
    return g_zero_reply;
}

// KOEIADS's host transfer, by which the movie player hands its audio (SPU
// ADPCM from the PSS) to the IOP. The buffer arithmetic is KOEIADS v1.05's:
// 0x9901 (SPU address, H, n) checks H (a multiple of 0x800, at least 0x800)
// and allocates a host buffer of 2H (0x2FAC); the SPU side holds n * H.
// 0x990F answers 2H (0x1B74); 0x9910 takes the rate and the interleave;
// 0x9914 answers the host buffer's address while the SPU side has room for
// H more bytes, 0 otherwise (0x1E08); 0x9915 queues the bytes the EE has
// written there (0x1E6C, a multiple of twice the interleave); 0x9912 starts
// playing, 0x9913 stops. Each word comes back as {result, status, value}.
// The SPU2's playback, which frees H bytes at a time, is replaced by a clock
// at the stream's byte rate: ADPCM, 16 bytes per 28 samples, two channels.
struct KoeiAds {
    uint32_t spu = 0, half = 0, blocks = 0, host = 0;
    uint32_t rate = 0, interleave = 0;
    uint64_t queued = 0;                              // bytes on the SPU side not yet played
    bool playing = false;
    double played_to = 0;                             // host time the playback clock last advanced
};
KoeiAds g_ads;

void ads_clock() {
    if (!g_ads.playing || !g_ads.rate) return;
    double now = hw_now(), bytes_per_s = g_ads.rate * 16.0 / 28.0 * 2;
    double secs_per_half = g_ads.half / bytes_per_s;
    while (g_ads.queued >= g_ads.half && now - g_ads.played_to >= secs_per_half) {
        g_ads.queued -= g_ads.half;
        g_ads.played_to += secs_per_half;
    }
    if (g_ads.queued < g_ads.half) g_ads.played_to = now;
}

uint32_t sv_koeiads(int fno, uint32_t buff, uint32_t size) {
    uint32_t a0 = ld32(buff), a1 = ld32(buff + 4), a2 = ld32(buff + 8);
    int32_t result = 0;
    uint32_t value = 0;
    ads_clock();
    switch (fno) {
    case 0x9901:
        if (a1 < 0x800 || (a1 & 0x7FF) || a2 == 0 || a0 + 2 * a1 > 0x1FFFFF) { result = -2; break; }
        g_ads.spu = a0;
        g_ads.half = a1;
        g_ads.blocks = a2;
        if (!g_ads.host) g_ads.host = alloc(2 * a1);
        TRACE(1, "[iop] koeiads buffers: SPU %06X, %u blocks of %X, host %06X\n", a0, a2, a1, g_ads.host);
        break;
    case 0x990F:                                      // host transfer init
        if (!g_ads.host) { result = -10; break; }
        g_ads.queued = 0;
        g_ads.playing = false;
        value = 2 * g_ads.half;
        break;
    case 0x9910:                                      // rate, interleave
        g_ads.rate = a0;
        g_ads.interleave = a1;
        TRACE(1, "[iop] koeiads host transfer: %u Hz, interleave %u\n", a0, a1);
        break;
    case 0x9911: g_ads.playing = false; break;        // term
    case 0x9912:                                      // start
        g_ads.playing = true;
        g_ads.played_to = hw_now();
        break;
    case 0x9913: g_ads.playing = false; break;        // stop
    case 0x9914:                                      // buffer position
        if ((uint64_t)g_ads.blocks * g_ads.half >= g_ads.queued + g_ads.half) value = g_ads.host;
        break;
    case 0x9915:                                      // bytes written
        if (a0 == 0 || a0 > 2 * g_ads.half || (g_ads.interleave && a0 % (2 * g_ads.interleave))) { result = -2; break; }
        g_ads.queued += a0;
        break;
    default:
        return sv_koei_sound(fno, buff, size);
    }
    st32(g_ads_reply, (uint32_t)result);
    st32(g_ads_reply + 4, 0);
    st32(g_ads_reply + 8, value);
    return g_ads_reply;
}

// ---- packets to the EE------------------------------------------------------------------------
void send(uint32_t cid, std::vector<uint8_t> pkt, uint32_t dest = 0, std::vector<uint8_t> data = {}) {
    pkt.resize((pkt.size() + 15) & ~size_t(15));
    uint32_t w0 = (uint32_t)pkt.size() | ((uint32_t)data.size() << 8);
    std::memcpy(&pkt[0], &w0, 4);
    std::memcpy(&pkt[4], &dest, 4);
    std::memcpy(&pkt[8], &cid, 4);
    TRACE(1, "[iop] -> EE command %08X (%zu bytes, data %zu to %08X)\n", cid, pkt.size(), data.size(), dest);
    g_out.push_back(Out{dest, std::move(data), std::move(pkt)});
}

// Called by the hardware, with its lock held, when SIF0 is armed: one
// packet (and its data part) into EE memory. Returns false if none waits.
bool sif0_source() {
    std::lock_guard<std::mutex> lk(g_iop);
    if (g_out.empty() || !g_ee_cmdbuf) return false;
    Out o = std::move(g_out.front());
    g_out.pop_front();
    if (!o.data.empty()) std::memcpy(g_ee_ram + (o.dest & 0x01FFFFFFu), o.data.data(), o.data.size());
    std::memcpy(g_ee_ram + (g_ee_cmdbuf & 0x01FFFFFFu), o.pkt.data(), o.pkt.size());
    return true;
}

std::vector<uint8_t> words(std::initializer_list<uint32_t> w) {
    std::vector<uint8_t> v(w.size() * 4);
    size_t k = 0;
    for (uint32_t x : w) { std::memcpy(&v[k], &x, 4); k += 4; }
    return v;
}

// ---- commands from the EE -----------------------------------------------------------------------
void rpc_end(uint32_t client, uint32_t ended, uint32_t server, uint32_t buff, uint32_t cbuff,
             uint32_t dest = 0, std::vector<uint8_t> data = {}) {
    // header, rec_id, pkt_addr, rpc_id, client, cid, server, buff, cbuff
    send(0x80000008, words({0, 0, 0, 0, 0, 0, 0, client, ended, server, buff, cbuff, 0, 0, 0, 0}), dest,
         std::move(data));
}

void command(uint32_t p) {
    uint32_t cid = ld32(p + 8), opt = ld32(p + 12);
    TRACE(1, "[iop] <- EE command %08X opt %X\n", cid, opt);
    switch (cid) {
    case 0x80000000:                                  // the EE's receive buffer
        g_ee_cmdbuf = ld32(p + 0x10);
        break;
    case 0x80000001:                                  // set a software register
        g_sreg[ld32(p + 0x10) & 31] = ld32(p + 0x14);
        break;
    case 0x80000002:                                  // init
        if (opt == 0) g_ee_cmdbuf = ld32(p + 0x10);
        else send(0x80000001, words({0, 0, 0, 0, 0, 1}));   // the EE's software register 0 := 1
        break;
    case 0x80000003: {                                // reset, with the reboot string
        std::string arg = cstr(p + 0x18, ld32(p + 0x10));
        TRACE(1, "[iop] reboot: \"%s\" (mode %X)\n", arg.c_str(), ld32(p + 0x14));
        g_out.clear();
        g_ee_cmdbuf = 0;
        std::memset(g_sreg, 0, sizeof g_sreg);
        g_modules.clear();
        g_heap_used.clear();
        boot_modules();
        g_reboot_done = hw_now() + 0.05;              // after the EE has cleared the flags
        break;
    }
    case 0x80000009: {                                // bind (client at +0x1C, server id at +0x20)
        uint32_t client = ld32(p + 0x1C), sid = ld32(p + 0x20);
        auto it = g_servers.find(sid);
        if (it == g_servers.end() || !it->second.up) {
            // No such server yet: the answer is server 0, and the EE binds again.
            int& n = g_servers[sid].failed_binds;
            if (++n == 1 || g_trace >= 1)
                std::fprintf(stderr, "ps2kit: IOP: no RPC server %08X%s\n", sid,
                             it == g_servers.end() ? "" : " (its module is not loaded)");
            if (n > 50) {
                std::fflush(stdout);
                rt_die("the EE keeps binding RPC server %08X, which no loaded module provides", sid);
            }
            rpc_end(client, cid, 0, 0, 0);
            break;
        }
        TRACE(1, "[iop] bind %08X (%s)\n", sid, it->second.name);
        rpc_end(client, cid, it->second.sd, it->second.buff, it->second.buff);
        break;
    }
    case 0x8000000A: {                                // call
        uint32_t client = ld32(p + 0x1C), fno = ld32(p + 0x20), ssize = ld32(p + 0x24);
        uint32_t recv = ld32(p + 0x28), rsize = ld32(p + 0x2C), sd = ld32(p + 0x34);
        Server* s = nullptr;
        uint32_t sid = 0;
        for (auto& e : g_servers)
            if (e.second.sd == sd) { s = &e.second; sid = e.first; }
        if (!s) rt_die("IOP: RPC call to an unknown server %06X", sd);
        if (!s->fn)
            rt_die("IOP: %s (server %08X) function %u is not modelled yet (%u bytes sent, %u expected back)",
                   s->name, sid, fno, ssize, rsize);
        TRACE(1, "[iop] call %s function %u, %u bytes, %u back to %08X\n", s->name, fno, ssize, rsize, recv);
        uint32_t r = s->fn((int)fno, s->buff, ssize);
        std::vector<uint8_t> back;
        if (rsize && recv) {
            back.resize(rsize);
            for (uint32_t k = 0; k < rsize; ++k) back[k] = g_ram[(r + k) & (IOP_RAM - 1)];
        }
        rpc_end(client, cid, sd, s->buff, s->buff, recv, std::move(back));
        break;
    }
    default:
        rt_die("IOP: command %08X from the EE is not modelled", cid);
    }
}

}  // namespace

// ---- the interface ---------------------------------------------------------------------------
void iop_init() {
    std::lock_guard<std::mutex> lk(g_iop);
    init_servers();
    hw_set_sif0_source(sif0_source);
    // At the game's start the IOP is up, and the boot ROM's own reboot of it
    // has been waited for (sceSifSyncIop clears BOOTEND when it sees it).
    hw_sif_iop(2, IOP_CMDBUF, true);                  // SMCOM: where commands go
    hw_sif_iop(4, SMFLG_SIF | SMFLG_CMD, false);
}

void iop_update() {
    bool kick = false;
    {
        std::lock_guard<std::mutex> lk(g_iop);
        if (g_reboot_done >= 0 && hw_now() >= g_reboot_done) {
            g_reboot_done = -1;
            hw_sif_iop(2, IOP_CMDBUF, true);
            hw_sif_iop(4, SMFLG_SIF | SMFLG_CMD | SMFLG_BOOTEND, false);
            TRACE(1, "[iop] up again\n");
        }
        kick = !g_out.empty();
    }
    if (kick) hw_sif0_kick();
}

std::string iop_status() {
    std::lock_guard<std::mutex> lk(g_iop);
    return "CD " + std::to_string(g_cd_reads) + " reads, " + std::to_string(g_cd_sectors) + " sectors; " +
           std::to_string(g_modules.size()) + " modules";
}

// A vertical blank, seen from the guest's side (between its instructions):
// what the IOP's modules do once a frame.
void iop_vblank() {
    std::lock_guard<std::mutex> lk(g_iop);
    pad_vblank();
}

// A SIF1 transfer from the EE: the bytes land in IOP RAM; with the
// interrupt attribute, the IOP's command layer reads its buffer.
void iop_sif1(uint32_t src, uint32_t dst, uint32_t size, uint32_t attr) {
    {
        std::lock_guard<std::mutex> lk(g_iop);
        for (uint32_t b = 0; b < size; ++b) g_ram[(dst + b) & (IOP_RAM - 1)] = ee_ld8(src + b);
        if ((attr & 0x4) && (dst & (IOP_RAM - 1)) == IOP_CMDBUF) command(IOP_CMDBUF);
    }
    iop_update();
}

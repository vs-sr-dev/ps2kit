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
uint32_t alloc(uint32_t size) {
    uint32_t a = g_alloc;
    g_alloc = (g_alloc + size + 63) & ~63u;
    if (g_alloc > IOP_RAM) rt_die("IOP RAM for server data exhausted");
    return a;
}

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

void server(uint32_t sid, const char* name, uint32_t (*fn)(int, uint32_t, uint32_t) = nullptr) {
    Server& s = g_servers[sid];
    s.name = name;
    s.fn = fn;
    if (!s.sd) {
        s.sd = alloc(0x40);
        s.buff = alloc(0x10000);
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
    server(0x80000003, "iopheap");
    server(0x80000006, "loadfile", sv_loadfile);
    server(0x80000592, "cdvdfsv init", sv_cd_init);
    server(0x80000593, "cdvdfsv s-command");
    server(0x80000595, "cdvdfsv n-command");
    server(0x80000597, "cdvdfsv search file");
    server(0x8000059A, "cdvdfsv disk ready");
    server(0x8000010F, "padman");
    server(0x8000011F, "padman ext");
    server(0x80000400, "mcserv", sv_mcserv);
    server(0x4000009A, "kessound");
    server(0x40000099, "koeiads");
    boot_modules();
}

// ---- the disc and the modules on it -------------------------------------------------------------
std::string g_disc;

// cdrom0:\DIR\NAME.EXT;1 -> <disc>/DIR/NAME.EXT
std::string disc_path(const std::string& p) {
    size_t colon = p.find(':');
    std::string dev = p.substr(0, colon), rest = p.substr(colon + 1);
    if (dev != "cdrom0" && dev != "cdrom") return "";
    rest = rest.substr(0, rest.find(';'));
    for (char& ch : rest)
        if (ch == '\\') ch = '/';
    while (!rest.empty() && rest[0] == '/') rest.erase(0, 1);
    return g_disc + "/" + rest;
}

// A loaded module, as its own header (.iopmod) names it.
struct Module { std::string file, name; uint16_t version; };
std::vector<Module> g_modules;

bool read_iopmod(const std::string& path, Module& m) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> d;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n);
    std::fclose(f);
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
    rt_die("IOP: mcserv function %d is not modelled yet", fno);
}

// cdvdfsv's init server: function 0 is sceCdInit's mode (0 init, 1 init
// without a disc check, 5 exit); nothing comes back.
uint32_t sv_cd_init(int fno, uint32_t buff, uint32_t) {
    if (fno != 0) rt_die("IOP: cdvdfsv init function %d is not modelled yet", fno);
    TRACE(1, "[iop] sceCdInit mode %u\n", ld32(buff));
    return buff;
}

// ---- packets to the EE ------------------------------------------------------------------------
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
void iop_init(const char* disc) {
    std::lock_guard<std::mutex> lk(g_iop);
    g_disc = disc;
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

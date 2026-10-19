// ps2kit runtime — what the kernel (kernel.cpp), the hardware (hw.cpp) and
// the boot (boot.cpp) share. Not seen by generated code.
#pragma once
#include "rt.h"

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

// ---- the kernel ---------------------------------------------------------------------
// Starts the guest: the main thread runs `entry` on a host thread of its own.
// Returns only when the guest calls Exit (or the runtime gives up).
int  kernel_run(uint32_t entry, const char* argv0);
void kernel_wake();                      // something was raised: wake an idle kernel

// ---- the hardware -------------------------------------------------------------------
void hw_init();
void hw_start_clock();                   // vblank and the host-timed events begin
double hw_now();                         // seconds since hw_start_clock

// INTC causes (INTC_STAT / INTC_MASK bits).
enum {
    INTC_GS = 0, INTC_SBUS = 1, INTC_VBON = 2, INTC_VBOF = 3, INTC_VIF0 = 4, INTC_VIF1 = 5,
    INTC_VU0 = 6, INTC_VU1 = 7, INTC_IPU = 8, INTC_TIM0 = 9, INTC_TIM1 = 10, INTC_TIM2 = 11,
    INTC_TIM3 = 12, INTC_SFIFO = 13, INTC_VU0WD = 14,
};

// Interrupt state as the kernel's dispatcher sees it. Each `take` returns the
// pending bits (status & mask) and clears them in the status register, as
// the kernel does before it calls the handlers.
uint32_t hw_intc_take();
uint32_t hw_dmac_take();
uint32_t hw_intc_mask();
void     hw_intc_mask_set(int cause, bool on);       // returns through hw_intc_mask
uint32_t hw_dmac_mask();                             // D_STAT CIM, bits 0-9
void     hw_dmac_mask_set(int channel, bool on);
void     hw_raise_intc(int cause);
void     hw_raise_dmac(int channel);

uint32_t hw_vblank_count();              // vertical blanks since the clock started
uint64_t hw_gs_csr();
uint32_t hw_gs_imr();
void     hw_gs_set_imr(uint32_t v);

// SIF registers as sceSifGetReg/sceSifSetReg number them: 1 MSCOM, 2 SMCOM,
// 3 MSFLG, 4 SMFLG. (The software registers, 0x80000000 + n, live in the kernel.)
uint32_t hw_sif_get(int reg);
void     hw_sif_set(int reg, uint32_t v);

// The IOP's side of the SIF registers: `set` replaces the value, otherwise
// the bits are added (SMFLG).
void hw_sif_iop(int reg, uint32_t v, bool set);
// SIF0 (IOP to EE): when the EE arms the channel, `source` is asked for a
// packet (it writes it into EE memory and returns true); the channel then
// ends and raises its interrupt. hw_sif0_kick: a packet is waiting.
void hw_set_sif0_source(bool (*source)());
void hw_sif0_kick();

// Host-timed events the kernel asks for (alarms): `due` is in hw_now() seconds.
void hw_set_deadline(double due);

// What the units beside the DMAC (ipu.cpp) use, with the hardware's lock held.
uint32_t& hw_dma_reg(int channel, uint32_t offset);  // CHCR 0x00, MADR 0x10, QWC 0x20, TADR 0x30
uint8_t*  hw_dma_ptr(uint32_t addr, uint32_t qwc);   // EE memory or the scratchpad
void      hw_dma_end_locked(int channel);            // STR clears, the channel's CIS bit is set

// ---- the IPU (ipu.cpp), with the hardware's lock held --------------------------------------
void ipu_init(std::mutex& hw_lock);      // the IPU's thread starts; it shares the lock
uint64_t ipu_read_locked(uint32_t addr);             // IPU_CMD, CTRL, BP, TOP; the output FIFO
void ipu_write_locked(uint32_t addr, uint64_t v);    // IPU_CMD, CTRL
void ipu_fifo_write_locked(const uint8_t qw[16]);    // the input FIFO
void ipu_dma_locked(int channel);                    // channel 3 or 4 started
std::string ipu_status();                            // commands run

// ---- the GS (gs.cpp), with the hardware's lock held ----------------------------------------
void gs_init();
void gs_path3(const uint8_t* data, uint32_t qwc);     // GIF path 3: quadwords of GIF packets
uint64_t gs_take_events();                           // bit 0 SIGNAL, bit 1 FINISH since last asked
uint64_t gs_siglblid();
std::string gs_status();
// The PCRTC's picture (RGBA) from PMODE, SMODE2, DISPFB1, DISPLAY1, DISPFB2, DISPLAY2, BGCOLOR.
bool gs_snapshot(const uint64_t regs[7], std::vector<uint32_t>& out, int& w, int& h);
bool hw_gs_picture(std::vector<uint32_t>& out, int& w, int& h);   // takes the lock

// ---- the window (display.cpp), on the main thread ---------------------------------------
void display_init(bool window, const std::vector<double>& shots, const std::string& shot_dir);
bool display_pump(double now);                       // false: the window was closed

// ---- the disc (disc.cpp)----------------------------------------------------------------
struct DiscFile { uint32_t lsn, size; char name[16]; uint8_t date[7]; bool is_dir; };
bool disc_open(const char* path);                    // .cue, .iso, or a single raw .bin
bool disc_read(uint32_t lsn, uint32_t count, uint8_t* out);   // 2048-byte user sectors
bool disc_find(const std::string& path, DiscFile& f);         // \DIR\NAME.EXT;1
bool disc_read_file(const std::string& path, std::vector<uint8_t>& out);
std::string disc_boot_path();                        // SYSTEM.CNF's BOOT2

// ---- the IOP (iop.cpp) ------------------------------------------------------------------
void iop_init();
void iop_update();                       // the IOP's own progress (a reboot completing)
void iop_vblank();                       // once a field: what the IOP's modules do per frame
void iop_sif1(uint32_t src, uint32_t dst, uint32_t size, uint32_t attr);

// ---- what each part reports in the run's heartbeat --------------------------------------
std::string hw_status();                 // DMA traffic
std::string iop_status();                // disc reads
std::string kernel_status();             // syscalls, interrupts, threads
void profile_start();                    // sample where the guest spends its time (--profile)
std::string profile_report();

// ---- tracing --------------------------------------------------------------------------
extern int g_trace;                      // 0 off, 1 syscalls, 2 + hardware registers
extern double g_run_seconds;             // > 0: end the run after this long (host time)
#define TRACE(level, ...) do { if (g_trace >= (level)) { std::fprintf(stderr, __VA_ARGS__); } } while (0)

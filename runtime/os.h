// ps2kit runtime — what the kernel (kernel.cpp), the hardware (hw.cpp) and
// the boot (boot.cpp) share. Not seen by generated code.
#pragma once
#include "rt.h"

#include <cstdio>

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

// ---- the IOP (iop.cpp) ------------------------------------------------------------------
// `disc` is the directory holding the disc's files (cdrom0: is read there).
void iop_init(const char* disc);
void iop_update();                       // the IOP's own progress (a reboot completing)
void iop_sif1(uint32_t src, uint32_t dst, uint32_t size, uint32_t attr);

// ---- tracing --------------------------------------------------------------------------
extern int g_trace;                      // 0 off, 1 syscalls, 2 + hardware registers
#define TRACE(level, ...) do { if (g_trace >= (level)) { std::fprintf(stderr, __VA_ARGS__); } } while (0)

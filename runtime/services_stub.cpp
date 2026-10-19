// ps2kit runtime — the no-hardware services, for tests that only run the
// game's own code: anything a hardware-free run must never reach (system
// calls, traps, hardware registers, VU0 micro mode) stops with a message,
// and no interrupt ever arrives.
#include "rt.h"

std::atomic<uint32_t> g_ee_pending{0};
void ee_poll(EEContext&) {}

void ee_syscall(EEContext& c, uint32_t pc) { rt_die("syscall %d at %08X", (int)c.r[3].sw[0], pc); }
void ee_break(EEContext&, uint32_t pc, uint32_t code) { rt_die("break 0x%X at %08X (%s)", code, pc, rt_name_of(pc)); }
void ee_unimplemented(EEContext&, uint32_t pc, const char* what) { rt_die("%s at %08X (%s)", what, pc, rt_name_of(pc)); }
uint64_t ee_hw_read(uint32_t a, int size) { rt_die("hardware read%d %08X", size * 8, a); }
void ee_hw_write(uint32_t a, uint64_t, int size) { rt_die("hardware write%d %08X", size * 8, a); }
void ee_hw_write128(uint32_t a, const EEReg&) { rt_die("hardware write128 %08X", a); }
uint32_t ee_mfc0(EEContext& c, int reg) { return c.cop0[reg]; }
void ee_mtc0(EEContext& c, int reg, uint32_t v) { c.cop0[reg] = v; }
void ee_ei(EEContext& c) { c.cop0[12] |= 0x10000u; }
void ee_di(EEContext& c) { c.cop0[12] &= ~0x10000u; }
uint32_t ee_cfc2(EEContext& c, int reg) { return c.vi[reg]; }
void ee_ctc2(EEContext& c, int reg, uint32_t v) { c.vi[reg] = v; }
void ee_vu0_callms(EEContext&, uint32_t addr) { rt_die("VU0 micro program at %04X", addr); }

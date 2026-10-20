// ps2kit runtime — the IOP's processor, an R3000A (MIPS I), interpreted.
//
// One context per IOP thread; the IOP's scheduler (iop_os.cpp) runs them in
// slices. Both delay slots are the R3000's: a branch's next instruction
// runs before its target, and a load's value reaches its register only
// after the next instruction (which still reads the old value; if it
// writes that register itself, its own value wins).
#pragma once
#include <cstdint>

struct IopCpu {
    uint32_t r[32] = {};
    uint32_t hi = 0, lo = 0;
    uint32_t pc = 0, npc = 4;              // the instruction to run, the one after it
    uint32_t ld_reg = 0, ld_val = 0;       // a load in its delay slot (0: none)
};

// Where the interpreter stopped.
enum IopStop {
    IOP_BUDGET,                            // n instructions run
    IOP_SYSCALL,                           // a SYSCALL: its 20-bit code in `code`; pc is the next instruction
    IOP_MAGIC,                             // pc reached 0xBFC00000-0xBFFFFFFF (the runtime's return addresses)
};
IopStop iop_cpu_run(IopCpu& c, int n, uint32_t& code, int& ran);   // ran: instructions run

// The IOP's address space, as the interpreter sees it (iop_os.cpp): RAM
// (2 MB, mirrored to 8 MB), the scratchpad, the I/O registers, the SPU2.
uint32_t iop_read(uint32_t addr, int size);
void iop_write(uint32_t addr, uint32_t v, int size);
uint8_t* iop_ram();                        // the 2 MB (iop.cpp)

// The runtime's return addresses: a call made by the host into IOP code
// (an interrupt handler, a thread's entry, an RPC server function) returns
// to one of these, which the interpreter never runs.
constexpr uint32_t IOP_MAGIC_BASE = 0xBFC00000u;

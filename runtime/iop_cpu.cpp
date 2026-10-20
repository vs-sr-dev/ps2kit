// ps2kit runtime — the IOP's R3000A, interpreted (see iop_cpu.h).
#include "iop_cpu.h"

#include "rt.h"

#include <cstring>

namespace {

inline int32_t sx16(uint32_t v) { return (int16_t)(v & 0xFFFF); }

// RAM is read and written in place; everything else goes to iop_read/iop_write.
inline uint32_t phys(uint32_t a) { return a & 0x1FFFFFFFu; }
inline bool is_ram(uint32_t p) { return p < 0x00800000u; }

inline uint32_t rd32(uint32_t a) {
    uint32_t p = phys(a);
    if (is_ram(p)) { uint32_t v; std::memcpy(&v, iop_ram() + (p & 0x1FFFFC), 4); return v; }
    return iop_read(a & ~3u, 4);
}
inline uint32_t rd16(uint32_t a) {
    uint32_t p = phys(a);
    if (is_ram(p)) { uint16_t v; std::memcpy(&v, iop_ram() + (p & 0x1FFFFE), 2); return v; }
    return iop_read(a & ~1u, 2) & 0xFFFF;
}
inline uint32_t rd8(uint32_t a) {
    uint32_t p = phys(a);
    if (is_ram(p)) return iop_ram()[p & 0x1FFFFF];
    return iop_read(a, 1) & 0xFF;
}
inline void wr32(uint32_t a, uint32_t v) {
    uint32_t p = phys(a);
    if (is_ram(p)) { std::memcpy(iop_ram() + (p & 0x1FFFFC), &v, 4); return; }
    iop_write(a & ~3u, v, 4);
}
inline void wr16(uint32_t a, uint32_t v) {
    uint32_t p = phys(a);
    if (is_ram(p)) { uint16_t h = (uint16_t)v; std::memcpy(iop_ram() + (p & 0x1FFFFE), &h, 2); return; }
    iop_write(a & ~1u, v & 0xFFFF, 2);
}
inline void wr8(uint32_t a, uint32_t v) {
    uint32_t p = phys(a);
    if (is_ram(p)) { iop_ram()[p & 0x1FFFFF] = (uint8_t)v; return; }
    iop_write(a, v & 0xFF, 1);
}

}  // namespace

IopStop iop_cpu_run(IopCpu& c, int n, uint32_t& code, int& ran) {
    uint32_t* r = c.r;
    for (int k = 0; k < n; ++k) {
        ran = k;
        uint32_t pc = c.pc;
        if (pc >= IOP_MAGIC_BASE && pc < 0xC0000000u) {
            if (c.ld_reg) { r[c.ld_reg] = c.ld_val; c.ld_reg = 0; }    // the return's slot held a load
            return IOP_MAGIC;
        }
        if (pc & 3) rt_die("IOP: instruction fetch from %08X", pc);
        uint32_t op = rd32(pc);
        c.pc = c.npc;
        c.npc += 4;
        // The load in the slot before this instruction lands after it,
        // unless this instruction writes the same register (written below).
        uint32_t ld_reg = c.ld_reg, ld_val = c.ld_val;
        c.ld_reg = 0;
        uint32_t wreg = 0;                         // the register this instruction wrote
        uint32_t rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31;
        uint32_t imm = op & 0xFFFF;
        int32_t simm = sx16(op);
        auto set = [&](uint32_t reg, uint32_t v) { if (reg) { r[reg] = v; wreg = reg; } };
        auto load = [&](uint32_t reg, uint32_t v) { if (reg) { c.ld_reg = reg; c.ld_val = v; } };
        auto branch = [&](bool taken) { if (taken) c.npc = c.pc + ((uint32_t)simm << 2); };
        switch (op >> 26) {
        case 0x00:
            switch (op & 63) {
            case 0x00: set(rd, r[rt] << ((op >> 6) & 31)); break;                       // SLL
            case 0x02: set(rd, r[rt] >> ((op >> 6) & 31)); break;                       // SRL
            case 0x03: set(rd, (uint32_t)((int32_t)r[rt] >> ((op >> 6) & 31))); break;  // SRA
            case 0x04: set(rd, r[rt] << (r[rs] & 31)); break;                           // SLLV
            case 0x06: set(rd, r[rt] >> (r[rs] & 31)); break;                           // SRLV
            case 0x07: set(rd, (uint32_t)((int32_t)r[rt] >> (r[rs] & 31))); break;      // SRAV
            case 0x08: c.npc = r[rs]; break;                                            // JR
            case 0x09: { uint32_t t = r[rs]; set(rd, c.pc + 4); c.npc = t; break; }    // JALR
            case 0x0C:                                                                  // SYSCALL
                code = (op >> 6) & 0xFFFFF;
                if (ld_reg && ld_reg != wreg) r[ld_reg] = ld_val;
                ran = k + 1;
                return IOP_SYSCALL;
            case 0x0D: rt_die("IOP: BREAK %X at %08X", (op >> 6) & 0xFFFFF, pc);
            case 0x10: set(rd, c.hi); break;                                            // MFHI
            case 0x11: c.hi = r[rs]; break;                                             // MTHI
            case 0x12: set(rd, c.lo); break;                                            // MFLO
            case 0x13: c.lo = r[rs]; break;                                             // MTLO
            case 0x18: {                                                                // MULT
                int64_t p = (int64_t)(int32_t)r[rs] * (int32_t)r[rt];
                c.lo = (uint32_t)p; c.hi = (uint32_t)(p >> 32);
                break;
            }
            case 0x19: {                                                                // MULTU
                uint64_t p = (uint64_t)r[rs] * r[rt];
                c.lo = (uint32_t)p; c.hi = (uint32_t)(p >> 32);
                break;
            }
            case 0x1A: {                                                                // DIV
                int32_t a = (int32_t)r[rs], b = (int32_t)r[rt];
                if (b == 0) { c.lo = a < 0 ? 1 : 0xFFFFFFFFu; c.hi = (uint32_t)a; }
                else if (a == INT32_MIN && b == -1) { c.lo = (uint32_t)a; c.hi = 0; }
                else { c.lo = (uint32_t)(a / b); c.hi = (uint32_t)(a % b); }
                break;
            }
            case 0x1B: {                                                                // DIVU
                uint32_t a = r[rs], b = r[rt];
                if (b == 0) { c.lo = 0xFFFFFFFFu; c.hi = a; }
                else { c.lo = a / b; c.hi = a % b; }
                break;
            }
            case 0x20: {                                                                // ADD
                int64_t s = (int64_t)(int32_t)r[rs] + (int32_t)r[rt];
                if (s != (int32_t)s) rt_die("IOP: ADD overflow at %08X", pc);
                set(rd, (uint32_t)s);
                break;
            }
            case 0x21: set(rd, r[rs] + r[rt]); break;                                   // ADDU
            case 0x22: {                                                                // SUB
                int64_t s = (int64_t)(int32_t)r[rs] - (int32_t)r[rt];
                if (s != (int32_t)s) rt_die("IOP: SUB overflow at %08X", pc);
                set(rd, (uint32_t)s);
                break;
            }
            case 0x23: set(rd, r[rs] - r[rt]); break;                                   // SUBU
            case 0x24: set(rd, r[rs] & r[rt]); break;                                   // AND
            case 0x25: set(rd, r[rs] | r[rt]); break;                                   // OR
            case 0x26: set(rd, r[rs] ^ r[rt]); break;                                   // XOR
            case 0x27: set(rd, ~(r[rs] | r[rt])); break;                                // NOR
            case 0x2A: set(rd, (int32_t)r[rs] < (int32_t)r[rt]); break;                 // SLT
            case 0x2B: set(rd, r[rs] < r[rt]); break;                                   // SLTU
            default: rt_die("IOP: special function %02X at %08X (%08X)", op & 63, pc, op);
            }
            break;
        case 0x01: {                                                                    // REGIMM
            bool lt = (int32_t)r[rs] < 0;
            uint32_t kind = rt & 0x1F;
            if (kind == 0x10 || kind == 0x11) set(31, c.pc + 4);                        // BLTZAL, BGEZAL
            if (kind == 0x00 || kind == 0x10) branch(lt);                               // BLTZ
            else if (kind == 0x01 || kind == 0x11) branch(!lt);                         // BGEZ
            else rt_die("IOP: REGIMM %02X at %08X", kind, pc);
            break;
        }
        case 0x02: c.npc = (c.pc & 0xF0000000u) | ((op & 0x03FFFFFF) << 2); break;      // J
        case 0x03: set(31, c.pc + 4); c.npc = (c.pc & 0xF0000000u) | ((op & 0x03FFFFFF) << 2); break;   // JAL
        case 0x04: branch(r[rs] == r[rt]); break;                                       // BEQ
        case 0x05: branch(r[rs] != r[rt]); break;                                       // BNE
        case 0x06: branch((int32_t)r[rs] <= 0); break;                                  // BLEZ
        case 0x07: branch((int32_t)r[rs] > 0); break;                                   // BGTZ
        case 0x08: {                                                                    // ADDI
            int64_t s = (int64_t)(int32_t)r[rs] + simm;
            if (s != (int32_t)s) rt_die("IOP: ADDI overflow at %08X", pc);
            set(rt, (uint32_t)s);
            break;
        }
        case 0x09: set(rt, r[rs] + (uint32_t)simm); break;                              // ADDIU
        case 0x0A: set(rt, (int32_t)r[rs] < simm); break;                               // SLTI
        case 0x0B: set(rt, r[rs] < (uint32_t)simm); break;                              // SLTIU
        case 0x0C: set(rt, r[rs] & imm); break;                                         // ANDI
        case 0x0D: set(rt, r[rs] | imm); break;                                         // ORI
        case 0x0E: set(rt, r[rs] ^ imm); break;                                         // XORI
        case 0x0F: set(rt, imm << 16); break;                                           // LUI
        case 0x10: rt_die("IOP: COP0 %08X at %08X (the kernel's own work)", op, pc);
        case 0x12: rt_die("IOP: COP2 (GTE) %08X at %08X", op, pc);
        case 0x20: load(rt, (uint32_t)(int32_t)(int8_t)rd8(r[rs] + simm)); break;      // LB
        case 0x21: {                                                                    // LH
            uint32_t a = r[rs] + simm;
            if (a & 1) rt_die("IOP: LH from %08X at %08X", a, pc);
            load(rt, (uint32_t)(int32_t)(int16_t)rd16(a));
            break;
        }
        case 0x22: case 0x26: {                                                         // LWL, LWR
            uint32_t a = r[rs] + simm, w = rd32(a & ~3u), sh = (a & 3) * 8;
            uint32_t cur = ld_reg == rt ? ld_val : r[rt];                               // a load in flight is merged
            uint32_t v = (op >> 26) == 0x22 ? (cur & (0x00FFFFFFu >> sh)) | (w << (24 - sh))
                                            : (cur & ~(0xFFFFFFFFu >> sh)) | (w >> sh);
            if (ld_reg == rt) ld_reg = 0;
            load(rt, v);
            break;
        }
        case 0x23: {                                                                    // LW
            uint32_t a = r[rs] + simm;
            if (a & 3) rt_die("IOP: LW from %08X at %08X", a, pc);
            load(rt, rd32(a));
            break;
        }
        case 0x24: load(rt, rd8(r[rs] + simm)); break;                                  // LBU
        case 0x25: {                                                                    // LHU
            uint32_t a = r[rs] + simm;
            if (a & 1) rt_die("IOP: LHU from %08X at %08X", a, pc);
            load(rt, rd16(a));
            break;
        }
        case 0x28: wr8(r[rs] + simm, r[rt]); break;                                     // SB
        case 0x29: {                                                                    // SH
            uint32_t a = r[rs] + simm;
            if (a & 1) rt_die("IOP: SH to %08X at %08X", a, pc);
            wr16(a, r[rt]);
            break;
        }
        case 0x2A: case 0x2E: {                                                         // SWL, SWR
            uint32_t a = r[rs] + simm, w = rd32(a & ~3u), sh = (a & 3) * 8, v = r[rt];
            w = (op >> 26) == 0x2A ? (w & ~(0xFFFFFFFFu >> (24 - sh))) | (v >> (24 - sh))
                                   : (w & ~(0xFFFFFFFFu << sh)) | (v << sh);
            wr32(a & ~3u, w);
            break;
        }
        case 0x2B: {                                                                    // SW
            uint32_t a = r[rs] + simm;
            if (a & 3) rt_die("IOP: SW to %08X at %08X", a, pc);
            wr32(a, r[rt]);
            break;
        }
        default: rt_die("IOP: opcode %02X at %08X (%08X)", op >> 26, pc, op);
        }
        if (ld_reg && ld_reg != wreg) r[ld_reg] = ld_val;
    }
    ran = n;
    return IOP_BUDGET;
}

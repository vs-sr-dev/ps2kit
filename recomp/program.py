"""The program model the recompiler needs: units, entry points, labels, switch tables.

Units are the code ranges that become C++: every function of the
executable's function map (ps2kit.elf.functions(), or a map given as TSV),
plus every non-zero gap between them.

Entry points are the addresses a C++ function must exist for: every unit
start, and every target of a jump or branch that leaves its unit (a tail
call). A target outside every unit is reported as a bad target. A target
inside another unit but not at its start gets an entry of its own, covering
the code from there to the unit's end, and the backward branches out of
that body add more, to a fixed point (the same rule as wiikit's).

Switch tables: GCC's form on the EE is

    sltiu  vT, vI, N          bound (before the branch to the default)
    lui    vB, %hi(table)
    sll    vS, vI, 2
    addiu  vB, vB, %lo(table)
    addu   vS, vS, vB
    lw     vX, 0(vS)
    jr     vX

A linear tracker over the unit follows constants (lui/addiu/ori), a base
plus an unknown index (addu), and a word loaded from such a base; a `jr` on
a register holding such a word is a table jump. Its size is the `sltiu`
bound when one precedes it closely, otherwise the run of words pointing into
the unit. Every other `jr` that is not `jr ra` is an indirect tail call.
"""
import bisect
import struct

from .. import ee

CONST, BASE, TABLE = "const", "base", "table"


class Unit:
    __slots__ = ("start", "end", "name", "source", "ins")

    def __init__(self, start, end, name, source=""):
        self.start, self.end, self.name, self.source = start, end, name, source
        self.ins = []                     # [Insn]

    def __contains__(self, a):
        return self.start <= a < self.end


class Program:
    def __init__(self, elf, functions, log=print):
        """`functions`: {va: (name, size, source)} as ps2kit.elf.functions() gives."""
        self.elf, self.log = elf, log
        self.units = self._units(functions)
        self._starts = [u.start for u in self.units]
        for u in self.units:
            u.ins = [ee.decode(w, a) for a, w in elf.words(u.start, u.end)]
        self.undecoded = [(i.va, i.word) for u in self.units for i in u.ins if not i.valid]
        self.tables = {}                  # jr address -> [targets]
        self.unresolved_tables = []
        self._resolve_tables()
        self.entries = self._entries()

    # --- units --------------------------------------------------------------------

    def _units(self, functions):
        lo, _off, size = self.elf.sections[".text"]
        hi = lo + size
        units = [Unit(va, va + sz, name, src) for va, (name, sz, src) in sorted(functions.items())
                 if lo <= va < hi and sz > 0]
        gaps, a = [], lo
        for u in units:
            if u.start > a:
                gaps.append((a, u.start))
            a = max(a, u.end)
        if a < hi:
            gaps.append((a, hi))
        for s, e in gaps:
            data = self.elf.read(s, e - s)
            nz = [i for i in range(0, len(data), 4) if data[i:i + 4] != b"\0\0\0\0"]
            if nz:
                s2, e2 = s + nz[0], s + nz[-1] + 4
                units.append(Unit(s2, e2, "unk_%08X" % s2))
        units.sort(key=lambda u: u.start)
        return units

    def unit_at(self, a):
        i = bisect.bisect_right(self._starts, a) - 1
        return self.units[i] if i >= 0 and a in self.units[i] else None

    # --- switch tables ---------------------------------------------------------------

    def _resolve_tables(self):
        for u in self.units:
            regs, bound = {}, None
            for i in u.ins:
                op, rs, rt, rd = i.op, i.rs, i.rt, i.rd
                if op == "jr" and rs != 31:
                    v = regs.get(rs)
                    if v and v[0] == TABLE:
                        near = bound[1] if bound and i.va - bound[0] <= 48 else None
                        targets = self._read_table(v[1], u, near)
                        if targets:
                            self.tables[i.va] = targets
                        else:
                            self.unresolved_tables.append(i.va)
                    regs.clear()
                    continue
                if op == "sltiu":
                    bound = (i.va, i.imm & 0xFFFFFFFF)
                if op == "lui":
                    regs[rt] = (CONST, (i.imm << 16) & 0xFFFFFFFF)
                elif op in ("addiu", "daddiu") and regs.get(rs, (None,))[0] == CONST:
                    regs[rt] = (CONST, (regs[rs][1] + i.imm) & 0xFFFFFFFF)
                elif op == "ori" and regs.get(rs, (None,))[0] == CONST:
                    regs[rt] = (CONST, regs[rs][1] | i.imm)
                elif op in ("addu", "daddu"):
                    a, b = regs.get(rs), regs.get(rt)
                    if a and a[0] == CONST and not b:
                        regs[rd] = (BASE, a[1])
                    elif b and b[0] == CONST and not a:
                        regs[rd] = (BASE, b[1])
                    else:
                        regs.pop(rd, None)
                elif op == "lw" and regs.get(rs, (None,))[0] == BASE:
                    regs[rt] = (TABLE, (regs[rs][1] + i.imm) & 0xFFFFFFFF)
                else:
                    for r in _writes(i):
                        regs.pop(r, None)

    def _read_table(self, base, u, n):
        out = []
        limit = n if n is not None else 4096
        for k in range(limit):
            try:
                w = self.elf.u32(base + 4 * k)
            except ValueError:
                break
            if w not in u or w % 4:
                break
            out.append(w)
        if n is not None and len(out) != n:
            return None
        return out or None

    # --- entry points -----------------------------------------------------------------

    def _entries(self):
        entries = {u.start for u in self.units}
        self.bad_targets = []
        extra_done = set()
        while True:
            new = set()
            for u in self.units:
                for i in u.ins:
                    t = i.target
                    if t is None or (t in u and i.op != "jal"):
                        continue
                    if self.unit_at(t) is None:
                        self.bad_targets.append((i.va, t))
                    elif t not in entries:
                        new.add(t)
            for e in sorted(entries | new):
                u = self.unit_at(e)
                if e in extra_done or u is None or e == u.start:
                    continue
                extra_done.add(e)
                for i in u.ins:
                    if i.va >= e and i.target is not None and i.target in u and i.target < e:
                        new.add(i.target)
            if not new - entries:
                break
            entries |= new
        return sorted(entries)

    def body(self, entry):
        """(unit, [Insn]) of the C++ function for this entry."""
        u = self.unit_at(entry)
        return u, [i for i in u.ins if i.va >= entry]

    def labels(self, entry):
        """Addresses inside this entry's body that are jumped to."""
        u, ins = self.body(entry)
        out = set()
        for i in ins:
            if i.target is not None and i.op not in ("jal", "bltzal", "bgezal", "bltzall", "bgezall") \
                    and entry <= i.target < u.end:
                out.add(i.target)
            if i.va in self.tables:
                out.update(x for x in self.tables[i.va] if x >= entry)
        return out


def _writes(i):
    """GPRs an instruction may write (for the table tracker: anything else forgets)."""
    f = i.form
    if f in ("rt_rs_simm", "rt_rs_uimm", "rt_uimm", "rt_mem", "rt_c0", "rt_fs", "rt_c2"):
        return (i.rt,)
    if f.startswith("rd") or i.op in ("jalr",):
        return (i.rd,)
    if i.op in ("jal", "bltzal", "bgezal", "bltzall", "bgezall"):
        return (31,)
    return ()

// Decompile functions by name (or by hex address) into one C file.
//   analyzeHeadless <projdir> <name> -process <file> -noanalysis -scriptPath <this dir>
//       -postScript Decomp.java NAME_OR_ADDR [...] OUT.c
// A function that does not exist yet at an address is created there first.
// @category ps2kit
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Symbol;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Decomp extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: Decomp.java NAME_OR_ADDR [...] OUT.c");
            return;
        }
        DecompInterface d = new DecompInterface();
        d.openProgram(currentProgram);
        try (PrintWriter out = new PrintWriter(new FileWriter(args[args.length - 1]))) {
            for (int k = 0; k < args.length - 1; ++k) {
                Function f = lookup(args[k]);
                if (f == null) {
                    out.println("// " + args[k] + ": not found\n");
                    continue;
                }
                DecompileResults r = d.decompileFunction(f, 120, monitor);
                out.println("// " + f.getName() + " @ " + f.getEntryPoint());
                out.println(r.decompileCompleted() ? r.getDecompiledFunction().getC() : "// failed: " + r.getErrorMessage());
            }
        }
    }

    private Function lookup(String s) throws Exception {
        if (s.matches("(0x)?[0-9A-Fa-f]{6,8}")) {
            Address a = toAddr(Long.parseLong(s.replace("0x", ""), 16));
            Function f = getFunctionAt(a);
            if (f == null) {
                disassemble(a);
                f = createFunction(a, null);
            }
            return f;
        }
        for (Symbol sym : currentProgram.getSymbolTable().getSymbols(s)) {
            Function f = getFunctionAt(sym.getAddress());
            if (f != null) return f;
            disassemble(sym.getAddress());
            f = createFunction(sym.getAddress(), s);
            if (f != null) return f;
        }
        return null;
    }
}

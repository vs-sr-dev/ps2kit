// Disassemble given ranges and write every instruction as TSV: address, mnemonic, operands.
//
// An independent decoder to check ps2kit.ee against (Ghidra with
// ghidra-emotionengine-reloaded, processor r5900:LE:32:default). Ranges come
// from a TSV whose first two columns are a hex start and a decimal size, as
// `python -m ps2kit.elf GAME.ELF --functions map.tsv` writes; the program can
// be imported without analysis, since this script disassembles them itself.
//
//   analyzeHeadless <projdir> <proj> -import GAME.ELF -processor r5900:LE:32:default \
//       -noanalysis -scriptPath ps2kit/ghidra -postScript ExportListing.java map.tsv out.tsv
//@category ps2kit
import ghidra.app.cmd.disassemble.DisassembleCommand;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

public class ExportListing extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        AddressSet set = new AddressSet();
        try (BufferedReader r = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = r.readLine()) != null) {
                String[] f = line.split("\t");
                if (f.length < 2) continue;
                long va = Long.parseLong(f[0], 16);
                long size = Long.parseLong(f[1]);
                if (size <= 0) continue;
                Address a = toAddr(va);
                set.addRange(a, a.add(size - 1));
            }
        }
        // restricted to the ranges: flow never leaves them
        DisassembleCommand cmd = new DisassembleCommand(set, set, true);
        cmd.applyTo(currentProgram, monitor);
        int n = 0;
        try (PrintWriter w = new PrintWriter(args[1], "UTF-8")) {
            InstructionIterator it = currentProgram.getListing().getInstructions(set, true);
            while (it.hasNext()) {
                Instruction ins = it.next();
                StringBuilder ops = new StringBuilder();
                for (int i = 0; i < ins.getNumOperands(); i++) {
                    if (i > 0) ops.append(',');
                    ops.append(ins.getDefaultOperandRepresentation(i));
                }
                w.printf("%08x\t%s\t%s%n", ins.getAddress().getOffset(), ins.getMnemonicString(), ops);
                n++;
            }
        }
        println("ExportListing: " + n + " instructions in " + set.getNumAddresses() + " bytes");
    }
}

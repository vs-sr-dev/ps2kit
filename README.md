# ps2kit

A game-agnostic toolkit for PlayStation 2 reverse engineering and native PC
ports: disc images, executables and their debug information, the Emotion
Engine's instruction set, GS memory, VIF and VU microcode, IOP modules,
movies and SPU audio. It is growing toward its own static recompiler from
EE code to C++ and a runtime that replaces the console's hardware under the
recompiled game, the same shape as
[wiikit](https://github.com/vs-sr-dev/wiikit) for the Wii.

Each PS2 game has its own engine and formats, but a large part of every port
is the *same* work: the same disc layout, the same SDK, the same CPU, GS and
VUs. ps2kit collects that shared part. It grows inside the ports: each piece
is written because a game needed it, then kept free of that game's
knowledge. Game formats and game fixes live in the ports.

## Ports built on it

| Port | Game | What it asked of ps2kit |
|---|---|---|
| [pc-extermination](https://github.com/vs-sr-dev/pc-extermination) | Extermination (2001, PAL) | the first layers: fingerprint, SPU ADPCM, PSS, CodeWarrior overlays, IOP modules, the GS memory model, VIF and VU microcode. Its recompiled code runs on PS2Recomp's runtime with our patches, not yet on ps2kit's |
| pc-evergrace (probe) | Evergrace (2000, PAL) | a second disc through the fingerprint; seven fixes to the shared runtime patch |
| [pc-dynastywarriors2](https://github.com/vs-sr-dev/pc-dynastywarriors2) | Dynasty Warriors 2 (2000, PAL) | CD images (MODE2/2352 and CD-DA), `SYSTEM.CNF`; the symbol table and the ECOFF `.mdebug` of an unstripped executable; the EE decoder, checked instruction by instruction against Ghidra; ps2kit's own translator to C++ (all 4 073 functions), its runtime interface, a reference interpreter and differential tests; the runtime's boot: the EE kernel, the hardware registers, the IOP at the SIF boundary; the IPU (checked against ffmpeg on the game's movies), the GS with a software rasteriser, a window: its first pictures are its logo movies |

ps2kit was split out of pc-extermination with its history (`git subtree
split`) when Dynasty Warriors 2 began; the earlier commit messages describe
Extermination's sessions, the changes in them are ps2kit's.

## Using it

A port takes ps2kit as a git submodule at `ps2kit/`, so that
`python -m ps2kit.…` works from the port's root. Each port pins a ps2kit
commit and moves it forward deliberately.

```sh
python -m ps2kit.disc GAME.cue --info                 # CD (.cue/.bin) or DVD (.iso)
python -m ps2kit.disc GAME.cue --list --forms
python -m ps2kit.disc GAME.cue --extract build/disc
python -m ps2kit.fingerprint build/disc --deep        # what is standard on this disc
python -m ps2kit.elf SLES_123.45 --info
python -m ps2kit.elf SLES_123.45 --functions build/map/symbols.tsv   # unstripped executables
python -m ps2kit.mdebug SLES_123.45 --files
python -m ps2kit.ee SLES_123.45 --at 0x200008
python -m ps2kit.ee SLES_123.45 --census --functions build/map/symbols.tsv
python -m ps2kit.elf SLES_123.45 --xref 0x270490
python -m ps2kit.vu SLES_123.45 --dma 0x2322F0
python -m ps2kit.irx MODULE.IRX
python -m ps2kit.pss MOVIE.PSS --video m.m2v --audio m.wav
python -m ps2kit.recomp SLES_123.45 --out build/recomp            # EE code -> C++
python -m ps2kit.eeemu SLES_123.45 --call 0x2006A0 --a0 1         # the reference interpreter
python -m ps2kit.recomp.difftest SLES_123.45 --driver build/recomp-build/difftest.exe [--leaves]
build/recomp-build/ps2run GAME.cue [--trace 1] [--seconds 30] [--profile] [--headless] [--shot 10,20]   # the game
```

Building recompiled code needs CMake, Ninja and a C++20 compiler (clang
from MSYS2 so far); the generated project includes `runtime/runtime.cmake`.
`ps2run` boots the recompiled game from the user's disc image as the boot
ROM would (`SYSTEM.CNF`, the ELF read from the disc).

## Layers

| Layer | Question it answers | Now | Next |
|---|---|---|---|
| 1. Recognise | What is on this disc? | `disc` (CUE/BIN with MODE2/2352 and CD-DA tracks, plain ISO, ISO 9660, `SYSTEM.CNF`, XA Form 1/Form 2 per file); `fingerprint` (magics, SPU ADPCM with stereo interleave detection, GS packets, compiler, SDK library versions, IRX names and versions, overlays, file paths named by the code) | a knowledge base: each finding linked to the module that handles it |
| 2. Extract | Turn standard formats into standard files | `adpcm`, `pss`, `mwo3`, `irx`, `gs`, `vif`, `vu`, `gsmem` (see below) | PSMCT16S and Z formats, CSM2, TIM2, VAG/VAB, SShd banks, IOPRP romdir |
| 3. Map code | What does the code do, where? | `elf` (segments, sections, `.symtab`, the function map of an unstripped executable and its coverage of `.text`, lui/addiu xrefs, `jal` callers); `mdebug` (ECOFF debug: source files, procedures with frames, local symbols); `ee` (R5900 decoder: MIPS III, MMI, COP0, the FPU, COP2 transfers and VU0 macro mode; census); Ghidra with ghidra-emotionengine-reloaded as a second reader (`ghidra/ExportListing.java`) | function discovery for stripped executables; SDK functions named by signature |
| 4. Translate | Turn EE code into C++ | `recomp`: units and entries from the function map, switch tables sized from their bound checks, one C++ function per entry, delay slots (likely branches on the taken path only), direct calls and tail calls, indirect calls through a sorted table, interrupt polling on back-edges, every operation of the R5900 a game uses including MMI, the FPU and VU0 macro mode with its flags; `eeemu`, a reference interpreter; `recomp.difftest`, the two compared register for register and byte for byte (Extermination still uses PS2Recomp) | function discovery for stripped executables; overlays |
| 5. Runtime | Replace the hardware | `ee.h` (the guest as generated code sees it: registers, memory, the FPU/VU model), memory and dispatch (`core.cpp`), a no-hardware stub, the link check and the differential-test driver; `boot.cpp` (the boot ROM's part); `kernel.cpp` (the EE kernel at the syscall boundary: threads on host threads, semaphores, interrupt handlers, alarms, SIF, Deci2's TTY); `hw.cpp` (INTC, DMAC, timers, GS CSR and the vertical blank, SIO); `iop.cpp` (the IOP at the SIF boundary: sifcmd, sifrpc, reboot, module loading, RPC servers for cdvdfsv, loadfile, the IOP heap, padman, mcserv's init); `disc.cpp` (CUE/BIN/ISO, ISO 9660); `ipu.cpp` (the MPEG decoder: VDEC, FDEC, BDEC, CSC, its FIFOs and DMA channels); `gs.cpp` (local memory, GIF path 3, transfers, a software rasteriser, the PCRTC); `display.cpp` (SDL3 window, screenshots); a heartbeat and a sampling profiler | VIF1, VU1 and VU0 micro mode, GIF paths 1 and 2, the GS's Z layouts and CSM2, SPU2 and the games' sound drivers, memory cards with data, file I/O on the IOP |

## Principles

* Pure Python, no dependencies, for layers 1-4; the runtime (layer 5) is
  C++20 with SDL3 for the window; the GS is drawn in software so far.
* Every claim is checked on real data before it goes in, against an
  independent reader where one exists.
* Game knowledge stays out: a function that needs a game's table address
  takes it as a parameter.
* Other projects (PS2Recomp, PCSX2, the open-source PS2SDK) are read as
  references for method and behaviour, never copied.

## Checks behind each module

| Module | Checked by |
|---|---|
| `disc` | Dynasty Warriors 2 (Redump CUE, MODE2/2352 + CD-DA): ISO 9660 tree of 50 files, `SYSTEM.CNF`; every one of the 276 830 data sectors' subheaders read (two equal copies each; 276 629 Form 1 data, 51 end-of-file markers, 150 empty); the CD-DA track a record points at found as track 2 |
| `elf` (functions) | Dynasty Warriors 2: 3 737 `.symtab` functions and 336 more from `.mdebug` (static copies of C++ inlines) tile `.text` exactly: no overlap, every gap all zero (alignment) |
| `mdebug` | Dynasty Warriors 2: 508 files, 4 056 procedures. 3 720 at the `.symtab` address of the same name; the other 336 are further static copies of C++ inlines whose first copy `.symtab` keeps (same name, another address); 17 `.symtab` functions have no procedure |
| `ee` | Dynasty Warriors 2: all 307 646 instructions of its 4 073 functions decoded (202 operations, none unknown) and equal to Ghidra's EE decoder, up to aliases and notation; the VU0 macro instructions' dest fields equal to Ghidra's suffixes; all 40 425 branch and jump targets equal |
| `fingerprint` | Extermination, Evergrace, Dynasty Warriors 2; symbols: an empty `.symtab`, none, 3 737 functions with `.mdebug` |
| `recomp` | Dynasty Warriors 2: 4 073 functions, 307 646 instructions, none unsupported; 142 switch tables, each of the size its `sltiu` bound gives; the C++ compiles and links (clang, `-O1`, no warning) |
| `recomp` + `runtime` against `eeemu` | Dynasty Warriors 2: 56 named functions (newlib's R5900 string routines, libgcc's 64-bit arithmetic, the SDK's soft float, `sprintf`, `rand`, libvu0) × 10 machine states, 560 runs, every register and every byte of memory equal; all 1 080 leaf functions fuzzed, 1 751 runs equal, 409 skipped (the reference could not finish them); 179 of the 202 operations the game uses exercised (7 of the rest are system operations) |
| `runtime` (boot, kernel, IOP) | Dynasty Warriors 2 from its Redump image: from `crt0` through the IOP reboot, ten modules, file searches and sector reads, pads and sound, into its 50 Hz vblank loop, up to its first memory card request; two runs give the same 946 IOP commands and RPC calls and the same sequence of system calls and results. Every SIF packet layout read from the game's own sifcmd/sifrpc/iopreset code; each module's RPC ids from the constant it passes to `sceSifRegisterRpc`; file positions equal to `disc`'s; the interrupt dispatch checked against the BIOS kernel's own code (ROM 1.60) |
| `eeemu` (floating point) | round toward zero computed exactly (rationals truncated to 24 bits): 1/3, the square root of 2, overflow saturation, denormal flush and the sign rules of exact zeros checked by hand |
| `adpcm` | Extermination's music by ear (stereo, interleave 0x400) |
| `gsmem` | Extermination: PSMT8 read of the title page equal to `gs.unswizzle8`; PSMT4 room pages; 23 278 TEX0 values render with correct colours |
| `vu` | Extermination's 22 microprograms (about 7 000 instructions): no unknown opcode |

## The GS memory model

`gsmem` models the 4 MB of GS local memory with plain arithmetic: the page
and block geometry of each format, the two block orders, and one column
pattern that every format packs its pixels into. It writes a host-to-local
transfer as the GS would and reads any rectangle back in any format, so the
usual trick of uploading 4- and 8-bit textures through 32-bit transfers
needs no special case.

## Known gaps

* The FPU and VU model is PCSX2's default (round toward zero, denormals
  flushed, infinities and NaNs clamped): the console's multiplier can differ
  in the last bit, and it treats exponent 255 as an ordinary exponent. The
  differential tests check the translation against a second reading of the
  same model, not the model against the console.
* The R5900 leaves the upper 64 bits of a GPR after an ordinary 64-bit
  write as they were (PCSX2's interpreter does the same); not checked on
  hardware.
* The runtime takes interrupts at loop back-edges and after system calls,
  not within cycles; the GS FIELD bit therefore changes a few lines after
  the kernel takes the vblank interrupt, so that handlers that wait for it
  (Dynasty Warriors 2's) see it change as on the console. DMA and disc reads
  complete at once.
* Koei's sound drivers (KESSOUND, KOEIADS) have a silent stand-in that
  answers zeros; the IOP's other modules run only as far as their RPC
  servers are written.

* `ee.against` compares with Ghidra's listing, which does not show the
  interlock bit of `qmfc2`/`qmtc2`/`cfc2`/`ctc2`: that bit is read from the
  encoding alone, unchecked by a second reader.
* capstone cannot disassemble the EE (in MIPS64 mode it reads `lq`/`sq`
  and MMI as MSA/DSP instructions); `ee` replaces it.
* An overlay alone gives a disassembler nothing to start from:
  `mwo3.seeds()` collects candidate entries; trust the seeds over Ghidra's
  own analysis.
* Ghidra creates no references to some globals reached with `lui` pairs;
  `elf.xref` finds them.

## Licence

MIT — see [LICENSE](LICENSE). ps2kit contains no game data and no Sony
code; it reads and replaces, it does not include.

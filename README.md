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
| pc-dynastywarriors2 | Dynasty Warriors 2 (2000, PAL) | CD images (MODE2/2352 and CD-DA), `SYSTEM.CNF`; the symbol table and the ECOFF `.mdebug` of an unstripped executable; the EE decoder, checked instruction by instruction against Ghidra; the first game for ps2kit's own recompiler |

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
```

## Layers

| Layer | Question it answers | Now | Next |
|---|---|---|---|
| 1. Recognise | What is on this disc? | `disc` (CUE/BIN with MODE2/2352 and CD-DA tracks, plain ISO, ISO 9660, `SYSTEM.CNF`, XA Form 1/Form 2 per file); `fingerprint` (magics, SPU ADPCM with stereo interleave detection, GS packets, compiler, SDK library versions, IRX names and versions, overlays, file paths named by the code) | a knowledge base: each finding linked to the module that handles it |
| 2. Extract | Turn standard formats into standard files | `adpcm`, `pss`, `mwo3`, `irx`, `gs`, `vif`, `vu`, `gsmem` (see below) | PSMCT16S and Z formats, CSM2, TIM2, VAG/VAB, SShd banks, IOPRP romdir |
| 3. Map code | What does the code do, where? | `elf` (segments, sections, `.symtab`, the function map of an unstripped executable and its coverage of `.text`, lui/addiu xrefs, `jal` callers); `mdebug` (ECOFF debug: source files, procedures with frames, local symbols); `ee` (R5900 decoder: MIPS III, MMI, COP0, the FPU, COP2 transfers and VU0 macro mode; census); Ghidra with ghidra-emotionengine-reloaded as a second reader (`ghidra/ExportListing.java`) | function discovery for stripped executables; SDK functions named by signature |
| 4. Translate | Turn EE code into C++ | — (Extermination uses PS2Recomp) | `recomp`: one C++ function per EE function, delay slots and likely branches, 128-bit registers, MMI, the FPU's non-IEEE rules, VU0 macro mode |
| 5. Runtime | Replace the hardware | — | the kernel (threads, semaphores, interrupts, SIF), SDK-level replacements where the game links the SDK, GS renderer, VU1 and VU0 micro mode, IPU/movies, SPU2, CD/DVD, pads, memory cards |

## Principles

* Pure Python, no dependencies, for layers 1-4; the runtime (layer 5) will
  be C++20 with SDL3 and OpenGL 4.5, as in the sibling kits.
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
| `fingerprint` | Extermination, Evergrace, Dynasty Warriors 2 |
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

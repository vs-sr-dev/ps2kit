"""ps2kit — game-agnostic building blocks for PS2 reverse engineering and ports.

Each module handles one thing the PS2 platform or its SDK imposes on every
game, independent of any particular title:

    disc        CD (.cue/.bin, MODE2 and CD-DA) and DVD (.iso) images, SYSTEM.CNF
    elf         load an EE executable, map addresses, symbols, function map, xrefs
    mdebug      ECOFF debug information: source files, procedures, local symbols
    ee          the R5900 instruction set: decode, disassemble, census
    eeemu       a reference EE interpreter, the recompiler's oracle
    recomp      EE code to C++ (recomp.difftest: checked against eeemu)
    adpcm       SPU (PS-ADPCM) decoding
    pss         Sony PSS movies: MPEG-2 video plus SShd/SSbd audio
    mwo3        CodeWarrior "MWo3" overlay modules: function seeds, ELF wrapper
    irx         IOP modules: relocation, import and export tables
    gs          GS upload packets (DMA/VIF/GIF) and their image transfers
    gsmem       GS local memory: any transfer in, any format and TEX0 out
    vif         VIF code streams: walk them, expand UNPACKs
    vu          VU0/VU1 microcode: disassembly, MPG uploads in VIF/DMA
    fingerprint scan a disc tree and report what it recognises

Everything here is pure Python 3.8+ with no dependencies; runtime/ is the
C++ runtime the recompiled code runs on. Game-specific
knowledge belongs in the game's own tools/, not here.
"""

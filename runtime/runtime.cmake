# The ps2kit runtime, included by the generated CMakeLists.txt.
#   ps2kit_core  guest memory, dispatch
#   ps2kit_stub  no hardware: tests that only call the game's own code
#   ps2kit_os    the EE kernel at the syscall boundary, the hardware registers,
#                VIF0, VIF1 and the VUs' micro mode, the IPU, the GS, the window and
#                the controller (SDL3), the IOP at the SIF boundary, the disc
# A target links ps2kit_core, then ps2kit_stub or ps2kit_os, then recomp.
set(_rt ${CMAKE_CURRENT_LIST_DIR})
add_library(ps2kit_core OBJECT ${_rt}/core.cpp)
add_library(ps2kit_stub OBJECT ${_rt}/services_stub.cpp)
add_library(ps2kit_os OBJECT ${_rt}/kernel.cpp ${_rt}/hw.cpp ${_rt}/ipu.cpp ${_rt}/gs.cpp ${_rt}/display.cpp
            ${_rt}/input.cpp ${_rt}/vu.cpp ${_rt}/vu0.cpp ${_rt}/vif1.cpp
            ${_rt}/iop.cpp ${_rt}/iop_os.cpp ${_rt}/iop_cpu.cpp ${_rt}/spu2.cpp ${_rt}/disc.cpp ${_rt}/profile.cpp)
# The VU interpreter runs millions of instruction pairs a second: optimised
# beyond the -O1 the recompiled code is built with.
set_source_files_properties(${_rt}/vu.cpp PROPERTIES COMPILE_OPTIONS "-O2;-ffp-contract=off")
# So do the rasteriser and VIF1, for every pixel and every code.
set_source_files_properties(${_rt}/gs.cpp ${_rt}/vif1.cpp PROPERTIES COMPILE_OPTIONS "-O2")
# The IOP's interpreter and the SPU2, for every instruction and every sample.
set_source_files_properties(${_rt}/iop_cpu.cpp ${_rt}/spu2.cpp PROPERTIES COMPILE_OPTIONS "-O2")
find_package(SDL3 REQUIRED CONFIG)
target_link_libraries(ps2kit_os PUBLIC SDL3::SDL3)
foreach(t ps2kit_core ps2kit_stub ps2kit_os)
  target_include_directories(${t} PUBLIC ${_rt})
endforeach()
# difftest: one recompiled function on a given machine state (ps2kit/recomp/difftest.py)
add_executable(difftest ${_rt}/difftest.cpp)
target_link_libraries(difftest ps2kit_core ps2kit_stub recomp)
# gsreplay: a GS recording (PS2KIT_GS_DUMP) played again, timed, the two rasterisers compared
add_executable(gsreplay ${_rt}/gsreplay.cpp ${_rt}/gs.cpp ${_rt}/profile.cpp)
target_include_directories(gsreplay PRIVATE ${_rt})
# with line information, for its --profile and addr2line
target_compile_options(gsreplay PRIVATE -g)
target_link_options(gsreplay PRIVATE -g)
# roundcheck: the fast rasteriser's filtered rounding against the reference's, every float
add_executable(roundcheck ${_rt}/roundcheck.cpp)
# vubench: a VU1 micro program from a PS2KIT_VU1_DUMP, timed and its results written
add_executable(vubench ${_rt}/vubench.cpp ${_rt}/vu.cpp)
target_compile_options(vubench PRIVATE -g)
target_link_options(vubench PRIVATE -g)
target_link_libraries(vubench ps2kit_core ps2kit_stub recomp)
# vutrans: VU1 microcode from a PS2KIT_VU1_DUMP translated to C++ (vu1_trans.cpp)
# (alone: not linked with the recompiled code, which holds what it writes)
add_executable(vutrans ${_rt}/vutrans.cpp ${_rt}/vu.cpp)
target_include_directories(vutrans PRIVATE ${_rt})
# The game's VU1 programs translated by vutrans, when its build has them
# (vu1_trans.cpp beside the recompiled code); else none: the interpreter runs
# them all. Optimised like the interpreter, and without fused multiply-adds,
# so that both compute the same bits.
file(GLOB _vu1_trans CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vu1_trans.cpp)
if(_vu1_trans)
  target_sources(recomp PRIVATE ${_vu1_trans})
  set_source_files_properties(${_vu1_trans} PROPERTIES COMPILE_OPTIONS "-O2;-ffp-contract=off;-Wno-unused-label")
else()
  target_sources(recomp PRIVATE ${_rt}/vu1_none.cpp)
endif()
# ps2run: the game, from its own executable (runtime/boot.cpp)
add_executable(ps2run ${_rt}/boot.cpp)
target_link_libraries(ps2run ps2kit_core ps2kit_os recomp)
if(WIN32)
  target_link_libraries(ps2run winmm)
  add_custom_command(TARGET ps2run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:ps2run>)
endif()

# The ps2kit runtime, included by the generated CMakeLists.txt.
#   ps2kit_core  guest memory, dispatch
#   ps2kit_stub  no hardware: tests that only call the game's own code
#   ps2kit_os    the EE kernel at the syscall boundary, the hardware registers,
#                VIF0 and VU0's micro mode, the IPU, the GS, the window and
#                the controller (SDL3), the IOP at the SIF boundary, the disc
# A target links ps2kit_core, then ps2kit_stub or ps2kit_os, then recomp.
set(_rt ${CMAKE_CURRENT_LIST_DIR})
add_library(ps2kit_core OBJECT ${_rt}/core.cpp)
add_library(ps2kit_stub OBJECT ${_rt}/services_stub.cpp)
add_library(ps2kit_os OBJECT ${_rt}/kernel.cpp ${_rt}/hw.cpp ${_rt}/ipu.cpp ${_rt}/gs.cpp ${_rt}/display.cpp
            ${_rt}/input.cpp ${_rt}/vu0.cpp ${_rt}/vif1.cpp
            ${_rt}/iop.cpp ${_rt}/disc.cpp)
find_package(SDL3 REQUIRED CONFIG)
target_link_libraries(ps2kit_os PUBLIC SDL3::SDL3)
foreach(t ps2kit_core ps2kit_stub ps2kit_os)
  target_include_directories(${t} PUBLIC ${_rt})
endforeach()
# difftest: one recompiled function on a given machine state (ps2kit/recomp/difftest.py)
add_executable(difftest ${_rt}/difftest.cpp)
target_link_libraries(difftest ps2kit_core ps2kit_stub recomp)
# ps2run: the game, from its own executable (runtime/boot.cpp)
add_executable(ps2run ${_rt}/boot.cpp)
target_link_libraries(ps2run ps2kit_core ps2kit_os recomp)
if(WIN32)
  target_link_libraries(ps2run winmm)
  add_custom_command(TARGET ps2run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:ps2run>)
endif()

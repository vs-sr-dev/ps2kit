// ps2kit runtime — what the runtime's own files share (not seen by generated code).
#pragma once
#include "ee.h"

#include <cstdarg>

[[noreturn]] void rt_die(const char* fmt, ...);
void rt_init_memory();
void rt_init_fpu();                          // per thread that runs guest code
const EEFuncEntry* rt_find(uint32_t addr);
const char* rt_name_of(uint32_t addr);       // the function containing addr

// The hardware behind every address that is not RAM or scratchpad.
uint64_t ee_hw_read(uint32_t a, int size);
void     ee_hw_write(uint32_t a, uint64_t v, int size);
void     ee_hw_write128(uint32_t a, const EEReg& v);

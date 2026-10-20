// ps2kit runtime — no VU1 program translated: a game's build without vutrans's
// output (vu1_trans.cpp) runs every VU1 program in the interpreter.
#include "vu_ops.h"

extern const VUTrans g_vu1_trans[] = {{0, nullptr, nullptr}};
extern const size_t g_vu1_ntrans = 0;

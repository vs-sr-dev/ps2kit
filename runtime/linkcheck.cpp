// Link check: every recompiled function, the dispatch table and the runtime
// services link into one executable. Nothing runs.
#include "ee.h"
#include <cstddef>
#include <cstdio>

extern const EEFuncEntry g_ee_funcs[];
extern const size_t g_ee_nfuncs;

int main() {
    std::printf("linked %zu recompiled functions; first %08X (%s), last %08X (%s)\n", g_ee_nfuncs,
                g_ee_funcs[0].addr, g_ee_funcs[0].name,
                g_ee_funcs[g_ee_nfuncs - 1].addr, g_ee_funcs[g_ee_nfuncs - 1].name);
    return 0;
}

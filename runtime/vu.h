// ps2kit runtime — the micro-mode interpreter both vector units share (vu.cpp).
// Not seen by generated code: vu0.cpp and vif1.cpp drive it.
#pragma once
#include "os.h"

#include <set>

struct VUUnit {
    int n;                               // 0 or 1
    EEContext* c;                        // vf, vacc, vi: VU0's are the calling thread's, VU1's its own
    uint8_t* micro;                      // micro memory: 4 KB (VU0), 16 KB (VU1)
    uint32_t micro_mask;
    uint8_t* mem;                        // data memory: 4 KB (VU0), 16 KB (VU1)
    uint32_t mem_mask;
    uint32_t p = 0;                      // VU1's P register (the EFU's result)
    // VU1 only: XGKICK (a GIF packet at a quadword address) and VIF1's TOP and ITOP.
    void (*xgkick)(uint32_t qaddr) = nullptr;
    uint32_t (*xtop)() = nullptr;
    uint32_t (*xitop)() = nullptr;
    uint64_t micro_gen = 0;              // changed by whoever writes micro memory (decoded pairs are dropped)
    struct VUDecoded* dec = nullptr;     // vu.cpp's decoded pairs
    uint64_t runs = 0, pairs = 0, cycles = 0;
    uint64_t old_q = 0, old_p = 0, old_flags = 0;   // reads that saw a value still in its pipeline's past
    std::set<uint32_t> warned;           // addresses already reported
};

// Runs the micro program at `start` (bytes into micro memory) to its end.
void vu_exec(VUUnit& u, uint32_t start);

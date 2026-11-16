// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <hggc_runtime.h>

extern "C" __device__ int megamhc_wait(uint64_t address, uint64_t expected) {
  if (threadIdx.x == 0) {
    uint64_t observed;
    do {
      asm volatile("ppu.ld.global.gpu.acquire.b64 %0, [ %1 + 0 ];"
                   : "=l"(observed) : "l"(address) : "memory");
    } while (observed < expected);
  }
  __syncthreads();
  return 0;
}

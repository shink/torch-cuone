#pragma once

#include <cuda.h>

#include "src/c10/CU1Macros.h"

#define C10_CU1_CHECK_WARN(EXPR)                                              \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                                  \
      [[maybe_unused]] auto error_unused = cudaGetLastError();                 \
      TORCH_WARN("CUDA warning: ", cudaGetErrorString(__err));                 \
    }                                                                          \
  } while (0)

#define C10_CU1_CHECK(EXPR)                                                    \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    c10_cu1::c10_cu1_check_implementation(static_cast<int32_t>(__err));        \
  } while (0)

namespace c10_cu1 {
C10_CU1_API void c10_cu1_check_implementation(const int32_t err);
} // namespace c10_cu1

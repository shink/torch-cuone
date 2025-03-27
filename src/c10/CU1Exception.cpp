#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include "src/c10/CU1Exception.h"

namespace c10_cu1 {

void c10_cu1_check_implementation(const int32_t err) {
  const auto cuda_error = static_cast<cudaError_t>(err);
  if (C10_LIKELY(cuda_error == cudaSuccess)) {
    return;
  }

  TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(cuda_error));
}

} // namespace c10_cu1

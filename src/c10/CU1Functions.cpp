#include <c10/macros/Macros.h>
#include <c10/util/WaitCounter.h>
#include <limits>

#include "src/c10/CU1Functions.h"

namespace c10_cu1 {
namespace {
int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = c10_cu1::GetDeviceCount(&count);
  if (err == cudaSuccess) {
    return count;
  } else if (err == cudaErrorNoDevice) {
    count = 0; // Zero devices is ok here
  } else {
    C10_CU1_CHECK(err);
  }
}
} // namespace

c10::DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(result <=
                                std::numeric_limits<c10::DeviceIndex>::max(),
                            "Too many CUDA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error &ex) {
      TORCH_WARN("CUDA initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No CUDA GPUs are available");
  TORCH_INTERNAL_ASSERT(count <= std::numeric_limits<c10::DeviceIndex>::max(),
                        "Too many CUDA devices, DeviceIndex overflowed");
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex current_device() {
  c10::DeviceIndex cur_device = -1;
  C10_CU1_CHECK(c10_cu1::GetDevice(&cur_device));
  return cur_device;
}

void set_device(c10::DeviceIndex device) {
  C10_CU1_CHECK(c10_cu1::SetDevice(device));
}

void device_synchronize() { C10_CU1_CHECK(cudaDeviceSynchronize()); }

// Wrappers for raw CUDA device management functions
cudaError_t GetDeviceCount(int *dev_count) {
  return cudaGetDeviceCount(dev_count);
}

thread_local static c10::DeviceIndex targetDeviceIndex = -1;

cudaError_t GetDevice(c10::DeviceIndex *device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return cudaSuccess;
  }
  int tmp_device = -1;
  auto err = cudaGetDevice(&tmp_device);
  if (err == cudaSuccess) {
    TORCH_INTERNAL_ASSERT(tmp_device >= 0 &&
                              tmp_device <=
                                  std::numeric_limits<c10::DeviceIndex>::max(),
                          "cudaGetDevice returns invalid device ", tmp_device);
    *device = static_cast<c10::DeviceIndex>(tmp_device);
  }
  return err;
}

cudaError_t SetDevice(c10::DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_CU1_CHECK(cudaGetDevice(&cur_device));
  if (device == cur_device) {
    return cudaSuccess;
  }
  return cudaSetDevice(device);
}

cudaError_t MaybeSetDevice(c10::DeviceIndex device) {
  return c10_cu1::SetDevice(device);
}

// This function always initializes the CUDA context
// on to_device
c10::DeviceIndex ExchangeDevice(c10::DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_CU1_CHECK(cudaGetDevice(&tmp_device));
    cur_device = static_cast<c10::DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_CU1_CHECK(cudaSetDevice(to_device));
  return cur_device;
}

} // namespace c10_cu1

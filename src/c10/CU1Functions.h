#pragma once

#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include "src/c10/CU1Exception.h"
#include "src/c10/CU1Macros.h"

namespace c10_cu1 {

// People basically ~never want this function to fail; it should
// just return zero if things are not working.
// It still might log a warning for user first time it's invoked
C10_CU1_API c10::DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
C10_CU1_API c10::DeviceIndex device_count_ensure_non_zero();

C10_CU1_API c10::DeviceIndex current_device();

C10_CU1_API void set_device(c10::DeviceIndex device);

C10_CU1_API void device_synchronize();

// Raw CUDA device management functions
C10_CU1_API cudaError_t GetDeviceCount(int *dev_count);

C10_CU1_API cudaError_t GetDevice(c10::DeviceIndex *device);

C10_CU1_API cudaError_t SetDevice(c10::DeviceIndex device);

C10_CU1_API cudaError_t MaybeSetDevice(c10::DeviceIndex device);

C10_CU1_API c10::DeviceIndex ExchangeDevice(c10::DeviceIndex device);

} // namespace c10_cu1

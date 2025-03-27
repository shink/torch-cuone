#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <cassert>

#include "src/c10/CU1Functions.h"

namespace c10_cu1 {
namespace impl {

struct CU1GuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  CU1GuardImpl() {}

  explicit CU1GuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1,
                          "DeviceType must be PrivateUse1, but actual is: ", t);
  }

  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(
        d.is_privateuseone(),
        "DeviceType must be PrivateUse1, but actual is: ", d.type());
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      c10_cu1::set_device(d.index());
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    auto device = c10_cu1::current_device();
    return c10::Device(c10::DeviceType::PrivateUse1, device);
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(
        d.type() == c10::DeviceType::PrivateUse1,
        "DeviceType must be PrivateUse1, but actual is: ", d.type());
    c10_cu1::set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    C10_CU1_CHECK_WARN(c10_cu1::MaybeSetDevice(d.index()));
  }

  // TODO
  c10::Stream getStream(c10::Device d) const noexcept override {}

  // TODO
  c10::Stream getDefaultStream(c10::Device d) const override {}

  // TODO
  c10::Stream
  getStreamFromGlobalPool(c10::Device d,
                          bool isHighPriority = false) const override {}

  // TODO
  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {}

  // TODO
  c10::DeviceIndex deviceCount() const noexcept override {}

  // TODO
  void
  destroyEvent(void *event,
               const c10::DeviceIndex device_index) const noexcept override {}

  // TODO
  void record(void **event, const c10::Stream &stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {}

  // TODO
  void block(void *event, const c10::Stream &stream) const override {}

  // TODO
  // May be called from any device
  bool queryEvent(void *event) const override {}

  // TODO
  void recordDataPtrOnStream(const c10::DataPtr &data_ptr,
                             const c10::Stream &stream) const override {}
};

} // namespace impl
} // namespace c10_cu1

#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Storage.h>

namespace at_cu1 {

struct TORCH_API CU1HooksInterface : public at::PrivateUse1HooksInterface {
  virtual ~CU1HooksInterface() = default;

  const at::Generator &getDefaultGenerator(c10::DeviceIndex device_index) {
    static at::Generator default_generator;
    return default_generator;
  }

  void init() const override;

  at::Generator
  getNewGenerator(c10::DeviceIndex device_index = -1) const override;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  void resizePrivateUse1Bytes(const c10::Storage &storage,
                              size_t new_bytes) const;

  bool isAvailable() const override;
};

struct TORCH_API CU1HooksArgs : public at::PrivateUse1HooksArgs {};

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface *get_cu1_hooks();

} // namespace at_cu1

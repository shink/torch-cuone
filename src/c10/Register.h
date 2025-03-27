#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Registry.h>

// Declared in pytorch/c10/core/DeviceType.h
#define C10_REGISTER_PRIVATEUSE1_BACKEND(name)                                 \
  int register_privateuse1_backend() {                                         \
    c10::register_privateuse1_backend(#name);                                  \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp_backend_##name = register_privateuse1_backend();

// Declared in pytorch/c10/core/StorageImpl.h
#define C10_SET_STORAGE_IMPL_CREATE(make_storage_impl)                         \
  int set_storage_impl_create() {                                              \
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1,                    \
                              make_storage_impl);                              \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp_storage_impl = set_storage_impl_create();

// Declared in pytorch/c10/core/impl/DeviceGuardImplInterface.h
#define C10_REGISTER_PRIVATEUSE1_GUARD_IMPL(guard_impl)                        \
  C10_REGISTER_GUARD_IMPL(PrivateUse1, guard_impl)

// Declared in pytorch/c10/util/Registry.h
#define C10_REGISTER_PRIVATEUSE1_CLASS(clsname)                                \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

// Declared in pytorch/c10/util/Registry.h
#define C10_TORCH_DECLARE_PRIVATEUSE1_REGISTRY(...)                            \
  TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, ##__VA_ARGS__)

// Declared in pytorch/c10/util/Registry.h
#define C10_DEFINE_PRIVATEUSE1_REGISTRY(...)                                   \
  C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, ##__VA_ARGS__)

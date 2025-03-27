#pragma once

// Declared in pytorch/torch/csrc/jit/serialization/pickler.h
#define REGISTER_TENSOR_BACKEND_META_REGISTRY(serialization, deserialization)  \
  int register_tensor_backend_meta_registry() {                                \
    torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1,        \
                                          serialization, deserialization);     \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp_tensor_meta = register_tensor_backend_meta_registry();

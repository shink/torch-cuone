#pragma once

// Declared in pytorch/ATen/detail/PrivateUse1HooksInterface.h
#define AT_REGISTER_PRIVATEUSE1_HOOKS_INTERFACE(get_hooks)                     \
  int register_privateuse1_hooks_interface() {                                 \
    at::RegisterPrivateUse1HooksInterface(get_hooks);                          \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp_hooks = register_privateuse1_hooks_interface();

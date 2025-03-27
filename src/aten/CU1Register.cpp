#include "src/aten/CU1HooksInterface.h"
#include "src/aten/Register.h"
#include "src/c10/Register.h"

AT_REGISTER_PRIVATEUSE1_HOOKS_INTERFACE(at_cu1::get_cu1_hooks());

C10_TORCH_DECLARE_PRIVATEUSE1_REGISTRY(at_cu1::CU1HooksInterface,
                                       at_cu1::CU1HooksArgs);

C10_DEFINE_PRIVATEUSE1_REGISTRY(at_cu1::CU1HooksInterface,
                                at_cu1::CU1HooksArgs);

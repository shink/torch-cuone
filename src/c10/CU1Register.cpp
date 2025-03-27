#include <c10/core/DeviceType.h>

#include "src/c10/CU1GuardImpl.h"
#include "src/c10/Register.h"

C10_REGISTER_PRIVATEUSE1_BACKEND(cuone);

C10_REGISTER_PRIVATEUSE1_GUARD_IMPL(c10_cu1::impl::CU1GuardImpl);

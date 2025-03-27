#include <c10/util/CallOnce.h>

#include "src/aten/CU1HooksInterface.h"
#include "src/c10/CU1Functions.h"

namespace at_cu1 {

void CU1HooksInterface::init() const {}

// TODO
at::Generator
CU1HooksInterface::getNewGenerator(c10::DeviceIndex device_index) const {
  return at::Generator();
}

// TODO
bool CU1HooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
  return false;
}

// TODO
void CU1HooksInterface::resizePrivateUse1Bytes(const c10::Storage &storage,
                                               size_t new_bytes) const {}

bool CU1HooksInterface::isAvailable() const {
  return c10_cu1::device_count() > 0;
}

at::PrivateUse1HooksInterface *get_cu1_hooks() {
  static at::PrivateUse1HooksInterface *cu1_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] { cu1_hooks = new CU1HooksInterface(); });
  return cu1_hooks;
}

} // namespace at_cu1

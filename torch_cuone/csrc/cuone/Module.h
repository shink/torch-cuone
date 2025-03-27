#pragma once

void RegisterNPUDeviceProperties(PyObject *module);
void BindGetDeviceProperties(PyObject *module);
void RegisterNPUDeviceMemories(PyObject *module);
void BindGetDeviceMemories(PyObject *module);
void RegisterNpuPluggableAllocator(PyObject *module);
void initCommMethods();
PyObject *THNPModule_getDevice_wrap(PyObject *self);
PyObject *THNPModule_setDevice_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDeviceName_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDriverVersion(PyObject *self);
PyObject *THNPModule_isDriverSufficient(PyObject *self);
PyObject *THNPModule_getCurrentBlasHandle_wrap(PyObject *self);

#define CHANGE_UNIT_SIZE 1024.0

#include <Python.h>
#include <ATen/Parallel.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>

#include "torch_cuone/csrc/cuone/Module.h"

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
    if (!vector.empty()) {
        // remove nullptr terminator
        vector.pop_back();
    }
    while (true) {
        vector.push_back(*methods);
        if (!methods->ml_name) {
            break;
        }
        methods++;
    }
}

PyObject* module;

PyMethodDef* THCU1Module_get_methods();

static std::vector<PyMethodDef> methods;

PyObject* initModule() {
    AddPyMethodDefs(methods, THCU1Module_get_methods());

    static struct PyModuleDef torch_cuone_module = {
        PyModuleDef_HEAD_INIT,
        "torch_cuone._C",
        nullptr,
        -1,
        methods.data()
    };
    module = PyModule_Create(&torch_cuone_module);
    return module;
}

PyMODINIT_FUNC PyInit__C(void) {
    return initModule();
}

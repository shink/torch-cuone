#include <chrono>
#include <future>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

static PyObject *THCU1Module_foo(PyObject *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  std::string msg = "Hello from the CU1 module!";
  std::cout << msg << std::endl;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THCU1Module_methods[] = {
    {"_foo", (PyCFunction)THCU1Module_foo, METH_NOARGS, nullptr},
    {nullptr}
};

PyMethodDef *THCU1Module_get_methods() { return THCU1Module_methods; }

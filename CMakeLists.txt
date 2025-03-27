cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_CUONE C CXX CUDA)

set(PLUGIN_NAME torch_cuone)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR lib)

# NOTE: Python is required
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# NOTE: PyTorch is required
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE PYTORCH_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# NOTE: CUDA is required
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

set(CPP_SRC)
set(PY_SRC)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_cuone/csrc)
add_library(${PLUGIN_NAME} SHARED ${CPP_SRC} ${PY_SRC})

target_link_libraries(${PLUGIN_NAME} PUBLIC ${TORCH_LIBRARIES})
install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

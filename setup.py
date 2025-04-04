import multiprocessing
import os
import shutil
import subprocess
import sys

import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.sdist
from setuptools import Extension, find_packages, setup

# Disable autoloading at the beginning of process
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

TORCH_DIR = os.path.dirname(os.path.realpath(torch.__file__))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")
PACKAGE_DIR = os.path.join(BASE_DIR, "torch_cuone")

CUDA_HOME = os.getenv("CUDA_HOME", "/usr/local/cuda")

DEBUG = os.getenv("DEBUG", "0") == "1"
USE_CXX11_ABI = os.getenv("_GLIBCXX_USE_CXX11_ABI", "1") == "1"
BUILD_TEST = os.getenv("BUILD_TEST", "OFF")
MAX_JOBS = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))

RUN_BUILD_DEPS = True
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg in ["clean", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args


def get_cmake_build_type():
    build_type = "Release"
    if DEBUG:
        build_type = "Debug"
    return build_type


# TODO: Codegen
def generate_bindings_code():
    pass


def CppExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for C++.
    """
    kwargs["language"] = "c++"

    # include_dirs
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs.append(os.path.join(TORCH_DIR, "include"))
    include_dirs.append(os.path.join(TORCH_DIR, "include/torch/csrc/api/include"))
    include_dirs.append(os.path.join(CUDA_HOME, "include"))
    kwargs["include_dirs"] = include_dirs

    # library_dirs
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs.append(os.path.join(TORCH_DIR, "lib"))
    library_dirs.append(os.path.join(CUDA_HOME, "lib64"))
    kwargs["library_dirs"] = library_dirs

    # libraries
    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")
    libraries.append("cudart")
    kwargs["libraries"] = libraries

    # extra_link_args
    extra_link_args = kwargs.get("extra_link_args", [])
    extra_link_args.append("-lcudart")
    kwargs["extra_link_args"] = extra_link_args
    return Extension(name, sources, *args, **kwargs)


class build_ext(setuptools.command.build_ext.build_ext):
    pass


class install(setuptools.command.install.install):
    def run(self):
        super().run()


class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        def _rm(dir):
            path = os.path.join(BASE_DIR, dir)
            print(f"Removing {path}", end="...")
            try:
                os.remove(path)
            except OSError:
                shutil.rmtree(path, ignore_errors=True)
            finally:
                print("OK")

        dirs = ["build", "dist", "torch_cuone.egg-info"]
        for dir in dirs:
            _rm(dir)


def build_deps():
    # TODO: Codegen
    # generate_bindings_code(BASE_DIR)

    cmake = "cmake"
    if cmake is None:
        raise RuntimeError("cmake is required")

    build_dir = os.path.join(BUILD_DIR)
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_BUILD_TYPE=" + get_cmake_build_type(),
        "-DCMAKE_INSTALL_PREFIX=" + os.path.realpath(PACKAGE_DIR),
        # "-DPYTHON_INCLUDE_DIR=" + get_paths().get("include"),
    ]

    subprocess.check_call([cmake, BASE_DIR] + cmake_args, cwd=BUILD_DIR, env=os.environ)

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--",
    ]

    build_args += ["-j", MAX_JOBS]

    command = [cmake] + build_args
    subprocess.check_call(command, cwd=BUILD_DIR, env=os.environ)


def configure_extension_build():
    # Include directories
    include_directories = [
        BASE_DIR,
    ]

    # Extra compile and link args
    extra_link_args = []
    extra_compile_args = [
        "-std=c++17",
        "-Wno-sign-compare",
        "-Wno-deprecated-declarations",
        "-Wno-return-type",
    ]
    if DEBUG:
        extra_compile_args += ["-O0", "-g"]
        extra_link_args += ["-O0", "-g", "-Wl,-z,now"]
    else:
        extra_compile_args += ["-DNDEBUG"]
        extra_link_args += ["-Wl,-z,now"]

    # Extension
    extension = []
    C = CppExtension(
        name="torch_cuone._C",
        sources=["torch_cuone/csrc/InitBindings.cpp"],
        libraries=["torch_cuone"],
        library_dirs=["lib", os.path.join(BASE_DIR, "torch_cuone/lib")],
        include_dirs=include_directories,
        extra_compile_args=extra_compile_args
        + ["-fstack-protector-all"]
        + ['-D__FILENAME__="InitBindings.cpp"'],
        extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
        define_macros=[
            ("_GLIBCXX_USE_CXX11_ABI", "1" if USE_CXX11_ABI else "0"),
            ("GLIBCXX_USE_CXX11_ABI", "1" if USE_CXX11_ABI else "0"),
        ],
    )
    extension.append(C)

    cmdclass = {
        "build_ext": build_ext,
        "clean": clean,
        "install": install,
    }

    excludes = ["codegen", "codegen.*"]
    packages = find_packages(exclude=excludes)
    return extension, cmdclass, packages


def main():
    install_requires = ["pyyaml"]
    if sys.version_info >= (3, 12, 0):
        install_requires.append("setuptools")

    if RUN_BUILD_DEPS:
        build_deps()

    (
        extensions,
        cmdclass,
        packages,
    ) = configure_extension_build()

    setup(
        name="torch_cuone",
        version="0.1",
        description="CUDA implementation for PyTorch via PrivateUseOne mechanism",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        keywords="pytorch, cuda, privateuse1",
        python_requires=">=3.8",
        license="BSD-3-Clause",
        url="https://github.com/shink/torch-cuone",
        author="Yuanhao Ji",
        author_email="jiyuanhao@apache.org",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
        ],
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        package_data={
            "torch_cuone": [
                "*.so",
                "lib/*.so*",
            ],
        },
        install_requires=install_requires,
        entry_points={
            "torch.backends": [
                "torch_cuone = torch_cuone:_autoload",
            ],
        },
    )


if __name__ == "__main__":
    main()

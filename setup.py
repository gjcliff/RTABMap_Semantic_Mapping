import os
import subprocess
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cfg = "Release" if not self.debug else "Debug"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = ["--config", cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake commands
        subprocess.check_call(
            ["cmake", ext.sourcedir, *cmake_args], cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args], cwd=self.build_temp)


class CMakeExtension(Extension):
    """An extension to pass to setuptools that triggers a CMake build."""

    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


setup(
    name="rtabmap_database_export",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CMakeExtension("database_exporter_py", sourcedir=".")
    ],
    cmdclass={
        "build_ext": CMakeBuildExt,
    },
    zip_safe=False,
)

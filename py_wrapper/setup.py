from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "database_exporter_py",  # Python module name
        [
            "py_wrapper/wrapper.cpp",  # Source file for Pybind11
            "src/database_exporter.cpp",  # Add the implementation of DatabaseExporter
        ],
        include_dirs=["../include"],  # Include directory for headers
    ),
]

setup(
    name="database_exporter_py",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

#include "../include/database_exporter.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(database_exporter_py, m) {
  py::class_<DatabaseExporter>(m, "DatabaseExporter")
      .def(py::init<std::string, std::string>())
      .def("load_rtabmap_db", &DatabaseExporter::load_rtabmap_db,
           "Loads the rtabmap database");
}

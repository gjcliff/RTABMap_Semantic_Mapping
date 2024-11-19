#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util2d.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_surface.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UTimer.h>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav2_map_server/map_io.hpp>
#include <pcl/cloud_iterator.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include "pcl/io/pcd_io.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <Python.h>

#include <filesystem>
#include <iostream>

namespace py = pybind11;

class DatabaseExporter {
public:
  DatabaseExporter(std::string rtabmap_database_name,
                           std::string model_name);

  ~DatabaseExporter();
  bool load_rtabmap_db();
  void get_detections(py::object &net);

private:
  cv::Mat numpy_to_mat(const py::array_t<uint8_t> &np_array);
  py::array mat_to_numpy(const cv::Mat &mat);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  nav_msgs::msg::OccupancyGrid::SharedPtr
  point_cloud_to_occupancy_grid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  std::string generate_timestamp_string();

  cv::dnn::Net net_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rtabmap_cloud_;
  nav_msgs::msg::OccupancyGrid::SharedPtr rtabmap_occupancy_grid_;

  std::string rtabmap_database_path_;
  std::string model_path_;
  std::string timestamp_;
  std::vector<cv::Mat> images_;
  std::vector<cv::Mat> depths_;
  std::vector<std::vector<rtabmap::CameraModel>> camera_models_;
  std::vector<std::vector<rtabmap::StereoCameraModel>> stereo_models_;

  bool export_images_;
};

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

#include <nav2_map_server/map_io.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl/cloud_iterator.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "pcl/io/pcd_io.h"
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <filesystem>
#include <iostream>
#include <random>

namespace py = pybind11;

struct Result {
  bool success = false;
  std::string timestamp = "";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  std::list<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                       std::map<std::pair<int, int>, int>>>
      mapping_data;
};

struct BoundingBox {
  int x1, y1, x2, y2;
  BoundingBox(int x1, int y1, int x2, int y2)
      : x1(x1), y1(y1), x2(x2), y2(y2) {}
};

class DatabaseExporter {
public:
  DatabaseExporter(std::string rtabmap_database_name, std::string model_name);
  ~DatabaseExporter();

  // @brief Convert a numpy array to a cv::Mat
  // @param np_array The numpy array
  // @return The cv::Mat
  cv::Mat numpy_to_mat(const py::array_t<uint8_t> &np_array);

  // @brief Convert a cv::Mat to a numpy array
  // @param mat The cv::Mat
  // @return The numpy array
  py::array mat_to_numpy(const cv::Mat &mat);

  // @brief Load the rtabmap database
  // @return The result of the operation
  Result load_rtabmap_db();

  // @brief Get the detections from the neural network
  // @param net The neural network
  // @return The detections
  void get_detections(py::object &net);

private:
  // @brief Filter the point cloud using statistical and radius outlier removal
  // @param cloud The point cloud
  // @return The filtered point cloud
  // @return The filtered point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  // @brief Project the point cloud onto the camera image, and keep track of
  // which points are in each image
  // @param image_size The size of the image
  // @param camera_matrix The camera matrix
  // @param cloud The point cloud
  // @param camera_transform The transform of the camera
  // @return The input image overlayed with the point cloud projection
  std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
  project_cloud_to_camera(const cv::Size &image_size,
                          const cv::Mat &camera_matrix,
                          const pcl::PCLPointCloud2::Ptr cloud,
                          const rtabmap::Transform &camera_transform);

  // @brief Convert a point cloud to an occupancy grid
  // @param cloud The point cloud
  // @return The occupancy grid
  nav_msgs::msg::OccupancyGrid::SharedPtr
  point_cloud_to_occupancy_grid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  // @brief Generate a timestamp string
  // @return The timestamp string in the format %Y-%m-%d_%H-%M-%S
  std::string generate_timestamp_string();

  cv::dnn::Net net_;
  nav_msgs::msg::OccupancyGrid::SharedPtr rtabmap_occupancy_grid_;

  std::string rtabmap_database_path_;
  std::string model_path_;
  std::string timestamp_;
  std::list<cv::Mat> images_;
  std::vector<std::vector<rtabmap::CameraModel>> camera_models_;
  std::vector<std::vector<rtabmap::StereoCameraModel>> stereo_models_;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rtabmap_cloud_;
  // rgb image, depth image, transform from camera to world, pixel to point map
  std::list<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                       std::map<std::pair<int, int>, int>>>
      mapping_data_;

  bool export_images_;

  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<> dis_;
};

#include "database_exporter.hpp"

struct MouseData {
    cv::Mat image;
    cv::Point start_point;
    cv::Point end_point;
    cv::Rect bounding_box;
    bool drawing = false;
    bool finished = false;
};

class DatabaseUtils {
public:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud_from_bounding_box(
      std::tuple<std::string, float, BoundingBox> bounding_box,
      std::map<std::pair<int, int>, int> pixel_to_point_map,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, rtabmap::Transform pose);

  std::vector<Object> semantic_mapping(
      py::object &net, DatabaseExporter &exporter,
      std::list<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                           std::map<std::pair<int, int>, int>>> &mapping_data,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::string &timestamp);

  void mouse_callback(int event, int x, int y, int, void *data);
  void manual_labeling(Result &result, std::vector<Object> &objects);
};

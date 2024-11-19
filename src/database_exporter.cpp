#include "database_exporter.hpp"
#include "pylifecycle.h"
#include <pybind11/attr.h>
#include <pybind11/embed.h>

DatabaseExporter::DatabaseExporter(std::string rtabmap_database_name,
                                   std::string model_name)
    : timestamp_(generate_timestamp_string()) {
  // check if database name is empty
  if (rtabmap_database_name.empty()) {
    std::cout << "RTABMap database name is empty" << std::endl;
    return;
  }

  rtabmap_database_path_ =
      std::string(PROJECT_PATH) + "/databases/" + rtabmap_database_name;

  // initialize shared pointers for the cloud and occupancy grid
  rtabmap_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  rtabmap_occupancy_grid_ =
      nav_msgs::msg::OccupancyGrid::SharedPtr(new nav_msgs::msg::OccupancyGrid);

  // check if model name is empty
  if (model_name.empty()) {
    std::cout << "Model name is empty, not performing semantic mapping"
              << std::endl;
  } else {
    model_path_ = std::string(PROJECT_PATH) + "/models/" + model_name;
    std::cout << "Loading model: " << model_path_ << std::endl;

    net_ = cv::dnn::readNet(model_path_);
  }

  // create output directories for the deconstructor
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;
  if (!std::filesystem::create_directory(path)) {
    std::cout << "Failed to create output directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/depths")) {
    std::cout << "Failed to create depths directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/cloud")) {
    std::cout << "Failed to create cloud directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/grid")) {
    std::cout << "Failed to create grid directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/images")) {
    std::cout << "Failed to create images directory" << std::endl;
    return;
  }
}

DatabaseExporter::~DatabaseExporter() {
  // create the output directory
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;

  // save the point cloud
  std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
  pcl::io::savePCDFileBinary(cloud_path, *rtabmap_cloud_);

  // save the occupancy grid
  std::string grid_path = path + "/grid/" + timestamp_;
  rtabmap_occupancy_grid_ = point_cloud_to_occupancy_grid(rtabmap_cloud_);
  nav2_map_server::SaveParameters save_params;
  save_params.map_file_name = grid_path;
  save_params.image_format = "pgm";
  save_params.free_thresh = 0.196;
  save_params.occupied_thresh = 0.65;
  nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, save_params);
  int imagesExported = 0;
  for (size_t i = 0; i < images_.size(); ++i) {
    std::string image_path = path + "/images/" + std::to_string(i) + ".jpg";
    cv::imwrite(image_path, images_.at(i));
    ++imagesExported;
  }
  for (size_t i = 0; i < depths_.size(); ++i) {
    cv::Mat depthExported = depths_.at(i);
    std::string ext;
    std::string depth_path = path + "/depth/" + std::to_string(i);
    if (depths_.at(i).type() != CV_16UC1 && depths_.at(i).type() != CV_32FC1) {
      ext = ".jpg";
    } else {
      ext = ".png";
      if (depths_.at(i).type() == CV_32FC1) {
        depthExported = rtabmap::util2d::cvtDepthFromFloat(depths_.at(i));
      }
    }

    cv::imwrite(depth_path + ext, depthExported);
    ++imagesExported;
  }

  // save calibration per image (calibration can change over time, e.g.
  // camera has auto focus)
  for (size_t i = 0; i < camera_models_.size(); i++) {
    for (size_t j = 0; j < camera_models_.at(i).size(); j++) {
      rtabmap::CameraModel model = camera_models_.at(i).at(j);
      std::string modelName = std::to_string(i);
      std::string dir = path + "/camera_models/";

      if (camera_models_.at(i).size() > 1) {
        modelName += "_" + uNumber2Str((int)j);
      }
      model.setName(modelName);
      model.save(dir);
    }
  }

  for (size_t i = 0; i < stereo_models_.size(); ++i) {
    for (size_t j = 0; j < stereo_models_.at(i).size(); ++j) {
      rtabmap::StereoCameraModel model = stereo_models_.at(i).at(j);
      std::string modelName = std::to_string(i);
      std::string dir = path + "/stereo_models/";
      if (stereo_models_.at(i).size() > 1) {
        modelName += "_" + std::to_string(j);
      }
      model.setName(modelName, "left", "right");
      model.save(dir);
    }
  }
  std::cout << "Images exported: " << imagesExported << std::endl;
}

cv::Mat DatabaseExporter::numpy_to_mat(const py::array_t<uint8_t> &np_array) {
  py::buffer_info buf = np_array.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (uchar *)buf.ptr);
  return mat;
}

py::array DatabaseExporter::mat_to_numpy(const cv::Mat &mat) {
  return py::array_t<uint8_t>({mat.rows, mat.cols, mat.channels()},
                              {mat.step[0], mat.step[1]}, mat.data);
}

void DatabaseExporter::get_detections(py::object &net)
{
  for (const auto &image : images_) {
    py::array np_array = mat_to_numpy(image);
    py::list detections = net.attr("predict")(np_array);
    std::cout << "detections: " << detections.size() << std::endl;
  }
}

nav_msgs::msg::OccupancyGrid::SharedPtr
DatabaseExporter::point_cloud_to_occupancy_grid(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
  // calculate the centroid
  Eigen::Matrix<float, 4, 1> centroid;
  pcl::ConstCloudIterator<pcl::PointXYZRGB> cloud_iterator(*cloud);
  pcl::compute3DCentroid(cloud_iterator, centroid);

  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();

  for (const auto &point : cloud->points) {
    if (point.x > max_x) {
      max_x = point.x;
    }
    if (point.y > max_y) {
      max_y = point.y;
    }
    if (point.x < min_x) {
      min_x = point.x;
    }
    if (point.y < min_y) {
      min_y = point.y;
    }
  }

  nav_msgs::msg::OccupancyGrid::SharedPtr occupancy_grid =
      std::make_shared<nav_msgs::msg::OccupancyGrid>();
  cloud->width = cloud->points.size();
  occupancy_grid->info.resolution = 0.05;
  occupancy_grid->info.width =
      std::abs(max_x - min_x) / occupancy_grid->info.resolution + 1;
  occupancy_grid->info.height =
      std::abs(max_y - min_y) / occupancy_grid->info.resolution + 1;
  occupancy_grid->info.origin.position.x = min_x;
  occupancy_grid->info.origin.position.y = min_y;
  occupancy_grid->info.origin.position.z = 0;
  occupancy_grid->info.origin.orientation.x = 0;
  occupancy_grid->info.origin.orientation.y = 0;
  occupancy_grid->info.origin.orientation.z = 0;
  occupancy_grid->info.origin.orientation.w = 1;
  occupancy_grid->data.resize(
      occupancy_grid->info.width * occupancy_grid->info.height, 0);
  for (const auto &point : cloud->points) {
    int x = (point.x - min_x) / occupancy_grid->info.resolution;
    int y = (point.y - min_y) / occupancy_grid->info.resolution;
    int index = y * occupancy_grid->info.width + x;
    occupancy_grid->data.at(index) = 100;
  }
  return occupancy_grid;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr DatabaseExporter::filter_point_cloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
  // statistical outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(100); // increase for more permissive, decrease for less
  sor.setStddevMulThresh(
      1.0); // increase for more permissive, decrease for less
  sor.filter(*sor_cloud);

  // radius outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr radius_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius_outlier;
  radius_outlier.setInputCloud(sor_cloud);
  radius_outlier.setRadiusSearch(
      0.5); // adjust based on spacing in the point cloud
  radius_outlier.setMinNeighborsInRadius(
      3); // increase for more aggressive outlier removal
  radius_outlier.filter(*radius_cloud);
  radius_cloud->width = radius_cloud->points.size();

  return radius_cloud;
}
std::string DatabaseExporter::generate_timestamp_string() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);

  std::ostringstream oss;

  oss << std::put_time(ptm, "%Y-%m-%d_%H-%M-%S");

  return oss.str();
}

bool DatabaseExporter::load_rtabmap_db() {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  if (driver->openConnection(rtabmap_database_path_)) {
    parameters = driver->getLastParameters();
    driver->closeConnection(false);
  } else {
    std::cout << "Failed to open database" << std::endl;
    return false;
  }
  delete driver;
  driver = 0;

  UTimer timer;

  std::cout << "Loading database: " << rtabmap_database_path_ << std::endl;
  rtabmap::Rtabmap rtabmap;
  rtabmap.init(parameters, rtabmap_database_path_);
  std::cout << "Loaded database in " << timer.ticks() << std::endl << "s";

  std::map<int, rtabmap::Signature> nodes;
  std::map<int, rtabmap::Transform> optimizedPoses;
  std::multimap<int, rtabmap::Link> links;
  std::cout << "Optimizing the map..." << std::endl;
  rtabmap.getGraph(optimizedPoses, links, true, true, &nodes, true, true, true,
                   true);
  std::cout << "Optimizing the map... done (" << timer.ticks()
            << "s, poses=" << optimizedPoses.size() << ")." << std::endl;

  if (optimizedPoses.size() == 0) {
    std::cout << "No optimized poses found" << std::endl;
    return false;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr assembledCloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr assembledCloudI(
      new pcl::PointCloud<pcl::PointXYZI>);
  std::map<int, rtabmap::Transform> robotPoses;
  std::vector<std::map<int, rtabmap::Transform>> cameraPoses;
  std::map<int, rtabmap::Transform> scanPoses;
  std::map<int, double> cameraStamps;
  std::map<int, std::vector<rtabmap::CameraModel>> cameraModels;
  std::map<int, cv::Mat> cameraDepths;
  std::vector<int> rawViewpointIndices;
  std::map<int, rtabmap::Transform> rawViewpoints;
  int imagesExported = 0;
  for (std::map<int, rtabmap::Transform>::iterator iter =
           optimizedPoses.lower_bound(1);
       iter != optimizedPoses.end(); ++iter) {
    rtabmap::Signature node = nodes.find(iter->first)->second;

    // uncompress data
    std::vector<rtabmap::CameraModel> models = node.sensorData().cameraModels();
    std::vector<rtabmap::StereoCameraModel> stereoModels =
        node.sensorData().stereoCameraModels();

    cv::Mat rgb;
    cv::Mat depth;

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudI;
    if (node.getWeight() != -1) {
      int decimation = 4;
      int maxRange = 4.0;
      int minRange = 0.0;
      float noiseRadius = 0.0f;
      int noiseMinNeighbors = 5;
      bool exportImages = true;
      bool texture = true;
      cv::Mat tmpDepth;
      rtabmap::LaserScan scan;
      node.sensorData().uncompressData(
          exportImages ? &rgb : 0,
          (texture || exportImages) &&
                  !node.sensorData().depthOrRightCompressed().empty()
              ? &tmpDepth
              : 0,
          &scan);
      if (scan.empty()) {
        std::cout << "Node " << iter->first
                  << " doesn't have scan data, empty cloud is created."
                  << std::endl;
      }
      if (decimation > 1 || minRange > 0.0f || maxRange) {
        scan = rtabmap::util3d::commonFiltering(scan, decimation, minRange,
                                                maxRange);
      }
      if (scan.hasRGB()) {
        cloud = rtabmap::util3d::laserScanToPointCloudRGB(
            scan, scan.localTransform());
        if (noiseRadius > 0.0f && noiseMinNeighbors > 0) {
          indices = rtabmap::util3d::radiusFiltering(cloud, noiseRadius,
                                                     noiseMinNeighbors);
        }
      } else {
        cloudI = rtabmap::util3d::laserScanToPointCloudI(scan,
                                                         scan.localTransform());
        if (noiseRadius > 0.0f && noiseMinNeighbors > 0) {
          indices = rtabmap::util3d::radiusFiltering(cloudI, noiseRadius,
                                                     noiseMinNeighbors);
        }
      }
    }

    node.sensorData().uncompressData(&rgb, &depth);

    // if (!rgb.empty()) {
    //   cv::imshow("RGB", rgb);
    //   cv::waitKey(10);
    // }

    if (!rgb.empty() && !net_.empty()) {
      int center_x = rgb.cols / 2;
      int center_y = rgb.rows / 2;

      int crop_width = 416;
      int crop_height = 416;

      cv::Mat crop =
          rgb(cv::Rect(center_x - crop_width / 2, center_y - crop_height / 2,
                       crop_width, crop_height));

      cv::Mat resized_image;
      cv::resize(crop, resized_image, cv::Size(crop_width, crop_height));

      cv::Mat blob = cv::dnn::blobFromImage(resized_image, 1 / 255.0,
                                            cv::Size(crop_width, crop_height),
                                            cv::Scalar(0, 0, 0), true, false);
      if (blob.empty()) {
        std::cout << "Failed to create blob" << std::endl;
        return false;
      }
      net_.setInput(blob);
      std::vector<cv::Mat> detections;
      net_.forward(detections);

      std::cout << "detections size: " << detections.size() << std::endl;
      // float confidence_threshold = 0.5;
      // for (size_t i = 0; i < detections.size(); ++i) {
      //   float *data = (float *)detections[i].data;
      //
      //   // Loop through each detection
      //   for (int j = 0; j < detections[i].rows;
      //        ++j, data += detections[i].cols) {
      //     float confidence = data[4]; // Confidence score of the detection
      //     RCLCPP_INFO_STREAM(get_logger(), "confidence: " << confidence);
      //     RCLCPP_INFO_STREAM(
      //       get_logger(), "confidence_threshold: " <<
      //       confidence_threshold);
      //     if (confidence > confidence_threshold) {
      //       int left = (int)(data[0] * resized_image.cols);   // x1
      //       int top = (int)(data[1] * resized_image.rows);    // y1
      //       int right = (int)(data[2] * resized_image.cols);  // x2
      //       int bottom = (int)(data[3] * resized_image.rows); // y2
      //
      //       // Draw bounding box
      //       cv::rectangle(resized_image, cv::Point(left, top),
      //                     cv::Point(right, bottom), cv::Scalar(0, 255, 0),
      //                     2);
      //
      //       // Display confidence text
      //       std::string label = cv::format("%.2f", confidence);
      //       cv::putText(resized_image, label, cv::Point(left, top - 10),
      //                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
      //                   cv::Scalar(255, 255, 255), 2);
      //     }
      //   }
      // }
      //
      // // Display the output image with bounding boxes
      // cv::imshow("Detection", resized_image);
      // cv::waitKey(50);
    }

    // saving images stuff
    if (!rgb.empty()) {
      images_.push_back(rgb);
      // save calibration per image (calibration can change over time, e.g.
      // camera has auto focus)
      camera_models_.push_back(models);
      stereo_models_.push_back(stereoModels);
    }

    float voxelSize = 0.0f;
    float filter_ceiling = std::numeric_limits<float>::max();
    float filter_floor = 0.0f;
    if (voxelSize > 0.0f) {
      if (cloud.get() && !cloud->empty())
        cloud = rtabmap::util3d::voxelize(cloud, indices, voxelSize);
      else if (cloudI.get() && !cloudI->empty())
        cloudI = rtabmap::util3d::voxelize(cloudI, indices, voxelSize);
    }
    if (cloud.get() && !cloud->empty())
      cloud = rtabmap::util3d::transformPointCloud(cloud, iter->second);
    else if (cloudI.get() && !cloudI->empty())
      cloudI = rtabmap::util3d::transformPointCloud(cloudI, iter->second);

    if (filter_ceiling != 0.0 || filter_floor != 0.0f) {
      if (cloud.get() && !cloud->empty()) {
        cloud = rtabmap::util3d::passThrough(
            cloud, "z",
            filter_floor != 0.0f ? filter_floor
                                 : (float)std::numeric_limits<int>::min(),
            filter_ceiling != 0.0f ? filter_ceiling
                                   : (float)std::numeric_limits<int>::max());
      }
      if (cloudI.get() && !cloudI->empty()) {
        cloudI = rtabmap::util3d::passThrough(
            cloudI, "z",
            filter_floor != 0.0f ? filter_floor
                                 : (float)std::numeric_limits<int>::min(),
            filter_ceiling != 0.0f ? filter_ceiling
                                   : (float)std::numeric_limits<int>::max());
      }
    }

    rtabmap::Transform lidarViewpoint =
        iter->second * node.sensorData().laserScanRaw().localTransform();
    rawViewpoints.insert(std::make_pair(iter->first, lidarViewpoint));

    if (cloud.get() && !cloud->empty()) {
      if (assembledCloud->empty()) {
        *assembledCloud = *cloud;
      } else {
        *assembledCloud += *cloud;
      }
      rawViewpointIndices.resize(assembledCloud->size(), iter->first);
    } else if (cloudI.get() && !cloudI->empty()) {
      if (assembledCloudI->empty()) {
        *assembledCloudI = *cloudI;
      } else {
        *assembledCloudI += *cloudI;
      }
      rawViewpointIndices.resize(assembledCloudI->size(), iter->first);
    }

    if (models.empty()) {
      for (size_t i = 0; i < node.sensorData().stereoCameraModels().size();
           ++i) {
        models.push_back(node.sensorData().stereoCameraModels()[i].left());
      }
    }

    robotPoses.insert(std::make_pair(iter->first, iter->second));
    cameraStamps.insert(std::make_pair(iter->first, node.getStamp()));
    if (models.empty() && node.getWeight() == -1 && !cameraModels.empty()) {
      // For intermediate nodes, use latest models
      models = cameraModels.rbegin()->second;
    }
    if (!models.empty()) {
      if (!node.sensorData().imageCompressed().empty()) {
        cameraModels.insert(std::make_pair(iter->first, models));
      }
      if (true) {
        if (cameraPoses.empty()) {
          cameraPoses.resize(models.size());
        }
        UASSERT_MSG(models.size() == cameraPoses.size(),
                    "Not all nodes have same number of cameras to export "
                    "camera poses.");
        for (size_t i = 0; i < models.size(); ++i) {
          cameraPoses[i].insert(std::make_pair(
              iter->first, iter->second * models[i].localTransform()));
        }
      }
    }
    if (!depth.empty() &&
        (depth.type() == CV_16UC1 || depth.type() == CV_32FC1)) {
      cameraDepths.insert(std::make_pair(iter->first, depth));
    }
    if (true && !node.sensorData().laserScanCompressed().empty()) {
      scanPoses.insert(std::make_pair(
          iter->first,
          iter->second *
              node.sensorData().laserScanCompressed().localTransform()));
    }
    std::cout << "Create and assemble the clouds... done (" << timer.ticks()
              << "s, "
              << (!assembledCloud->empty() ? (int)assembledCloud->size()
                                           : (int)assembledCloudI->size())
              << " points)." << std::endl;

    if (imagesExported > 0)
      std::cout << "Exported " << imagesExported << " images" << std::endl;

    rtabmap_cloud_ = assembledCloud;
  }

  pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2);
  pcl::toPCLPointCloud2(*rtabmap_cloud_, *cloud2);

  for (std::map<int, std::vector<rtabmap::CameraModel>>::iterator iter =
           cameraModels.begin();
       iter != cameraModels.end(); ++iter) {
    cv::Mat frame = cv::Mat::zeros(iter->second.front().imageHeight(),
                                   iter->second.front().imageWidth(), CV_8UC3);
    cv::Mat depth(iter->second.front().imageHeight(),
                  iter->second.front().imageWidth() * iter->second.size(),
                  CV_32FC1);
    for (size_t i = 0; i < iter->second.size(); ++i) {
      cv::Mat subDepth = rtabmap::util3d::projectCloudToCamera(
          iter->second.at(i).imageSize(), iter->second.at(i).K(), cloud2,
          robotPoses.at(iter->first) * iter->second.at(i).localTransform());
      subDepth.copyTo(
          depth(cv::Range::all(),
                cv::Range(i * iter->second.front().imageWidth(),
                          (i + 1) * iter->second.front().imageWidth())));
    }

    for (int y = 0; y < depth.rows; ++y) {
      for (int x = 0; x < depth.cols; ++x) {
        if (depth.at<float>(y, x) > 0.0f) // Valid depth
        {
          cv::circle(frame, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        }
      }
    }

    depths_.push_back(frame);

    // cv::imshow("Overlay", frame);
    // cv::waitKey(10); // Press any key to continue

    depth = rtabmap::util2d::cvtDepthFromFloat(depth);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudWithoutNormals(
      new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PointCloud<pcl::PointXYZ>::Ptr rawAssembledCloud(
      new pcl::PointCloud<pcl::PointXYZ>);

  pcl::copyPointCloud(*rtabmap_cloud_, *cloudWithoutNormals);
  rawAssembledCloud = cloudWithoutNormals;

  pcl::PointCloud<pcl::Normal>::Ptr normals =
      rtabmap::util3d::computeNormals(cloudWithoutNormals, 20, 0);

  bool groundNormalsUp = true;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudToExport(
      new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudIToExport(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  if (!assembledCloud->empty()) {
    UASSERT(assembledCloud->size() == normals->size());
    pcl::concatenateFields(*assembledCloud, *normals, *cloudToExport);
    std::cout << "Computing normals of the assembled cloud... done! ("
              << timer.ticks() << "s, " << (int)assembledCloud->size()
              << " points)" << std::endl;
    assembledCloud->clear();

    // adjust with point of views
    std::cout << "Adjust normals to viewpoints of the assembled cloud... ("
              << cloudToExport->size() << " points)" << std::endl;
    rtabmap::util3d::adjustNormalsToViewPoints(rawViewpoints, rawAssembledCloud,
                                               rawViewpointIndices,
                                               cloudToExport, groundNormalsUp);
    std::cout << "adjust normals to viewpoints of the assembled cloud... ("
              << timer.ticks() << "s, " << (int)cloudToExport->size()
              << " points)" << std::endl;
  } else if (!assembledCloudI->empty()) {
    UASSERT(assembledCloudI->size() == normals->size());
    pcl::concatenateFields(*assembledCloudI, *normals, *cloudIToExport);
    std::cout << "Computing normals of the assembled cloud... done! ("
              << timer.ticks() << "s, " << (int)assembledCloudI->size()
              << " points)" << std::endl;
    assembledCloudI->clear();

    // adjust with point of views
    std::cout << "Adjust normals to viewpoints of the assembled cloud... ("
              << cloudIToExport->size() << " points)" << std::endl;
    rtabmap::util3d::adjustNormalsToViewPoints(rawViewpoints, rawAssembledCloud,
                                               rawViewpointIndices,
                                               cloudIToExport, groundNormalsUp);
    std::cout << "Adjust normals to viewpoints of the assembled cloud... ("
              << timer.ticks() << "s, " << (int)cloudIToExport->size()
              << " points)" << std::endl;
  }

  std::vector<std::pair<std::pair<int, int>, pcl::PointXY>> pointToPixel;
  float textureRange = 0.0f;
  float textureAngle = 0.0f;
  std::vector<float> textureRoiRatios;
  cv::Mat projMask;
  bool distanceToCamPolicy = false;
  const rtabmap::ProgressState progressState;
  pointToPixel = rtabmap::util3d::projectCloudToCameras(
      *cloudToExport, robotPoses, cameraModels, textureRange, textureAngle,
      textureRoiRatios, projMask, distanceToCamPolicy, &progressState);

  // color the cloud
  std::vector<int> pointToCamId;
  std::vector<float> pointToCamIntensity;
  pointToCamId.resize(!cloudToExport->empty() ? cloudToExport->size()
                                              : cloudIToExport->size());

  UASSERT(pointToPixel.empty() || pointToPixel.size() == pointToCamId.size());
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr assembledCloudValidPoints(
      new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  assembledCloudValidPoints->resize(pointToCamId.size());

  int imagesDone = 1;
  for (std::map<int, rtabmap::Transform>::iterator iter = robotPoses.begin();
       iter != robotPoses.end(); ++iter) {
    int nodeID = iter->first;
    cv::Mat image;
    if (uContains(nodes, nodeID) &&
        !nodes.at(nodeID).sensorData().imageCompressed().empty()) {
      nodes.at(nodeID).sensorData().uncompressDataConst(&image, 0);
    }
    if (!image.empty()) {
      UASSERT(cameraModels.find(nodeID) != cameraModels.end());
      int modelsSize = cameraModels.at(nodeID).size();
      for (size_t i = 0; i < pointToPixel.size(); ++i) {
        int cameraIndex = pointToPixel[i].first.second;
        if (nodeID == pointToPixel[i].first.first && cameraIndex >= 0) {
          pcl::PointXYZRGBNormal pt;
          float intensity = 0;
          if (!cloudToExport->empty()) {
            pt = cloudToExport->at(i);
          } else if (!cloudIToExport->empty()) {
            pt.x = cloudIToExport->at(i).x;
            pt.y = cloudIToExport->at(i).y;
            pt.z = cloudIToExport->at(i).z;
            pt.normal_x = cloudIToExport->at(i).normal_x;
            pt.normal_y = cloudIToExport->at(i).normal_y;
            pt.normal_z = cloudIToExport->at(i).normal_z;
            intensity = cloudIToExport->at(i).intensity;
          }

          int subImageWidth = image.cols / modelsSize;
          cv::Mat subImage = image(
              cv::Range::all(), cv::Range(cameraIndex * subImageWidth,
                                          (cameraIndex + 1) * subImageWidth));

          int x = pointToPixel[i].second.x * (float)subImage.cols;
          int y = pointToPixel[i].second.y * (float)subImage.rows;
          UASSERT(x >= 0 && x < subImage.cols);
          UASSERT(y >= 0 && y < subImage.rows);

          if (subImage.type() == CV_8UC3) {
            cv::Vec3b bgr = subImage.at<cv::Vec3b>(y, x);
            pt.b = bgr[0];
            pt.g = bgr[1];
            pt.r = bgr[2];
          } else {
            UASSERT(subImage.type() == CV_8UC1);
            pt.r = pt.g = pt.b = subImage.at<unsigned char>(
                pointToPixel[i].second.y * subImage.rows,
                pointToPixel[i].second.x * subImage.cols);
          }

          int exportedId = nodeID;
          pointToCamId[i] = exportedId;
          if (!pointToCamIntensity.empty()) {
            pointToCamIntensity[i] = intensity;
          }
          assembledCloudValidPoints->at(i) = pt;
        }
      }
    }
    std::cout << "Processed " << imagesDone++ << "/"
              << static_cast<int>(robotPoses.size()) << " images\n";
  }

  pcl::IndicesPtr validIndices(new std::vector<int>(pointToPixel.size()));
  size_t oi = 0;
  for (size_t i = 0; i < pointToPixel.size(); ++i) {
    if (pointToPixel[i].first.first <= 0) {
      pcl::PointXYZRGBNormal pt;
      float intensity = 0;
      if (!cloudToExport->empty()) {
        pt = cloudToExport->at(i);
      } else if (!cloudIToExport->empty()) {
        pt.x = cloudIToExport->at(i).x;
        pt.y = cloudIToExport->at(i).y;
        pt.z = cloudIToExport->at(i).z;
        pt.normal_x = cloudIToExport->at(i).normal_x;
        pt.normal_y = cloudIToExport->at(i).normal_y;
        pt.normal_z = cloudIToExport->at(i).normal_z;
        intensity = cloudIToExport->at(i).intensity;
      }

      pointToCamId[i] = 0; // invalid
      pt.b = 0;
      pt.g = 0;
      pt.r = 255;
      if (!pointToCamIntensity.empty()) {
        pointToCamIntensity[i] = intensity;
      }
      assembledCloudValidPoints->at(i) = pt; // red
      validIndices->at(oi++) = i;
    } else {
      validIndices->at(oi++) = i;
    }
  }

  if (oi != validIndices->size()) {
    validIndices->resize(oi);
    assembledCloudValidPoints = rtabmap::util3d::extractIndices(
        assembledCloudValidPoints, validIndices, false, false);
    std::vector<int> pointToCamIdTmp(validIndices->size());
    std::vector<float> pointToCamIntensityTmp(validIndices->size());
    for (size_t i = 0; i < validIndices->size(); ++i) {
      pointToCamIdTmp[i] = pointToCamId[validIndices->at(i)];
      pointToCamIntensityTmp[i] = pointToCamIntensity[validIndices->at(i)];
    }
    pointToCamId = pointToCamIdTmp;
    pointToCamIntensity = pointToCamIntensityTmp;
    pointToCamIdTmp.clear();
    pointToCamIntensityTmp.clear();
  }

  cloudToExport = assembledCloudValidPoints;
  cloudIToExport->clear();

  pcl::copyPointCloud(*cloudToExport, *rtabmap_cloud_);

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  return true;
}

int main(int argc, char *argv[]) {
  try {
    py::scoped_interpreter guard{};
    py::module yolov8 = py::module::import("ultralytics");
    py::object YOLO = yolov8.attr("YOLO");
    py::object net = YOLO("yolov8m.pt");

    std::string rtabmap_database_name;
    std::string model_name;
    if (argc == 1) {
      return 1;
    } else if (argc == 2) {
      rtabmap_database_name = argv[1];
      model_name = "";
    } else if (argc == 3) {
      rtabmap_database_name = argv[1];
      model_name = argv[2];
    } else {
      return 1;
    }

    DatabaseExporter extractor(rtabmap_database_name, model_name);
    extractor.load_rtabmap_db();

    extractor.get_detections(net);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}

#include "database_exporter.hpp"
#include <opencv2/core/vsx_utils.hpp>

// I relied on this code heavily to write this
// https://github.com/introlab/rtabmap/blob/ff61266430017eb4924605b832cd688c8739af18/tools/Export/main.cpp#L1104-L1115

DatabaseExporter::DatabaseExporter(std::string rtabmap_database_name,
                                   std::string model_name)
  : timestamp_(generate_timestamp_string())
{
  // check if database name is empty
  if (rtabmap_database_name.empty()) {
    std::cout << "RTABMap database name is empty" << std::endl;
    return;
  }

  rtabmap_database_path_ =
    std::string(PROJECT_PATH) + "/databases/" + rtabmap_database_name;

  // initialize shared pointers for the cloud and occupancy grid
  rtabmap_cloud_ = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);

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
  if (!std::filesystem::create_directory(path + "/camera_models")) {
    std::cout << "Failed to create camera_models directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/stereo_models")) {
    std::cout << "Failed to create stereo_models directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/objects")) {
    std::cout << "Failed to create objects directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/landmarks")) {
    std::cout << "Failed to create landmarks directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/detections")) {
    std::cout << "Failed to create detections directory" << std::endl;
    return;
  }
  if (!std::filesystem::create_directory(path + "/meshes")) {
    std::cout << "Failed to create meshes directory" << std::endl;
    return;
  }
}

DatabaseExporter::~DatabaseExporter()
{
  // create the output directory
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;

  // save the point cloud
  std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
  pcl::io::savePCDFileBinary(cloud_path, *rtabmap_cloud_);

  // save the occupancy grid
  std::string grid_path = path + "/grid/" + timestamp_;
  std::cout << "occupancy grid size: " << rtabmap_cloud_->points.size()
            << std::endl;
  rtabmap_occupancy_grid_ = point_cloud_to_occupancy_grid(rtabmap_cloud_);
  nav2_map_server::SaveParameters save_params;
  save_params.map_file_name = grid_path;
  save_params.image_format = "pgm";
  save_params.free_thresh = 0.196;
  save_params.occupied_thresh = 0.65;
  nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, save_params);
  int rgbImagesExported = 0;
  int depthImagesExported = 0;
  int rgbDepthImagesExported = 0;
  for (const auto &data : mapping_data_) {
    // save rgb images
    std::string image_path =
      path + "/images/" + std::to_string(rgbImagesExported) + ".jpg";
    cv::imwrite(image_path, std::get<0>(data));
    ++rgbImagesExported;

    // save depth images
    cv::Mat depthExported = std::get<1>(data);
    std::string depth_path =
      path + "/depths/" + std::to_string(depthImagesExported) + ".jpg";

    cv::imwrite(depth_path, depthExported);
    ++depthImagesExported;

    // save rgb depth images
    cv::Mat rgb_depth = rgb_depths_.at(rgbDepthImagesExported);
    std::string rgb_depth_path = path + "/depths/" + "rgb_depth" +
                                 std::to_string(rgbDepthImagesExported) +
                                 ".jpg";
    cv::imwrite(rgb_depth_path, rgb_depth);
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
  std::cout << "RGB Images exported: " << rgbImagesExported << std::endl;
  std::cout << "Depth Images exported: " << depthImagesExported << std::endl;
}

cv::Mat DatabaseExporter::numpy_to_mat(const py::array_t<uint8_t> &np_array)
{
  py::buffer_info buf = np_array.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (uchar *)buf.ptr);
  return mat;
}

py::array DatabaseExporter::mat_to_numpy(const cv::Mat &mat)
{
  return py::array_t<uint8_t>({mat.rows, mat.cols, mat.channels()},
                              {mat.step[0], mat.step[1], sizeof(uint8_t)},
                              mat.data);
}

nav_msgs::msg::OccupancyGrid::SharedPtr
DatabaseExporter::point_cloud_to_occupancy_grid(
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
  // calculate the centroid
  Eigen::Matrix<float, 4, 1> centroid;
  pcl::ConstCloudIterator<pcl::PointXYZRGBNormal> cloud_iterator(*cloud);
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

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
DatabaseExporter::filter_point_cloud(
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
  // statistical outlier removal
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sor_cloud(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(50); // increase for more permissive, decrease for less
  sor.setStddevMulThresh(
    1.0); // increase for more permissive, decrease for less
  sor.filter(*sor_cloud);

  // radius outlier removal
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr radius_cloud(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> radius_outlier;
  radius_outlier.setInputCloud(sor_cloud);
  radius_outlier.setRadiusSearch(
    0.2); // adjust based on spacing in the point cloud
  radius_outlier.setMinNeighborsInRadius(
    3); // increase for more aggressive outlier removal
  radius_outlier.filter(*radius_cloud);
  radius_cloud->width = radius_cloud->points.size();

  // find the lowest point in the pointcloud
  auto min_point_iter = std::min_element(
    radius_cloud->points.begin(), radius_cloud->points.end(),
    [](const pcl::PointXYZRGBNormal &lhs, const pcl::PointXYZRGBNormal &rhs) {
      return lhs.z < rhs.z;
    });

  // passthrough filter
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pass_cloud(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
  // std::cout << "Min point: " << min_point_iter->z << std::endl;
  pass.setInputCloud(radius_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_point_iter->z + 0.4,
                       FLT_MAX); // adjust based on the scene
  pass.filter(*pass_cloud);
  pass_cloud->width = pass_cloud->points.size();

  // radius outlier removal
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr radius2_cloud(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> radius2_outlier;
  radius2_outlier.setInputCloud(pass_cloud);
  radius2_outlier.setRadiusSearch(
    0.2); // adjust based on spacing in the point cloud
  radius2_outlier.setMinNeighborsInRadius(
    3); // increase for more aggressive outlier removal
  radius2_outlier.filter(*radius2_cloud);
  radius2_cloud->width = radius2_cloud->points.size();

  return radius2_cloud;
}

std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
DatabaseExporter::project_cloud_to_camera(
  const cv::Size &image_size, const cv::Mat &camera_matrix,
  const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
  const rtabmap::Transform &camera_transform)
{
  assert(!camera_transform.isNull());
  assert(!cloud->empty());
  assert(camera_matrix.type() == CV_64FC1 && camera_matrix.cols == 3 &&
         camera_matrix.cols == 3);

  float fx = camera_matrix.at<double>(0, 0);
  float fy = camera_matrix.at<double>(1, 1);
  float cx = camera_matrix.at<double>(0, 2);
  float cy = camera_matrix.at<double>(1, 2);

  cv::Mat registered = cv::Mat::zeros(image_size, CV_32FC1);
  rtabmap::Transform t = camera_transform.inverse();

  // create a map from each pixel coordinate to the index of their point in the
  // pointcloud
  std::map<std::pair<int, int>, int> pixel_to_point_map;

  int count = 0;
  for (pcl::PointCloud<pcl::PointXYZRGBNormal>::const_iterator it =
         cloud->begin();
       it != cloud->end(); ++it) {
    pcl::PointXYZRGBNormal ptScan = *it;

    // transform point from world frame to camera frame
    ptScan = rtabmap::util3d::transformPoint(ptScan, t);

    // re-project in camera frame
    float z = ptScan.z;
    bool set = false;
    if (z > 0.0f) {
      float invZ = 1.0f / z;
      float dx = (fx * ptScan.x) * invZ + cx;
      float dy = (fy * ptScan.y) * invZ + cy;
      int dx_low = dx;
      int dy_low = dy;
      int dx_high = dx + 0.5f;
      int dy_high = dy + 0.5f;
      if (uIsInBounds(dx_low, 0, registered.cols) &&
          uIsInBounds(dy_low, 0, registered.rows)) {
        set = true;
        float &zReg = registered.at<float>(dy_low, dx_low);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if ((dx_low != dx_high || dy_low != dy_high) &&
          uIsInBounds(dx_high, 0, registered.cols) &&
          uIsInBounds(dy_high, 0, registered.rows)) {
        set = true;
        float &zReg = registered.at<float>(dy_high, dx_high);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if (set) {
        pixel_to_point_map[{dy_low, dx_low}] = count;
      }
    }
    count++;
  }

  return {registered, pixel_to_point_map};
}

std::string DatabaseExporter::generate_timestamp_string()
{
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);

  std::ostringstream oss;

  oss << std::put_time(ptm, "%Y-%m-%d_%H-%M-%S");

  return oss.str();
}

Result DatabaseExporter::load_rtabmap_db()
{
  Result result;
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  if (driver->openConnection(rtabmap_database_path_)) {
    parameters = driver->getLastParameters();
    driver->closeConnection(false);
  } else {
    std::cout << "Failed to open database" << std::endl;
    return result;
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
    return result;
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

  std::map<int, cv::Mat> rgb_images;
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
      int decimation = 1;
      int maxRange = 100.0;
      int minRange = 0.0;
      float noiseRadius = 0.0f;
      int noiseMinNeighbors = 0;
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
      // if (decimation > 1 || minRange > 0.0f || maxRange) {
      scan =
        rtabmap::util3d::commonFiltering(scan, decimation, minRange, maxRange);
      // }
      if (scan.hasRGB()) {
        cloud = rtabmap::util3d::laserScanToPointCloudRGB(
          scan, scan.localTransform());
        if (noiseRadius > 0.0f && noiseMinNeighbors > 0) {
          indices = rtabmap::util3d::radiusFiltering(cloud, noiseRadius,
                                                     noiseMinNeighbors);
        }
      } else {
        cloudI =
          rtabmap::util3d::laserScanToPointCloudI(scan, scan.localTransform());
        if (noiseRadius > 0.0f && noiseMinNeighbors > 0) {
          indices = rtabmap::util3d::radiusFiltering(cloudI, noiseRadius,
                                                     noiseMinNeighbors);
        }
      }
    }

    node.sensorData().uncompressData(&rgb, &depth);

    // saving images stuff
    if (!rgb.empty()) {
      rgb_images[iter->first] = rgb;
      // save calibration per image
      // calibration can change over time, e.g. camera has auto focus
      if (!models.empty()) {
        camera_models_.push_back(models);
      }
      if (!stereoModels.empty()) {
        stereo_models_.push_back(stereoModels);
      }
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
        // Not all nodes have same number of cameras to export camera poses.
        assert(models.size() == cameraPoses.size());
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
  }

  pcl::copyPointCloud(*assembledCloud, *rtabmap_cloud_);

  // extract the camera poses
  std::map<int, rtabmap::Transform> optimized_poses;

  for (std::map<int, rtabmap::Transform>::iterator iter =
         optimizedPoses.lower_bound(1);
       iter != optimizedPoses.end(); ++iter) {
    optimized_poses[iter->first] = iter->second;
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
    assert(assembledCloud->size() == normals->size());
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
    assert(assembledCloudI->size() == normals->size());
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
  std::cout << "Project the assembled cloud to cameras... ("
            << cloudToExport->size() << " points)" << std::endl;
  pointToPixel = rtabmap::util3d::projectCloudToCameras(
    *cloudToExport, robotPoses, cameraModels, textureRange, textureAngle,
    textureRoiRatios, projMask, distanceToCamPolicy, &progressState);
  std::cout << "Project the assembled cloud to cameras... done! ("
            << timer.ticks() << "s)" << std::endl;

  std::vector<int> pointToCamId;
  std::vector<float> pointToCamIntensity;
  pointToCamId.resize(!cloudToExport->empty() ? cloudToExport->size()
                                              : cloudIToExport->size());

  std::cout << "Assemble the cloud with colors from images... ("
            << cloudToExport->size() << " points)" << std::endl;
  // assert(pointToPixel.empty() || pointToPixel.size() == pointToCamId.size());
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr assembledCloudValidPoints(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  assembledCloudValidPoints->resize(pointToCamId.size());
  std::cout << "Assemble the cloud with colors from images... done! ("
            << timer.ticks() << "s)" << std::endl;

  // Figure out what color each point in the pointcloud should be based on pixel
  // color
  int imagesDone = 1;
  // for (std::map<int, rtabmap::Transform>::iterator iter = robotPoses.begin();
  //     iter != robotPoses.end(); ++iter) {
  //   std::cout
  // }
  for (std::map<int, rtabmap::Transform>::iterator iter = robotPoses.begin();
       iter != robotPoses.end(); ++iter) {
    std::cout << "Processing image " << imagesDone << "/" << robotPoses.size()
              << std::endl;
    int nodeID = iter->first;
    cv::Mat image;
    if (uContains(nodes, nodeID) &&
        !nodes.at(nodeID).sensorData().imageCompressed().empty()) {
      nodes.at(nodeID).sensorData().uncompressDataConst(&image, 0);
    }
    if (!image.empty()) {
      // assert(cameraModels.find(nodeID) != cameraModels.end());
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
          // assert(x >= 0 && x < subImage.cols);
          // assert(y >= 0 && y < subImage.rows);

          if (subImage.type() == CV_8UC3) {
            cv::Vec3b bgr = subImage.at<cv::Vec3b>(y, x);
            pt.b = bgr[0];
            pt.g = bgr[1];
            pt.r = bgr[2];
          } else {
            // assert(subImage.type() == CV_8UC1);
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

  std::cout << "getting valid indices" << std::endl;
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

  if (!cloudIToExport->empty()) {
    pcl::copyPointCloud(*cloudIToExport, *cloudToExport);
    cloudIToExport->clear();
  }

  pcl::copyPointCloud(*cloudToExport, *rtabmap_cloud_);
  rtabmap_cloud_ = filter_point_cloud(rtabmap_cloud_);

  for (std::map<int, std::vector<rtabmap::CameraModel>>::iterator iter =
         cameraModels.begin();
       iter != cameraModels.end(); ++iter) {
    cv::Mat frame = cv::Mat::zeros(iter->second.front().imageHeight(),
                                   iter->second.front().imageWidth(), CV_8UC3);
    cv::Mat depth(iter->second.front().imageHeight(),
                  iter->second.front().imageWidth() * iter->second.size(),
                  CV_32FC1);
    cv::Mat rgb_frame = rgb_images[iter->first];
    cv::Mat rgb_depth = rgb_frame.clone();
    std::pair<cv::Mat, std::map<std::pair<int, int>, int>> depth_map;
    for (size_t i = 0; i < iter->second.size(); ++i) {
      depth_map = project_cloud_to_camera(
        iter->second.at(i).imageSize(), iter->second.at(i).K(), rtabmap_cloud_,
        robotPoses.at(iter->first) * iter->second.at(i).localTransform());
      depth_map.first.copyTo(
        depth(cv::Range::all(),
              cv::Range(i * iter->second.front().imageWidth(),
                        (i + 1) * iter->second.front().imageWidth())));
      // cv::imshow("depth1", depth);
      // depth = rtabmap::util2d::cvtDepthFromFloat(depth);
      // cv::imshow("depth2", depth);
      for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
          if (depth.at<float>(y, x) > 0.0f) // Valid depth
          {
            cv::Vec3b color = rgb_frame.at<cv::Vec3b>(y, x);
            cv::Scalar circle_color = cv::Scalar(color[0], color[1], color[2]);
            cv::circle(frame, cv::Point(x, y), 3, circle_color, -1);
            cv::circle(rgb_depth, cv::Point(x, y), 3, cv::Scalar(0, 255, 0),
                       -1);
            cv::addWeighted(rgb_depth, 0.5, rgb_depth, 1 - 0.5, 0, rgb_depth);
            rgb_depths_.push_back(rgb_depth);
          }
        }
      }
      mapping_data_.push_back(
        {rgb_frame, frame, robotPoses.at(iter->first), depth_map.second});
    }
  }

  int poissonDepth = 0;
  float poissonSize = 0.03;
  int maxPolygons = 300000;
  float colorRadius = -1.0f;
  bool doClean = true;
  bool texture = true;
  int minCluster = 200;
  bool saveInDb = false;
  bool binary = true;
  float textureDepthError = 0;
  int laplacianThr = 0;
  bool multiband = false;
  int textureCount = 1;
  int textureSize = 8192;
  float gainValue = 1.0f;
  bool doGainCompensationRGB = true;
  bool doBlending = true;
  int lowBrightnessGain = 0;
  int highBrightnessGain = 10;
  int multibandDownScale = 2;
  std::string multibandNbContrib = "1 5 10 0";
  int multibandUnwrap = 0;
  bool multibandFillHoles = false;
  int multibandPadding = 5;
  double multibandBestScoreThr = 0.1;
  double multibandAngleHardthr = 90;
  bool multibandForceVisible = false;

  std::string outputDirectory = ".";

  std::cout << "Exporting the mesh... ("
            << (!cloudToExport->empty() ? (int)cloudToExport->size()
                                        : (int)cloudIToExport->size())
            << " points)" << std::endl;
  Eigen::Vector4f min, max;
  pcl::getMinMax3D(*cloudToExport, min, max);
  float mapLength = uMax3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  int optimizedDepth = 12;
  for (int i = 6; i < 12; ++i) {
    if (mapLength / float(1 << i) < poissonSize) {
      optimizedDepth = i;
      break;
    }
  }
  if (poissonDepth > 0) {
    optimizedDepth = poissonDepth;
  }

  // Mesh reconstruction
  std::cout << "Mesh reconstruction... depth=" << optimizedDepth << std::endl;
  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
  pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
  poisson.setDepth(optimizedDepth);
  poisson.setInputCloud(cloudToExport);
  poisson.reconstruct(*mesh);
  std::cout << "Mesh reconstruction... done (" << timer.ticks() << "s, "
            << (int)mesh->polygons.size() << " polygons)." << std::endl;

  if (mesh->polygons.size()) {
    std::cout << "Mesh color transfer (max polygons=" << maxPolygons
              << ", color radius=" << colorRadius << ", clean=" << doClean
              << ")..." << std::endl;
    rtabmap::util3d::denseMeshPostProcessing<pcl::PointXYZRGBNormal>(
      mesh, 0.0f, maxPolygons, cloudToExport, colorRadius, !texture, doClean,
      minCluster);
    std::cout << "Mesh color transfer... done (" << timer.ticks() << "s)."
              << std::endl;

    if (!texture) {
      std::string output_path = std::string(PROJECT_PATH) + "/output/" +
                                timestamp_ + "/meshes/" + timestamp_ + ".ply";
      std::cout << "Saving " << output_path << "..." << std::endl;
      pcl::io::savePLYFileBinary(output_path, *mesh);
    } else {
      // Camera filtering for texturing
      std::map<int, rtabmap::Transform> robotPosesFiltered;
      if (laplacianThr > 0) {
        std::cout << "Filtering " << robotPoses.size()
                  << " images from texturing..." << std::endl;
        for (std::map<int, rtabmap::Transform>::iterator iter =
               robotPoses.begin();
             iter != robotPoses.end(); ++iter) {
          assert(nodes.find(iter->first) != nodes.end());
          cv::Mat img;
          nodes.find(iter->first)
            ->second.sensorData()
            .uncompressDataConst(&img, 0);
          if (!img.empty()) {
            cv::Mat imgLaplacian;
            cv::Laplacian(img, imgLaplacian, CV_16S);
            cv::Mat m, s;
            cv::meanStdDev(imgLaplacian, m, s);
            double stddev_pxl = s.at<double>(0);
            double var = stddev_pxl * stddev_pxl;
            if (var < (double)laplacianThr) {
              std::cout << "Camera's image " << iter->first
                        << " is detected as blurry (var=" << var
                        << " < thr=" << laplacianThr
                        << "), camera is ignored for texturing." << std::endl;
            } else {
              robotPosesFiltered.insert(*iter);
            }
          }
        }
        std::cout << "Filtered " << robotPosesFiltered.size() << "/"
                  << robotPoses.size() << " images from texturing" << std::endl;
      } else {
        robotPosesFiltered = robotPoses;
      }

      std::cout << "Texturing " << mesh->polygons.size()
                << " polygons... robotPoses=" << robotPosesFiltered.size()
                << ", cameraModels=" << cameraModels.size()
                << ", cameraDepths=" << cameraDepths.size() << std::endl;
      std::vector<std::map<int, pcl::PointXY>> vertexToPixels;
      pcl::TextureMeshPtr textureMesh = rtabmap::util3d::createTextureMesh(
        mesh, robotPosesFiltered, cameraModels, cameraDepths, textureRange,
        textureDepthError, textureAngle,
        multiband
          ? 0
          : 50, // Min polygons in camera view to be textured by this camera
        textureRoiRatios, &progressState, &vertexToPixels, distanceToCamPolicy);
      std::cout << "Texturing... done (" << timer.ticks() << "s)." << std::endl;

      // Remove occluded polygons (polygons with no texture)
      if (doClean && textureMesh->tex_coordinates.size()) {
        std::cout << "Cleanup mesh..." << std::endl;
        rtabmap::util3d::cleanTextureMesh(
          *textureMesh, 100); // Min polygons in a cluster to keep them
        std::cout << "Cleanup mesh... done (" << timer.ticks() << "s)."
                  << std::endl;
      }

      if (textureMesh->tex_materials.size()) {
        std::cout << "Merging " << textureMesh->tex_materials.size()
                  << " texture(s)... (" << textureCount << " max textures)"
                  << std::endl;
        std::map<int, std::map<int, cv::Vec4d>> gains;
        std::map<int, std::map<int, cv::Mat>> blendingGains;
        std::pair<float, float> contrastValues(0, 0);
        cv::Mat textures = rtabmap::util3d::mergeTextures(
          *textureMesh, std::map<int, cv::Mat>(),
          std::map<int, std::vector<rtabmap::CameraModel>>(),
          rtabmap.getMemory(), 0, textureSize, textureCount, vertexToPixels,
          gainValue > 0.0f, gainValue, doGainCompensationRGB, doBlending, 0,
          lowBrightnessGain,
          highBrightnessGain, // low-high brightness/contrast balance
          false,              // exposure fusion
          0,                  // state
          0,                  // blank value (0=black)
          &gains, &blendingGains, &contrastValues);
        std::cout << "Merging to " << textureMesh->tex_materials.size()
                  << " texture(s)... done (" << timer.ticks() << "s)."
                  << std::endl;

        // TextureMesh OBJ
        bool success = false;
        assert(!textures.empty());
        for (size_t i = 0; i < textureMesh->tex_materials.size(); ++i) {
          textureMesh->tex_materials[i].tex_file += ".jpg";
          std::cout << "Saving texture to "
                    << textureMesh->tex_materials[i].tex_file << std::endl;
          assert(textures.cols % textures.rows == 0);
          outputDirectory =
            std::string(PROJECT_PATH) + "/output/" + timestamp_ + "/meshes/";
          success = cv::imwrite(
            outputDirectory + textureMesh->tex_materials[i].tex_file,
            cv::Mat(textures, cv::Range::all(),
                    cv::Range(textures.rows * i, textures.rows * (i + 1))));
          if (!success) {
            UERROR("Failed saving %s!",
                   textureMesh->tex_materials[i].tex_file.c_str());
          } else {
            std::cout << "Saved " << textureMesh->tex_materials[i].tex_file
                      << std::endl;
          }
        }
        if (success) {

          std::string outputPath = std::string(PROJECT_PATH) + "/output/" +
                                   timestamp_ + "/meshes/" + timestamp_ +
                                   "_mesh.obj";
          std::cout << "Saving obj ("
                    << textureMesh->cloud.data.size() /
                         textureMesh->cloud.point_step
                    << " vertices) to " << outputPath << std::endl;
          success = pcl::io::saveOBJFile(outputPath, *textureMesh) == 0;

          if (success) {
            std::cout << "Saved obj to " << outputPath << "!" << std::endl;
          } else {
            UERROR("Failed saving obj to %s!", outputPath.c_str());
          }
        }
        // timer.restart();
        std::string outputPath = std::string(PROJECT_PATH) + "/output/" +
                                 timestamp_ + "/meshes/" + timestamp_ +
                                 "_mesh_multiband.obj";
        std::cout << "MultiBand texturing (size=" << textureSize << std::endl;
        std::cout << "Downscale: " << multibandDownScale << std::endl;
        std::cout << "Unwrap method: "
                  << (multibandUnwrap == 1   ? "ABF"
                      : multibandUnwrap == 2 ? "LSCM"
                                             : "Basic")
                  << std::endl;
        std::cout << "Fill holes: " << multibandFillHoles << std::endl;
        std::cout << "Padding: " << multibandPadding << std::endl;
        std::cout << "Best score threshold: " << multibandBestScoreThr
                  << std::endl;
        std::cout << "Angle threshold: " << multibandAngleHardthr << std::endl;
        std::cout << "Force visible: " << multibandForceVisible << std::endl;
        std::cout << "Saving to " << outputPath << std::endl;

        if (rtabmap::util3d::multiBandTexturing(
              outputPath, textureMesh->cloud, textureMesh->tex_polygons[0],
              robotPosesFiltered, vertexToPixels, std::map<int, cv::Mat>(),
              std::map<int, std::vector<rtabmap::CameraModel>>(),
              rtabmap.getMemory(), 0, textureSize, multibandDownScale,
              multibandNbContrib, "jpg", gains, blendingGains, contrastValues,
              doGainCompensationRGB, multibandUnwrap, multibandFillHoles,
              multibandPadding, multibandBestScoreThr, multibandAngleHardthr,
              multibandForceVisible)) {
          std::cout << "MultiBand texturing...done (" << timer.ticks() << "s)."
                    << std::endl;
        } else {
          std::cout << "MultiBand texturing...failed! (" << timer.ticks()
                    << "s)" << std::endl;
        }
      }
    }
  }

  result.success = true;
  result.timestamp = timestamp_;
  result.cloud = rtabmap_cloud_;
  result.mapping_data = mapping_data_;

  std::cout << "Finished loading database" << std::endl;
  std::cout << "Number of images: " << mapping_data_.size() << std::endl;
  std::cout << "Number of points in cloud: " << rtabmap_cloud_->points.size()
            << std::endl;
  std::cout << "Timestamp: " << timestamp_ << std::endl;

  return result;
}

pcl::PointXYZ DatabaseExporter::calculate_centroid(
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
  pcl::PointXYZ centroid;
  for (const auto &point : cloud->points) {
    centroid.x += point.x;
    centroid.y += point.y;
    centroid.z += point.z;
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  cloud->points.resize(cloud->width * cloud->height);

  centroid.x /= cloud->size();
  centroid.y /= cloud->size();
  centroid.z /= cloud->size();

  return centroid;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud_from_bounding_box(
  std::tuple<std::string, float, BoundingBox> bounding_box,
  std::map<std::pair<int, int>, int> pixel_to_point_map,
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, rtabmap::Transform pose)
{
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<> dis;
  gen.seed(rd());

  std::string label = std::get<0>(bounding_box);
  BoundingBox box = std::get<2>(bounding_box);
  int r = dis(gen);
  int g = dis(gen);
  int b = dis(gen);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointXYZRGBNormal closest_point;
  float l2_closest = 0.0;
  // figure out what the closest point in the pointcloud is to the camera
  // and add all points to the object cloud
  for (int y = box.y1; y < box.y2; ++y) {
    for (int x = box.x1; x < box.x2; ++x) {
      if (pixel_to_point_map.find(std::make_pair(y, x)) !=
          pixel_to_point_map.end()) {
        int index = pixel_to_point_map.at(std::make_pair(y, x));
        pcl::PointXYZRGBNormal point = cloud->points.at(index);

        if (object_cloud->empty()) {
          closest_point = point;
          l2_closest = (point.x - pose.x()) * (point.x - pose.x()) +
                       (point.y - pose.y()) * (point.y - pose.y()) +
                       (point.z - pose.z()) * (point.z - pose.z());
        }

        float l2 = (point.x - pose.x()) * (point.x - pose.x()) +
                   (point.y - pose.y()) * (point.y - pose.y()) +
                   (point.z - pose.z()) * (point.z - pose.z());

        if (l2 < l2_closest) {
          closest_point = point;
          l2_closest = l2;
        }

        point.r = r;
        point.g = g;
        point.b = b;
        object_cloud->push_back(point);
      }
    }
  }

  if (object_cloud->empty()) {
    return object_cloud;
  }

  // filter out the points that are too far away from the closest point to
  // the camera
  float threshold = 1.0;
  object_cloud->points.erase(
    std::remove_if(
      object_cloud->points.begin(), object_cloud->points.end(),
      [&closest_point, threshold](const pcl::PointXYZRGBNormal &point) {
        float l2 = std::sqrt(std::pow(point.x - closest_point.x, 2) +
                             std::pow(point.y - closest_point.y, 2) +
                             std::pow(point.z - closest_point.z, 2));
        return l2 > threshold;
      }),
    object_cloud->points.end());

  object_cloud->width = object_cloud->points.size();
  object_cloud->height = 1;

  return object_cloud;
}

std::vector<Object> semantic_mapping(
  py::object &net, DatabaseExporter &exporter,
  std::vector<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                         std::map<std::pair<int, int>, int>>> &mapping_data,
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, std::string &timestamp)
{
  // start by getting the detections, and creating some pretty images
  std::vector<Object> objects; // cloud, closest point, label, confidence
  std::unordered_map<std::string, int> object_counts;
  std::vector<cv::Mat> detection_frames;
  std::vector<cv::Mat> annotated_depth_frames;
  int iter = 0;
  for (const auto &frame : mapping_data) {
    cv::Mat rgb = std::get<0>(frame);
    cv::Mat depth = std::get<1>(frame);
    rtabmap::Transform pose = std::get<2>(frame);
    std::map<std::pair<int, int>, int> pixel_to_point_map = std::get<3>(frame);
    py::array np_array = exporter.mat_to_numpy(rgb);
    py::list detections = net.attr("predict")(np_array);

    std::vector<std::tuple<std::string, float, BoundingBox>> bounding_boxes;
    for (auto detection : detections) {
      py::object boxes = detection.attr("boxes");
      py::object names = detection.attr("names");
      py::object speed = detection.attr("speed");
      if (!boxes.is_none()) {
        auto box_list =
          boxes.attr("xyxy")
            .cast<py::list>(); // Example of accessing box coordinates
        for (size_t i = 0; i < py::len(box_list); ++i) {
          py::object box = box_list[i];
          py::object conf_tensor = boxes.attr("conf");
          if (conf_tensor[py::int_(i)].cast<float>() < 0.9) {
            continue;
          }

          // Extract the box coordinates
          auto numpy_array =
            box_list[py::int_(0)].attr("cpu")().attr("numpy")();

          // Access individual elements using NumPy indexing.
          int x1 = numpy_array[py::int_(0)].cast<int>();
          int y1 = numpy_array[py::int_(1)].cast<int>();
          int x2 = numpy_array[py::int_(2)].cast<int>();
          int y2 = numpy_array[py::int_(3)].cast<int>();

          // std::cout << "Box: " << x1 << ", " << y1 << ", " << x2 << ", " <<
          // y2
          //           << std::endl;

          cv::Mat detection_frame = rgb.clone();
          cv::Mat depth_frame = depth.clone();

          // Draw the box on the image
          cv::rectangle(detection_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2);
          cv::rectangle(depth_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2);

          // Add the label to the image
          py::object names = detection.attr("names");
          py::object classes = boxes.attr("cls");
          std::string label =
            names[py::int_(classes[py::int_(i)])].cast<std::string>();
          cv::putText(detection_frame, label, cv::Point(x1, y1 - 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, label, cv::Point(x1, y1 - 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Add the confidence to the image
          float confidence = conf_tensor[py::int_(i)].cast<float>();
          cv::putText(detection_frame, std::to_string(confidence),
                      cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, std::to_string(confidence),
                      cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 0), 2);

          // add the bounding box with the label to the list for later
          // if (label == "chair" || label == "cup") {
          bounding_boxes.push_back(
            {label, confidence, BoundingBox(x1, y1, x2, y2)});
          // } else {
          //   continue;
          // }

          // Add the speed to the image
          py::object speed_py = detection.attr("speed");
          std::string speed = py::str(speed_py).cast<std::string>();
          cv::putText(detection_frame, speed, cv::Point(x1, y1 - 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, speed, cv::Point(x1, y1 - 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Add the timestamp to the image
          cv::putText(detection_frame, timestamp, cv::Point(x1, y1 - 70),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, timestamp, cv::Point(x1, y1 - 70),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Display the image
          // cv::imshow("Image", detection_frame);
          // cv::imshow("Depth", depth);
          // cv::waitKey(1);

          detection_frames.push_back(detection_frame);
          annotated_depth_frames.push_back(depth_frame);
        }
      }
    }
    // cv::destroyAllWindows();

    std::vector<Object> new_objects;
    std::vector<cv::Mat> images;
    for (const auto &elem : bounding_boxes) {
      std::string label = std::get<0>(elem);
      float conf = std::get<1>(elem);
      auto detection = detection_frames.at(iter);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud =
        object_cloud_from_bounding_box(elem, pixel_to_point_map, cloud, pose);

      // std::cout << "object cloud size: " << object_cloud->size() <<
      // std::endl; std::cout << "Detected " << label << " with confidence " <<
      // conf
      //           << std::endl;
      // cv::imshow("Detection", detection_frames.at(iter));
      // cv::imshow("depth", depth);
      // cv::waitKey(0);
      // calculate the centroid of the object's pointcloud
      pcl::PointXYZ centroid = exporter.calculate_centroid(object_cloud);

      // figure out if we've already seen this object before, and if we have
      // then add its points to the pointcloud, recalculate the centroid, and
      // update the label and confidence with the highest confidence value.
      if (objects.empty()) {
        objects.push_back({object_cloud, centroid, label, conf});
      } else {
        bool found = false;
        for (auto &elem : objects) {
          float l2 = std::sqrt(std::pow((elem.centroid.x - centroid.x), 2) +
                               std::pow((elem.centroid.y - centroid.y), 2) +
                               std::pow((elem.centroid.z - centroid.z), 2));
          if (l2 < 0.5) {
            elem.label = conf > elem.confidence ? label : elem.label;
            elem.confidence = conf > elem.confidence ? conf : elem.confidence;
            found = true;
          }
        }
        if (!found) {
          objects.push_back({object_cloud, centroid, label, conf});
          images.push_back(detection);
          new_objects.push_back({object_cloud, centroid, label, conf});
          object_counts[label]++;
        }
      }
      iter++;
    }
    std::string file_path = std::string(PROJECT_PATH) + "/output/" + timestamp +
                            "/landmarks/" + timestamp + ".yaml";
    YAML::Node node;

    for (const auto &object : new_objects) {

      if (object.cloud->size() < 5) {
        std::cout << "Object " << object.label
                  << " has less than 5 points, skipping. Size: "
                  << object.cloud->size() << std::endl;
        std::cout << "centroid: " << object.centroid.x << ", "
                  << object.centroid.y << ", " << object.centroid.z
                  << std::endl;
        continue;
      }
      std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp +
                         "/objects/" + object.label +
                         std::to_string(object_counts[object.label]) + ".pcd";
      pcl::io::savePCDFileBinary(path, *object.cloud);

      if (std::filesystem::exists(file_path)) {
        node = YAML::LoadFile(file_path);
      }

      // Save the new landmark's name and coordinates
      node[object.label + std::to_string(object_counts[object.label])]["x"] =
        object.centroid.x;
      node[object.label + std::to_string(object_counts[object.label])]["y"] =
        object.centroid.y;

      // Open the file for writing
      std::ofstream file(file_path);

      // Write to the file and close it
      file << node;
      file.close();
    }
  }
  iter = 0;
  std::string path =
    std::string(PROJECT_PATH) + "/output/" + timestamp + "/detections/";
  std::cout << "saving images to: " << path << std::endl;
  for (const auto &frame : detection_frames) {
    cv::Mat detection_frame = frame;

    // save the image
    if (std::filesystem::exists(path)) {
      cv::imwrite(path + "detection" + std::to_string(iter) + ".png",
                  detection_frame);
    }
    iter++;
  }
  iter = 0;
  for (const auto &frame : annotated_depth_frames) {
    // save the image
    if (std::filesystem::exists(path)) {
      cv::imwrite(path + "depth_detection" + std::to_string(iter) + ".png",
                  frame);
    }
    iter++;
  }
  std::cout << "Finished semantic mapping" << std::endl;
  return objects;
}

cv::Point start_point, end_point;
bool drawing = false;
cv::Mat image;

void mouse_callback(int event, int x, int y, int, void *data)
{
  MouseData *mouse_data = static_cast<MouseData *>(data);

  switch (event) {
  case cv::EVENT_LBUTTONDOWN: // Start drawing
    mouse_data->drawing = true;
    mouse_data->start_point = cv::Point(x, y);
    mouse_data->bounding_box = cv::Rect(x, y, 0, 0); // Initialize
    break;

  case cv::EVENT_MOUSEMOVE: // Update rectangle as the user drags
    if (mouse_data->drawing) {
      int width = x - mouse_data->start_point.x;
      int height = y - mouse_data->start_point.y;
      mouse_data->bounding_box = cv::Rect(
        mouse_data->start_point.x, mouse_data->start_point.y, width, height);
    }
    break;

  case cv::EVENT_LBUTTONUP: // Finish drawing
    if (mouse_data->drawing) {
      mouse_data->drawing = false;
      mouse_data->finished = true;
      int width = x - mouse_data->start_point.x;
      int height = y - mouse_data->start_point.y;
      mouse_data->bounding_box = cv::Rect(
        mouse_data->start_point.x, mouse_data->start_point.y, width, height);
      std::cout << "Bounding box: " << mouse_data->bounding_box << std::endl;
    }
    break;
  }
}

void manual_labeling(Result &result, std::vector<Object> &objects)
{
  for (const auto &frame : result.mapping_data) {
    cv::Mat rgb = std::get<0>(frame);
    cv::Mat depth = std::get<1>(frame);
    rtabmap::Transform pose = std::get<2>(frame);
    std::map<std::pair<int, int>, int> pixel_to_point_map = std::get<3>(frame);

    // Display the image and ask the user for a bounding box
    image = rgb;
    cv::imshow("Image", rgb);
    cv::setMouseCallback("Image", mouse_callback, nullptr);
    cv::waitKey(0);
  }
}

int main(int argc, char *argv[])
{
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
    Result result = extractor.load_rtabmap_db();

    std::vector<Object> objects = semantic_mapping(
      net, extractor, result.mapping_data, result.cloud, result.timestamp);

    // manual_labeling(result, objects);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}

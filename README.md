# RTABMap_DNN
Process RTABMap database files (.db) and run object detection with deep neural
networks to perform semantic mapping.

### Introduction
This repo is a piece of a larger project to perform semantic mapping with
mobile phones in order to create maps for a LUCI autonomous wheelchair. It takes
database files (.db) created by the RTABMap iPhone app, and extracts the point
cloud as a Point Cloud Library (.pcl) file, a 2D occupancy grid representation
of the point cloud, and rgb and emulated depth images for each pose in the pose
graph.

The executable is meant to be run with two arguments  
**RTABMap Database File (.db):**
* The name of a RTABMap database file (your_file.db) that is inside the ```databases```
directory of this repo. Copy your database files there for them to be processed.
**Image Recognition Model Name:**
* The name of the model (your_model.onnx) that you'd like to use to perform
object detection. I use OpenCV 4.10.0's DNN module to run object detection and
a yolov8n.onnx model, but you can use any model you'd like that works with this
verison of OpenCV.

### Warning
This project was designed specifically for RTABMap database files that were created
by the iPhone app without LIDAR. I plan to build in support for all types of
database files to extract the data from them, but as of today (11/15/2024) I
don't have the time. If you have database files created with methods other than
the RTABMap iPhone app, you can fork the repo and modify the code to get the information
you need by using [this](https://github.com/introlab/rtabmap/blob/ff61266430017eb4924605b832cd688c8739af18/tools/Export/main.cpp#L1104-L1115) as a guide.


### Docker
```bash
docker build -t rtabmap_dnn .
docker run --rm -v ./output:/app/output rtabmap_dnn <name_of_db_file> <name_of_dnn_model>
```
### Run on your host system
Make sure that you have ros2 humble installed on your system:  
https://docs.ros.org/en/humble/Installation.html

First source the ros2 humble setup file:
```bash
source /opt/ros/humble/setup.bash
```
Next install dependencies:
```bash
sudo apt update && apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    g++ \
    wget \
    unzip \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-pcl-conversions \
    libpcl-dev
```
Now we build and install opencv 4.10.0 from source:
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip && \
unzip opencv.zip && \
unzip opencv_contrib.zip && \
mkdir -p build && cd build && \
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules ../opencv-4.10.0 && \
make -j$(nproc) && \
make install && \
ldconfig && \
cd .. && \
rm -rf build opencv.zip opencv_contrib.zip opencv-4.10.0 opencv_contrib-4.10.0
```
Now we can build the rtabmap_dnn package:
```bash
mkdir build && cd build
cmake .. -G "Ninja"
ninja
```
To run the executable:
```bash
./database_exporter <name_of_db_file> <name_of_dnn_model>
```

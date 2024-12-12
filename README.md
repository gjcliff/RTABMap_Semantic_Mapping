# RTABMap_DNN
Process RTABMap database files (.db) and run object detection with deep neural
networks to perform semantic mapping.

This repo is part of a larger project. Check out my portfolio post about it [here](https://graham-clifford.com/Localizing-and-Navigating-in-Semantic-Maps-Created-by-an-iPhone/).

Table of Contents:
1. [Introduction](#introduction)
2. [Warnings](#warnings)
3. [Docker](#docker)
4. [Run on your host system](#run-on-your-host-system)

### Introduction
This repo is a piece of a larger project to perform semantic mapping with
mobile phones in order to create maps for a LUCI autonomous wheelchair. It takes
database files (.db) created by the RTABMap iPhone app, and extracts the point
cloud as a Point Cloud Library (.pcl) file, a 2D occupancy grid representation
of the point cloud, and rgb and emulated depth images for each pose in the pose
graph.

I used [this file](https://github.com/introlab/rtabmap/blob/ff61266430017eb4924605b832cd688c8739af18/tools/Export/main.cpp#L1104-L1115) from RTABMap's source code as a guide to write my code for
extracting information from the database files.

The executable has one argument
**RTABMap Database File (.db):**
* The name of a RTABMap database file (your_file.db) that is inside the ```databases/```
directory of this repo. Copy your database files there for them to be processed.
<!---->
<!-- **Image Recognition Model Name:** -->
<!-- * The name of the model (your_model.onnx) that you'd like to use to perform -->
<!-- object detection. I use OpenCV 4.10.0's DNN module to run object detection and -->
<!-- a yolov8n.onnx model, but you can use any model you'd like that works with this -->
<!-- verison of OpenCV. -->

### Docker

The idea is that you will create a map using the RTABMap for iPhone, and then
send this to your computer to use an input for the docker container. You can
find the database files in your iPhone's "Files" app, in the folder named
"RTABMap".
```bash
docker build -t rtabmap_dnn .
docker run --rm -e DISPLAY=$DISPLAY \
    -v ./output:/app/output \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    rtabmap_dnn <name_of_db_file>
```
or
```bash
docker build -t rtabmap_dnn .
docker run --rm -e DISPLAY=$DISPLAY -v ./output:/app/output\
    rtabmap_dnn <name_of_db_file>
```
If you'd like to visualize images during the process, you can enable X11 forwarding
by adding this option to your ```docker run``` command:
```bash
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix\
    -v ./output:/app/output rtabmap_dnn <name_of_db_file>
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
sudo apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    g++ \
    wget \
    unzip \
    pybind11-dev \
    python3-pip \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-pcl-conversions \
    libpcl-dev
```
Build and install RTABMap from source:
```bash
git clone https://github.com/introlab/rtabmap.git \
cd rtabmap \
git switch humble-devel \
cd build \
cmake .. \
make -j$(nproc) \
make install \
ldconfig \
cd /app \
rm -rf rtabmap
```
Install ultralytics:
```bash
pip install ultralytics
```
Build the rtabmap_dnn package:
```bash
mkdir build && cd build
cmake .. -G "Ninja"
ninja
```
To run the executable:
```bash
./database_exporter <name_of_db_file># <name_of_dnn_model>
```
### For running on LUCI
cd into ```/home/luci_ws/src/luci_aw_navigation/``` and run:
```bash
git pull
git switch mobile_phone
```
Copy the output file that was just created to your docker container:
```bash
docker cp output/<timestamp> luci_msr:/home/luci_ws/src/luci_aw_navigation/awl_navigation/output/
```
Copy the landmarks file into the landmarks directory:
```bash
docker exec -it luci_msr bash
cd src/luci_aw_navigation/awl_navigation/output/<timestamp>
cp landmarks/<timestamp>.yaml ../../landmarks/
colcon build
source install/setup.bash
```
The necessary commands for running AMCL localization are in the README.md file
the the luci_awl_navigation repo.

### Warnings
This project was designed specifically for RTABMap database files that were created
by the iPhone app without LIDAR. I plan to build in support for all types of
database files to extract the data from them, but as of today (11/15/2024) I
don't have the time. If you have database files created with methods other than
the RTABMap iPhone app, you can fork the repo and modify the code to get the information
you need by using [this](https://github.com/introlab/rtabmap/blob/ff61266430017eb4924605b832cd688c8739af18/tools/Export/main.cpp#L1104-L1115) as a guide.

Files created by the docker container in the ```output``` directory will be owned by
root. You can change the owner to your host system's user with this command:
```bash
sudo chown -R your_user:your_user output/*
```

FROM ros:humble-ros-base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    g++ \
    wget \
    unzip \
    ros-humble-rtabmap \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-pcl-conversions \
    libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules ../opencv-4.10.0 && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /app && \
    rm -rf build opencv.zip opencv_contrib.zip opencv-4.10.0 opencv_contrib-4.10.0

COPY ./include /app/include
COPY ./src /app/src
COPY ./databases /app/databases
COPY ./models /app/models
COPY ./CMakeLists.txt /app/CMakeLists.txt

RUN . /opt/ros/humble/setup.sh

RUN mkdir build && cd build && \
    cmake .. -G "Ninja" && \
    ninja

ENTRYPOINT ["/app/build/rtabmap_ros_node"]

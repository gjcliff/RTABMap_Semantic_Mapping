FROM ros:humble-ros-base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libclang-10-dev \
    git \
    libeigen3-dev \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2

COPY ./database_extractor /app/database_extractor

RUN cd /app/database_extractor && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j

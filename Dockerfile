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
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-pcl-conversions \
    libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/introlab/rtabmap.git \
    && cd rtabmap \
    && git switch humble-devel \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /app \
    && rm -rf rtabmap

COPY ./include /app/include
COPY ./src /app/src
COPY ./databases /app/databases
COPY ./models /app/models
COPY ./CMakeLists.txt /app/CMakeLists.txt
COPY ./entrypoint.sh /app/entrypoint.sh

RUN /bin/bash -c "source /opt/ros/humble/setup.bash \
    && mkdir build \
    && cd build \
    && cmake .. -G 'Ninja' \
    && ninja"

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

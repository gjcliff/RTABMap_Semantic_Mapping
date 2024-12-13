cmake_minimum_required(VERSION 3.15...3.29)
project(rtabmap_database_export)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_FINDPYTHON ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build." FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DPROJECT_PATH="${CMAKE_SOURCE_DIR}")

# set(CMAKE_PREFIX_PATH "/opt/ros/humble" ${CMAKE_PREFIX_PATH})

# find dependencies
find_package(PCL 1.12 REQUIRED)
find_package(RTABMap REQUIRED)
find_package(nav2_map_server REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(database_exporter
    src/database_exporter.cpp
)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    ${RTABMap_INCLUDE_DIRS}
    ${nav2_map_server_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(database_exporter PUBLIC
    ${PCL_LIBRARIES}
    ${opencv_LIBRARIES}
    ${pcl_conversions_LIBRARIES}
    ${nav2_map_server_LIBRARIES}
    rtabmap::rtabmap)

install(TARGETS database_exporter
    DESTINATION ${CMAKE_BINARY_DIR}/lib
)

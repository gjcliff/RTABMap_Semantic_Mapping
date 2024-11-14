cmake_minimum_required(VERSION 3.5)
project(rtabmap_database_export)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build." FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")

# find dependencies
find_package(rclcpp REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(RTABMap REQUIRED)
find_package(nav2_map_server REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED)

add_executable(database_exporter
    src/database_exporter.cpp
)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    ${RTABMap_INCLUDE_DIRS}
    ${nav2_map_server_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(database_exporter PUBLIC
    ${PCL_LIBRARIES}
    ${pcl_conversions_LIBRARIES}
    ${OpenCV_LIBS}
    ${nav2_map_server_LIBRARIES}
    rtabmap::rtabmap)

install(TARGETS database_exporter
    DESTINATION lib/${PROJECT_NAME}
)

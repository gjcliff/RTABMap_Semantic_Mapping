cmake_minimum_required(VERSION 3.8)
project(orb_slam3_ros2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)        # If no build type is set, set one
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build." FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")

# find dependencies
find_package(PCL 1.12 REQUIRED)
find_package(RTABMap REQUIRED)
find_package(nav2_map_server REQUIRED)

message("RTABMap_INCLUDE_DIRS: ${RTABMap_INCLUDE_DIRS}")
message("RTABMap_LIBRARIES: ${RTABMap_LIBRARIES}")

if(ORB_SLAM3_LIBRARY)
    message("Found ORB_SLAM3: ${ORB_SLAM3_LIBRARY}")
else()
    message("Could not find ORB_SLAM3 library")
endif()

add_executable(imu_mono_node_cpp
  src/imu_mono_realsense.cpp
)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
)

link_directories(
    include
)

target_link_libraries(imu_mono_node_cpp PUBLIC ${ORB_SLAM3_LIBRARY} ${PCL_LIBRARIES}
    ${libpointmatcher_LIBRARIES} ${libpointmatcher_ros_LIBRARIES} rtabmap::rtabmap)

install(TARGETS rtabmap_database_extractor_node
    DESTINATION lib/${PROJECT_NAME}
)

install (DIRECTORY include/
        DESTINATION include
)

# install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

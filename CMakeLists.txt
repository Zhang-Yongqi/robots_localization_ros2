cmake_minimum_required(VERSION 3.8)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

project(robots_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread -fexceptions -fPIC")

find_program(MOLD_LINKER mold)
find_program(LLD_LINKER lld)
if(MOLD_LINKER)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=mold")
    message(STATUS "Using mold linker")
elseif(LLD_LINKER)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
    message(STATUS "Using lld linker")
endif()

set(CMAKE_UNITY_BUILD ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)|(AARCH64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=6)
    message("core for MP: 6")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP: 4")
  else()
    add_definitions(-DMP_PROC_NUM=2)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(ROS2_DEPENDENCIES
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
)

find_package(nlink_message REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  include
  "src"
  )

set(MSG_FILES
  "msg/Pose6D.msg"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  DEPENDENCIES builtin_interfaces std_msgs
)
ament_export_dependencies(rosidl_default_runtime)
rosidl_get_typesupport_target(cpp_typesupport_target
${PROJECT_NAME} "rosidl_typesupport_cpp")

add_subdirectory(src)

add_executable(robots_localization_node src/robots_localization_node.cpp)
ament_target_dependencies(robots_localization_node ${ROS2_DEPENDENCIES} nlink_message livox_ros_driver2)
add_library(ikd_Tree SHARED include/ikd-Tree/ikd_Tree.cpp)
add_library(scan_aligner SHARED src/scan_aligner.cpp)
add_library(imu_processor SHARED src/imu_processor.cpp)
add_library(lidar_processor SHARED src/lidar_processor.cpp)
target_link_libraries(imu_processor fmt::fmt)
ament_target_dependencies(imu_processor ${ROS2_DEPENDENCIES})
target_link_libraries(scan_aligner fmt::fmt)
ament_target_dependencies(scan_aligner ${ROS2_DEPENDENCIES})
target_link_libraries(lidar_processor ${PCL_LIBRARIES})
ament_target_dependencies(lidar_processor ${ROS2_DEPENDENCIES} livox_ros_driver2)

add_executable(test_uwb_jacobian src/test_uwb_jacobian.cpp)

# PCH头文件预编译
target_precompile_headers(robots_localization_node PUBLIC
<pcl/common/transforms.h>
<pcl/filters/voxel_grid.h>
<pcl/filters/passthrough.h>
<pcl/filters/statistical_outlier_removal.h>
<pcl/io/pcd_io.h>
<pcl/point_types.h>
<pcl/point_cloud.h>
<geometry_msgs/msg/pose_stamped.hpp>
<geometry_msgs/msg/transform_stamped.hpp>
<geometry_msgs/msg/twist_stamped.hpp>
<geometry_msgs/msg/vector3.hpp>
<nav_msgs/msg/path.hpp>
<nav_msgs/msg/grid_cells.hpp>
<sensor_msgs/msg/imu.hpp>
<sensor_msgs/msg/point_cloud2.hpp>
include/common_lib.h
include/so3_math.h
include/use-ikfom.h
include/ikd-Tree/ikd_Tree.h
include/IKFoM_toolkit/esekfom/esekfom.hpp
include/IKFoM_toolkit/esekfom/util.hpp
include/IKFoM_toolkit/mtk/build_manifold.hpp
include/IKFoM_toolkit/mtk/startIdx.hpp
include/IKFoM_toolkit/mtk/types/S2.hpp
include/IKFoM_toolkit/mtk/types/SOn.hpp
include/IKFoM_toolkit/mtk/types/vect.hpp
include/IKFoM_toolkit/mtk/src/mtkmath.hpp
include/IKFoM_toolkit/mtk/src/SubManifold.hpp
include/IKFoM_toolkit/mtk/src/vectview.hpp
)

target_include_directories(robots_localization_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_include_directories(robots_localization_node PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(robots_localization_node 
${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${Sophus_LIBRARIES} ${CERES_LIBRARIES} fmt
ikd_Tree
scan_aligner
imu_processor
lidar_processor
monitor
)

target_link_libraries(robots_localization_node ${cpp_typesupport_target})
target_link_libraries(lidar_processor ${cpp_typesupport_target})
target_link_libraries(scan_aligner ${cpp_typesupport_target})
target_link_libraries(imu_processor ${cpp_typesupport_target})

install(TARGETS robots_localization_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS test_uwb_jacobian
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ikd_Tree scan_aligner imu_processor lidar_processor monitor
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY config launch rviz PCD
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
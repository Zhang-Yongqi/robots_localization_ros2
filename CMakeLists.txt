cmake_minimum_required(VERSION 3.0.2)

#set(CMAKE_C_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")
#set(CMAKE_C_COMPILER "/usr/bin/clang")
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_FLAGS "-std=c++14 -O3" )
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
# set(CC "ccache gcc")
# set(CXX "ccache g++")

project(robots_localization)
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++14 )



add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions -fPIC" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -std=c++14 -fexceptions")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)


message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=15)
    message("core for MP: 15")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
#find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  # livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Sophus REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
  "/home/cyl/code/ws_localization/devel/include"
  "/home/hitcrt/文档/cyl/rm2023_sentry_msgs/build/devel/include"
  "src"
  )

add_message_files(
  FILES
  Pose6D.msg
)
# add livox lidar custom msgs
add_message_files(
  DIRECTORY msg/livox
  FILES
  CustomPoint.msg
  LivoxMsg.msg 
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)



add_subdirectory(src)

add_executable(robots_localization src/robots_localization_node.cpp)
add_library(ikd_Tree SHARED include/ikd-Tree/ikd_Tree.cpp)
add_library(imu_processor SHARED src/imu_processor.cpp)
add_library(lidar_processor SHARED src/lidar_processor.cpp)
target_link_libraries(imu_processor fmt)


# PCH头文件预编译
target_precompile_headers(robots_localization PUBLIC
src/imu_processor.h
src/lidar_processor.h
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

target_include_directories(robots_localization PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(robots_localization 
${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${Sophus_LIBRARIES} fmt
ikd_Tree
imu_processor
lidar_processor
monitor
)



# add_executable(transform_map src/transform_map.cpp)
# target_link_libraries(transform_map ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${Sophus_LIBRARIES})


cmake_minimum_required(VERSION 3.0.2)

#set(CMAKE_C_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")
#set(CMAKE_C_COMPILER "/usr/bin/clang")
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_FLAGS "-std=c++17 -O3" )
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
# set(CC "ccache gcc")
# set(CXX "ccache g++")

project(robots_localization)
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17 )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions -fPIC" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -std=c++17 -fexceptions")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)


message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)|(AARCH64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=8)
    message("core for MP: 8")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
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
  message_generation
  eigen_conversions
  nlink_parser
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  include
  "usr/local/include/fmt"
  "src"
  )

add_message_files(
  FILES
  Pose6D.msg
  CustomMsg.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime nlink_parser
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)



add_subdirectory(src)

add_executable(robots_localization src/robots_localization_node.cpp)
add_library(ikd_Tree SHARED include/ikd-Tree/ikd_Tree.cpp)
add_library(scan_aligner SHARED src/scan_aligner.cpp)
add_library(imu_processor SHARED src/imu_processor.cpp)
add_library(lidar_processor SHARED src/lidar_processor.cpp)
target_link_libraries(imu_processor fmt ${catkin_LIBRARIES})
target_link_libraries(scan_aligner fmt ${catkin_LIBRARIES})

add_executable(test_uwb_jacobian src/test_uwb_jacobian.cpp)

# PCH头文件预编译
target_precompile_headers(robots_localization PUBLIC
src/scan_aligner.h
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
${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${Sophus_LIBRARIES} ${CERES_LIBRARIES} fmt
ikd_Tree
scan_aligner
imu_processor
lidar_processor
monitor
)


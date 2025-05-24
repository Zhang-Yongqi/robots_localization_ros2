#pragma once

#include <pcl_conversions/pcl_conversions.h>
#include <robots_localization/CustomMsg.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// 枚举类型：表示支持的雷达类型
enum LID_TYPE { AVIA = 1, VELO16, OUST64, UNIL2, RS };  //{1, 2, 3, 4, 5}
// 枚举类型：表示时间的单位
enum TIME_UNIT { SEC = 0, MS = 1, US = 2, NS = 3 };
// 枚举类型：表示特征点的类型
enum Feature {
  Nor,         // 正常点
  Poss_Plane,  // 可能的平面点
  Real_Plane,  // 确定的平面点
  Edge_Jump,   // 有跨越的边
  Edge_Plane,  // 边上的平面点
  Wire,  // 线段 这个也许当了无效点？也就是空间中的小线段？
  ZeroPoint  // 无效点，不使用
};
// 枚举类型：位置标识
enum Surround { Prev, Next };  // 前一个，后一个
// 枚举类型：表示有跨越边的类型 {正常，0，180，无穷远，在盲区}
enum E_jump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

// orgtype类：用于存储激光雷达点的一些其他属性
struct orgtype {
  double range;  // 点云在xy平面离雷达中心的距离
  double dista;  // 当前点与后一个点之间的距离
  // 假设雷达原点为O 前一个点为M 当前点为A 后一个点为N
  double angle[2];   // 角OAM和角OAN的cos值
  double intersect;  // 角MAN的cos值
  E_jump edj[2];     // 前后两点的类型
  Feature ftype;     // 点类型
  orgtype() {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace rslidar_ros {
struct Point {
    PCL_ADD_POINT4D;
    std::uint8_t intensity;
    uint16_t ring = 0;
    double timestamp = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}  // namespace rslidar_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
                                      uint16_t, ring, ring)(double, timestamp, timestamp))

namespace unilidar_ros {
struct Point {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY
    std::uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}  // namespace unilidar_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(unilidar_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(std::uint16_t, ring,
                                                                                     ring)(float, time, time))

// velodyne数据结构
namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;  // 4D点坐标类型，x、y、z 还有一个对齐变量
  float intensity;
  float time;
  uint16_t ring;  // 点所属的圈数
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
// 注册velodyne_ros的Point类型
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, time, time)(std::uint16_t,
                                                                    ring, ring))

// ouster数据结构
namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;  // 4D点坐标类型，x、y、z 还有一个对齐变量
  float intensity;
  uint32_t t;
  uint16_t reflectivity;  // 反射率
  uint8_t ring;           // 点所属的圈数
  uint16_t ambient;       // 没用
  uint32_t range;         // 距离
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros
// 注册ouster的Point类型
POINT_CLOUD_REGISTER_POINT_STRUCT(
    ouster_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(
        std::uint8_t, ring, ring)(std::uint16_t, ambient,
                                  ambient)(std::uint32_t, range, range)
                                  )

class LidarProcessor {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LidarProcessor();
  ~LidarProcessor();

  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg,
               PointCloudXYZI::Ptr &pcl_out);

  // 雷达类型、扫描线数、时间单位
  int lidar_type, N_SCANS, SCAN_RATE, time_unit, point_filter_num;
  float time_unit_scale;
  // 最小距离阈值(盲区)
  double blind;
  bool given_offset_time;

  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128];  // maximum 128 line lidar

 private:
  void rs_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void unilidar_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
};


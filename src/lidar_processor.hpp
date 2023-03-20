#pragma once

// #include <livox_ros_driver/CustomMsg.h>
#include <robots_localization/LivoxMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// 枚举类型：表示支持的雷达类型
enum LID_TYPE { AVIA = 1, VELO16, OUST64 };  //{1, 2, 3}
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
                                      intensity)(float, time, time)(uint16_t,
                                                                    ring, ring))

// ouster数据结构
namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;  // 4D点坐标类型，x、y、z 还有一个对齐变量
  float intensity;
  uint32_t t;
  uint16_t reflectivity;  // 反射率
  uint8_t ring;           // 点所属的圈数
  uint16_t ambient;       //没用
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
                                  ambient)(std::uint32_t, range, range))

class LidarProcessor {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LidarProcessor();
  ~LidarProcessor();

  void process(const robots_localization::LivoxMsg::ConstPtr &msg,
               PointCloudXYZI::Ptr &pcl_out);
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
  void avia_handler(const robots_localization::LivoxMsg::ConstPtr &msg);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
};

LidarProcessor::LidarProcessor() {}

LidarProcessor::~LidarProcessor() {}

void LidarProcessor::process(const robots_localization::LivoxMsg::ConstPtr &msg,
                             PointCloudXYZI::Ptr &pcl_out) {
  avia_handler(msg);
  *pcl_out = pl_surf;
}

void LidarProcessor::process(const sensor_msgs::PointCloud2::ConstPtr &msg,
                             PointCloudXYZI::Ptr &pcl_out) {
  switch (time_unit) {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type) {
    case OUST64:
      oust64_handler(msg);
      break;
    case VELO16:
      velodyne_handler(msg);
      break;
    default:
      printf("Error LiDAR Type");
      break;
  }
  *pcl_out = pl_surf;
}

void LidarProcessor::avia_handler(
    const robots_localization::LivoxMsg::ConstPtr &msg) {
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  int plsize = msg->point_num;
  if (plsize == 0) return;
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for (int i = 0; i < N_SCANS; i++) {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;

  for (uint i = 1; i < plsize; i++) {
    if ((msg->points[i].line < N_SCANS) &&
        ((msg->points[i].tag & 0x30) == 0x10 ||
         (msg->points[i].tag & 0x30) == 0x00)) {
      valid_num++;
      if (valid_num % point_filter_num == 0) {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000);
        // use curvature as time of each laser points, curvature unit: ms

        if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) ||
            (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
            (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7) &&
                (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y +
                     pl_full[i].z * pl_full[i].z >
                 (blind * blind))) {
          pl_surf.push_back(pl_full[i]);
        }
      }
    }
  }
}

void LidarProcessor::oust64_handler(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  if (plsize == 0) return;
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  double time_stamp = msg->header.stamp.toSec();
  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (i % point_filter_num != 0) continue;

    double range = pl_orig.points[i].x * pl_orig.points[i].x +
                   pl_orig.points[i].y * pl_orig.points[i].y +
                   pl_orig.points[i].z * pl_orig.points[i].z;
    if (range < (blind * blind)) continue;

    Eigen::Vector3d pt_vec;
    PointType added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
    // use curvature as time of each laser points, curvature unit: ms

    pl_surf.points.push_back(added_pt);
  }
}

void LidarProcessor::velodyne_handler(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
  /*****************************************************************/

  if (pl_orig.points[plsize - 1].time > 0) {
    given_offset_time = true;
  } else {
    given_offset_time = false;
    double yaw_first =
        atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--) {
      if (pl_orig.points[i].ring == layer_first) {
        yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
        break;
      }
    }
  }

  for (int i = 0; i < plsize; i++) {
    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = pl_orig.points[i].time * time_unit_scale;
    // curvature unit: ms

    if (!given_offset_time) {
      int layer = pl_orig.points[i].ring;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

      if (is_first[layer]) {
        // printf("layer: %d; is first: %d", layer, is_first[layer]);
        yaw_fp[layer] = yaw_angle;
        is_first[layer] = false;
        added_pt.curvature = 0.0;
        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
        continue;
      }

      // compute offset time
      if (yaw_angle <= yaw_fp[layer]) {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
      } else {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
      }

      if (added_pt.curvature < time_last[layer])
        added_pt.curvature += 360.0 / omega_l;

      yaw_last[layer] = yaw_angle;
      time_last[layer] = added_pt.curvature;
    }

    if (i % point_filter_num == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y +
              added_pt.z * added_pt.z >
          (blind * blind)) {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}
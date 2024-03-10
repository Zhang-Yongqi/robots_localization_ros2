// #include <common_lib.h>
#include <geometry_msgs/TwistStamped.h>
#include <ikd-Tree/ikd_Tree.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <unistd.h>

#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>

#include "imu_processor.h"
#include "lidar_processor.h"
#include "scan_aligner.h"

#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)
#define DET_RANGE (300.0f)
#define MOV_THRESHOLD (1.5f)

ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudFull_world;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubPath;
ros::Publisher pubIMUBias;
ros::Time start_time;
ros::Time end_time;
ros::Duration duration_time;
ros::Time start2;
ros::Time end2;
ros::Duration duration2;
ros::Time start3;
ros::Time end3;
ros::Duration duration3;

string root_dir = ROOT_DIR;
string lid_topic, imu_topic, reloc_topic, mode_topic, pcd_path;

bool time_sync_en = false;
bool path_en = false, scan_pub_en = false, dense_pub_en = false,
     scan_body_pub_en = false;
bool extrinsic_est_en = true, runtime_pos_log = false, flg_exit = false;
bool lidar_pushed, flg_first_scan = true, initialized = false, first_pub = true;

double time_diff_lidar_to_imu = 0.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double fov_deg = 0.0, filter_size_corner_min = 0.0, filter_size_surf_min = 0.0,
       filter_size_map_min = 0.0, cube_len = 0.0;
double last_timestamp_lidar = 0.0, last_timestamp_imu = -1.0,
       last_timestamp_imu_back = -1.0;
double lidar_end_time = 0.0, first_lidar_time = 0.0;
double total_residual = 0.0, res_mean_last = 0.0;
float det_range = 300.0f;

int NUM_MAX_ITERATIONS = 0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int feats_down_size = 0;

bool need_reloc = false, imu_only_ready = false, point_not_enough = false, initT_flag = true;
float point_num = 0.0f, point_valid_num = 0.0f, point_valid_proportion = 0.0f;
V3F reloc_initT(Zero3f);

vector<float> red_priorT(3, 0.0);
vector<float> blue_priorT(3, 0.0);
vector<float> YAW_RANGE(3, 0.0);
vector<float> priorR(9, 0.0);
V3F red_prior_T(Zero3f);
V3F blue_prior_T(Zero3f);
M3F prior_R(Eye3f);
bool mode_status = 0, mode_changed = false, pose_inited = false; // 0为红方，1 为蓝方
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

pcl::VoxelGrid<PointType> downSizeFilterSurf;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
state_ikfom state_point_imu;
vect3 pos_lid;

mutex mtx_buffer;
condition_variable sig_buffer;
deque<double> time_buffer;                    // 激光雷达数据
deque<PointCloudXYZI::Ptr> lidar_buffer;      // 雷达数据队列
deque<sensor_msgs::Imu::ConstPtr> imu_buffer; // IMU数据队列

// PointCloudXYZI: 点云坐标 + 信号强度形式
PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI()); // 雷达坐标系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());       // 世界坐标系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // 雷达滤波
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 存放法向

KD_TREE<PointType> ikdtree;
vector<PointVector> Nearest_Points;
float res_last[100000] = {0.0};
bool point_selected_surf[100000] = {0};

shared_ptr<LidarProcessor> p_lidar(new LidarProcessor());
shared_ptr<IMUProcessor> p_imu(new IMUProcessor());

nav_msgs::Path path;
geometry_msgs::PoseStamped msg_body_pose;
geometry_msgs::Quaternion geoQuat;
nav_msgs::Odometry odomAftMapped;
nav_msgs::Odometry odomAftMappedIMU;
geometry_msgs::TwistStamped odomIMUBias;  // 用来传IMU的偏置

std::ofstream fout_pose;
ros::Publisher sub_pub_imu;

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

void SigHandle(int sig)
{
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

// 参数加载
void loadConfig(const ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<string>("common/reloc_topic", reloc_topic, "/slaver/robot_command");
  nh.param<string>("common/mode_topic", mode_topic, "/slaver/robot_status");
  nh.param<string>("pcd_path", pcd_path,
                   "/home/zoe/catkin_ws/src/FAST_LIO/PCD/scans_mbot_0.pcd");
  nh.param<bool>("common/time_sync_en", time_sync_en, false);
  nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu,
                   0.0);

  nh.param<int>("preprocess/lidar_type", p_lidar->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_lidar->N_SCANS, 16);
  nh.param<int>("preprocess/scan_rate", p_lidar->SCAN_RATE, 10);
  nh.param<int>("preprocess/timestamp_unit", p_lidar->time_unit, US);
  nh.param<double>("preprocess/blind", p_lidar->blind, 0.01);
  nh.param<int>("point_filter_num", p_lidar->point_filter_num, 2);

  nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
  nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
  nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
  nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
  nh.param<double>("mapping/fov_degree", fov_deg, 180);
  nh.param<float>("mapping/det_range", det_range, 300.f);
  nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
  nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

  nh.param<bool>("publish/path_en", path_en, true);
  nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
  nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
  nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);

  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
  nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
  nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
  nh.param<double>("cube_side_length", cube_len, 200);
  nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);

  nh.param<string>("init_method", p_imu->method, "PPICP");
  if (p_imu->method == "NDT")
  {
  }
  else if (p_imu->method == "ICP")
  {
      nh.param<int>("ICP/max_iter", ScanAligner::max_iter, 10);
  } else if (p_imu->method == "PPICP") {
      nh.param<float>("PPICP/plane_dist", ScanAligner::plane_dist, 0.1);
      nh.param<int>("PPICP/max_iter", ScanAligner::max_iter, 10);
  } else if (p_imu->method == "GICP") {
      nh.param<int>("GICP/max_iter", ScanAligner::max_iter, 64);
  } else {
      std::cerr << "Not valid init method!" << std::endl;
  }
  nh.param<vector<float>>("prior/red_prior_T", red_priorT,
                          vector<float>(3, 0.0));
  nh.param<vector<float>>("prior/blue_prior_T", blue_priorT,
                          vector<float>(3, 0.0));
  nh.param<vector<float>>("prior/prior_R", priorR,
                          vector<float>(9, 0.0));
  nh.param<bool>("prior/estimateGrav", p_imu->estimateGrav, true);
}

double timediff_lidar_wrt_imu = 0.0; // lidar imu 时间差
bool timediff_set_flg = false;       // 是否已经计算了时间差
// livox激光雷达回调函数
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer.lock();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    // time_sync_en时间同步关闭，imu和lidar时间差>10，两个buffer都不为空，就输出
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() &&
        !lidar_buffer.empty()) {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf", last_timestamp_imu,
               last_timestamp_lidar);
    }
    // 如果是同一个时间系统，正常情况下不会相差大于1s（不是同一个时间系统）
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 &&
        !imu_buffer.empty()) {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is % .10lf ", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_lidar->process(msg, ptr);  // 数据格式转换
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 标准雷达回调函数
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  mtx_buffer.lock();
  scan_count++;
  if (msg->header.stamp.toSec() < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_lidar->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_lidar = msg->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all(); // 唤醒所有线程
}

// 接收IMU数据回调函数
// ConstPtr: 智能指针
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  // end_time = start_time;
  // start_time = ros::Time::now();
  // duration_time = start_time - end_time;
  // ROS_INFO("imu spin time is %f", duration_time.toSec() * 1000);

  publish_count++;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
  // 将IMU和激光雷达点云的时间戳对齐（livox）
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
  {
    msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu +
                                            msg_in->header.stamp.toSec());
  }
  // 将IMU和激光雷达点云的时间戳对齐（else）

  double timestamp = msg->header.stamp.toSec();

  // 上锁
  mtx_buffer.lock();
  if (timestamp < last_timestamp_imu)
  {
    ROS_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }
  last_timestamp_imu = timestamp;
  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  sig_buffer.notify_all();

  if (initialized && !need_reloc && imu_only_ready && last_timestamp_imu > 0.0 &&
      last_timestamp_imu > last_timestamp_imu_back) {
      if (p_imu->process_imu_only(msg, kf)) {
          state_point_imu = kf.get_x_imu();
          odomAftMappedIMU.header.frame_id = "camera_init";
          odomAftMappedIMU.child_frame_id = "body";
          odomAftMappedIMU.header.stamp = ros::Time().fromSec(last_timestamp_imu);
          odomAftMappedIMU.pose.pose.position.x = state_point_imu.pos(0);
          odomAftMappedIMU.pose.pose.position.y = state_point_imu.pos(1);
          odomAftMappedIMU.pose.pose.position.z = state_point_imu.pos(2);
          odomAftMappedIMU.twist.twist.linear.x = state_point_imu.vel(0);
          odomAftMappedIMU.twist.twist.linear.y = state_point_imu.vel(1);
          odomAftMappedIMU.twist.twist.linear.z = state_point_imu.vel(2);
          odomAftMappedIMU.pose.pose.orientation.x = state_point_imu.rot.coeffs()[0];
          odomAftMappedIMU.pose.pose.orientation.y = state_point_imu.rot.coeffs()[1];
          odomAftMappedIMU.pose.pose.orientation.z = state_point_imu.rot.coeffs()[2];
          odomAftMappedIMU.pose.pose.orientation.w = state_point_imu.rot.coeffs()[3];

          sub_pub_imu.publish(odomAftMappedIMU);
          // end3 = start3;
          // start3 = ros::Time::now();
          // duration3 = start3 - end3;
          // ROS_INFO("imu publish spin time is %f", duration3.toSec() * 1000);

          last_timestamp_imu_back = last_timestamp_imu;
      }
  }

  // end2 = ros::Time::now();
  // duration2 = end2 - start_time;
  // ROS_INFO("imu process time is %f", duration2.toSec() * 1000);
}

void reloc_cbk(const RobotCommand::ConstPtr &msg_in)
{
  mtx_buffer.lock();
  if (msg_in->commd_keyboard == 82)
  {
    reloc_initT << msg_in->target_position_x, msg_in->target_position_y,
        msg_in->target_position_z;
  }
  mtx_buffer.unlock();
}

void mode_cbk(const RobotStatus::ConstPtr &msg_in)
{
  if (!mode_changed && !pose_inited)
  {
    mtx_buffer.lock();
    // std::cout<<"id: "<<(int)msg_in->id<<std::endl;
    if ((int)msg_in->id > 100)
    {
      mode_status = 1; // 蓝方
    }
    else
    {
      mode_status = 0; // 红方
    }
    mode_changed = true;
    std::cout << "mode changed with " << mode_status << std::endl;
    mtx_buffer.unlock();
  }
}

// 将两帧激光雷达点云数据时间内的IMU数据从缓存队列中取出，进行时间对齐，并保存到meas中
// 输入数据：lidar_buffer, imu_buffer
// 输出数据：MeasureGroup
// 备注：必须同时有IMU数据和lidar数据
bool sync_packages(MeasureGroup &meas)
{
  if (lidar_buffer.empty() || imu_buffer.empty())
  {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed)
  {
    meas.lidar = lidar_buffer.front();          // lidar指针指向最旧的lidar数据

    if (meas.lidar->points.size() < 1) {
        cout << "lose lidar" << endl;
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        return false;
    }

    meas.lidar_beg_time = time_buffer.front();  // 记录最早时间
    // 更新结束时刻的时间

    /*** sort point clouds by offset time ***/
    sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);
    lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  // 必须有IMU数据
  if (last_timestamp_imu < lidar_end_time)
  { // imu落后lidar
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = imu_buffer.front()->header.stamp.toSec();  // 最旧IMU时间
  meas.imu.clear();

  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))  // 记录imu数据，imu时间小于当前帧lidar结束时间
  {
      imu_time = imu_buffer.front()->header.stamp.toSec();
      if (imu_time <
          meas.lidar_beg_time - (meas.lidar->points.front().curvature + meas.lidar->points.back().curvature) /
                                    double(1000)) {  // 舍弃过老imu数据
          imu_buffer.pop_front();
          continue;
      }
      if (imu_time > lidar_end_time) {
          break;
      }
      meas.imu.push_back(imu_buffer.front());  // 记录当前lidar帧内的imu数据到meas.imu
      imu_buffer.pop_front();
  }
  // std::cout << "meas.imu.size:    " << meas.imu.size() << std::endl;

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed = false;
  return true;
}

template <typename T>
void set_posestamp(T &out)
{
  out.pose.position.x = state_point.pos(0);
  out.pose.position.y = state_point.pos(1);
  out.pose.position.z = state_point.pos(2);
  out.pose.orientation.x = geoQuat.x;
  out.pose.orientation.y = geoQuat.y;
  out.pose.orientation.z = geoQuat.z;
  out.pose.orientation.w = geoQuat.w;
}

// 通过pubOdomAftMapped发布位姿odomAftMapped，同时计算协方差存在kf中，同tf计算位姿
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
  set_posestamp(odomAftMapped.pose); // 设置位置，欧拉角

  ////////////////触发重定位////////////////
  if (point_valid_proportion < 0.2 || point_not_enough) {
      need_reloc = true;
      initT_flag = false;
  } else {
      need_reloc = false;
      initT_flag = true;
  }

  odomAftMapped.twist.twist.linear.x = state_point.vel(0);
  odomAftMapped.twist.twist.linear.y = state_point.vel(1);
  odomAftMapped.twist.twist.linear.z = state_point.vel(2);
  pubOdomAftMapped.publish(odomAftMapped);
  auto P = kf.get_P();
  for (int i = 0; i < 6; i++)
  {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                  odomAftMapped.pose.pose.position.y,
                                  odomAftMapped.pose.pose.position.z));
  q.setW(odomAftMapped.pose.pose.orientation.w);
  q.setX(odomAftMapped.pose.pose.orientation.x);
  q.setY(odomAftMapped.pose.pose.orientation.y);
  q.setZ(odomAftMapped.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp,
                                        "camera_init", "body"));

  odomIMUBias.twist.linear.x = state_point.ba(0);
  odomIMUBias.twist.linear.y = state_point.ba(1);
  odomIMUBias.twist.linear.z = state_point.ba(2);
  // 利用twist的空位发布bg
  odomIMUBias.twist.angular.x = state_point.bg(0);
  odomIMUBias.twist.angular.y = state_point.bg(1);
  odomIMUBias.twist.angular.z = state_point.bg(2);
  pubIMUBias.publish(odomIMUBias);

  if (runtime_pos_log)
  {
    fout_pose << lidar_end_time << ", " << odomAftMapped.pose.pose.position.x
              << ", " << odomAftMapped.pose.pose.position.y << ", "
              << odomAftMapped.pose.pose.position.z << ", "
              << odomAftMapped.pose.pose.orientation.x << ", "
              << odomAftMapped.pose.pose.orientation.y << ", "
              << odomAftMapped.pose.pose.orientation.z << ", "
              << odomAftMapped.pose.pose.orientation.w << std::endl;
  }
}

// pi:激光雷达坐标系
// 函数功能：激光雷达坐标点转到世界坐标系
// state_point.offset_R_L_I*p_body + state_point.offset_T_L_I:转到IMU坐标系
// state_point.rot: IMU坐标系到世界坐标系的旋转
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

// 激光雷达坐标系到IMU坐标系
void pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar +
                 state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
  msg_body_pose.header.frame_id = "camera_init";
  path.header.frame_id = "camera_init";

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0)
  {
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
  }
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
  PointCloudXYZI::Ptr laserCloudWorld(global_map);
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFull.publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

// 发布feats_undistort转到IMU下的laserCloudIMUBody
void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
  for (int i = 0; i < size; i++)
  {
    pointBodyLidarToIMU(&feats_undistort->points[i],
                        &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.frame_id = "body";
  pubLaserCloudFull_body.publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

void publish_frame_world_local(const ros::Publisher &pubLaserCloudFull_world)
{
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
  for (int i = 0; i < size; i++)
  {
    pointBodyToWorld(&feats_undistort->points[i],
                     &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFull_world.publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

// 观测模型
void h_share_model(state_ikfom &s,
                   esekfom::dyn_share_datastruct<double> &ekfom_data)
{
  laserCloudOri->clear();
  corr_normvect->clear();
  total_residual = 0.0; // 残差和

  point_valid_num = 0.0;
  point_num = 0.0;
  // 最近邻面搜索，以及残差计算
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  /** closest surface search and residual computation **/
  // 遍历所有特征点，判断每个点的对应邻域是否符合平面点的假设
  for (int i = 0; i < feats_down_size; i++)
  {
    point_num += 1.0;
    // feats_down_body: 网格滤波器之后的激光点
    PointType &point_body = feats_down_body->points[i];
    // feats_down_world: 世界坐标系下的激光点
    PointType &point_world = feats_down_world->points[i];

    V3D p_body(point_body.x, point_body.y, point_body.z);
    /* transform to world frame */
    // 激光雷达坐标系->IMU坐标系->世界坐标系
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity; // 信号强度

    // NUM_MATCH_POINTS: 5
    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    auto &points_near = Nearest_Points[i];

    if (ekfom_data.converge)
    {
      /** Find the closest surfaces in the map **/
      // 在地图中找到与之最邻近的平面，world系下从ikdtree找NUM_MATCH_POINTS个最近点用于平面拟合
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                             pointSearchSqDis);
      // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，
      // 则认为该点不是有效点
      // 判断是否是有效匹配点，与LOAM系列类似，要求特征点最近邻的地图点数量大于阈值A，距离小于阈值B
      point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false
                               : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5
                                   ? false
                                   : true;
    }
    if (!point_selected_surf[i])
      continue; // 如果该点不是有效点

    VF(4) pabcd;                     // 法向量
    point_selected_surf[i] = false;  // 二次筛选平面点
    // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
    if (esti_plane(pabcd, points_near, 0.1f))
    { // 计算平面法向量
      // 根据它计算过程推测points_near的原点应该是这几个点中的一个，拟合了平面之后原点也就近似在平面
      // 上了，这样下面算出来的投影就是点到平面的距离。
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                  pabcd(2) * point_world.z + pabcd(3); // 计算点到平面的距离
      // 发射距离越长，测量误差越大，归一化，消除雷达点发射距离的影响
      // p_body是激光雷达坐标系下的点
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); // 判断残差阈值

      if (s > 0.9)
      { // 如果残差大于阈值，则认为该点是有效点
        point_selected_surf[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;  // 以intensity记录点到面残差
        res_last[i] = fabs(pd2);             // 残差，距离
        point_valid_num += 1.0;
      }
    }
  }
  point_valid_proportion = point_valid_num / point_num;
  if (point_valid_num < 100) {
      point_not_enough = true;
  } else {
      point_not_enough = false;
  }

  // 根据point_selected_surf状态判断哪些点是可用的
  effct_feat_num = 0;
  for (int i = 0; i < feats_down_size; i++)
  {
    if (point_selected_surf[i])
    { // 只保留有效的特征点
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      total_residual += res_last[i]; // 计算总残差
      effct_feat_num++;
    }
  }
  if (effct_feat_num < 1)
  {
    ekfom_data.valid = false;
    ROS_WARN("No Effective Points! \n");
    return;
  }
  res_mean_last = total_residual / effct_feat_num;  // 残差均值 （距离）

  /* Computation of Measuremnt Jacobian matrix H and measurents vector */
  // 测量雅可比矩阵H和测量向量的计算 H=J*P*J'
  // h_x是观测h相对于状态x的jacobian，尺寸为特征点数x12
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // (23)
  ekfom_data.h.resize(effct_feat_num);                 // 有效方程个数

  // 求观测值与误差的雅克比矩阵，如论文式14以及式12、13
  for (int i = 0; i < effct_feat_num; i++)
  {
    // 拿到有效点的坐标
    const PointType &laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    // 转换到IMU坐标系下
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType &norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);  // 对应局部法相量, world系下

    /*** calculate the Measuremnt Jacobian matrix H ***/
    // conjugate()用于计算四元数的共轭，表示旋转的逆
    V3D C(s.rot.conjugate() * norm_vec); // 世界坐标系的法向量旋转到IMU坐标系
    V3D A(point_crossmat * C);           // IMU坐标系下原点到点云点距离在法向上的投影

    if (extrinsic_est_en)
    { // extrinsic_est_en: IMU,lidar外参在线更新
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() *
            C); // Lidar坐标系下点向量在法向上的投影
      // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    }
    else
    {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
}

// bool isinit = false;  // 无串口时调试用
void mainProcess() {
    // // 无串口时调试用
    // if (isinit == false) {
    //     mode_changed = true;
    //     isinit = true;
    // }

    if (mode_changed) {
        if (mode_status) {
            p_imu->set_init_pose(blue_prior_T, prior_R);
            std::cout << "init with blue" << std::endl << std::endl;
        } else {
            p_imu->set_init_pose(red_prior_T, prior_R);
            std::cout << "init with red" << std::endl << std::endl;
        }
        mode_changed = false;
        pose_inited = true;
    }

    if (!pose_inited) {
        std::cout << "slaver received nothing, init failed" << std::endl;
        return;
    }

    if (sync_packages(Measures))  // 在Measure内，储存当前lidar数据及lidar扫描时间内对应的imu数据序列
    {
        ros::Time start = ros::Time::now();
        if (flg_first_scan)  // 第一帧lidar数据
        {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;  // 记录第一帧绝对时间
            flg_first_scan = false;
            return;
        }

        if (Measures.imu.empty()) {
            std::cout << "no imu meas" << std::endl;
            return;
        }
        ROS_ASSERT(Measures.lidar != nullptr);

        // 初始化位姿
        if (!initialized) {
            initialized = p_imu->init_pose(Measures, kf, global_map, ikdtree, YAW_RANGE);
            return;
        }

        // 重定位
        if (need_reloc) {
            std::cout << "!!!!!!!!!need relocalization!!!!!!!!!" << std::endl;
            if (reloc_initT.norm() > 0.01) {
                std::cout << "reloc_initT: " << reloc_initT << std::endl;
                if (!initT_flag) {
                    imu_only_ready = false;
                    p_imu->reset();
                    p_imu->set_init_pose(reloc_initT, prior_R);
                    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
                    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
                    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
                    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
                    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
                    initT_flag = true;
                }
                if (p_imu->init_pose(Measures, kf, global_map, ikdtree, YAW_RANGE)) {
                    need_reloc = false;
                    reloc_initT = V3F(0.0, 0.0, 0.0);
                }
                return;
            }
        }

        // 对IMU数据进行预处理，其中包含了前向传播、点云畸变处理
        // feats_undistort 为畸变纠正之后的点云,lidar系
        p_imu->process(Measures, kf, *feats_undistort);
        state_point = kf.get_x();  // 前向传播后body的状态预测值
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        if (feats_undistort->empty() || (feats_undistort == NULL)) {
            ROS_WARN("No point, skip this scan!\n");
            return;
        }

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);  // 降采样
        feats_down_size = feats_down_body->points.size();
        if (feats_down_size < 5) {
            ROS_WARN("No point, skip this scan!\n");
            return;
        }

        /*** iterated state estimation ***/
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        // 迭代扩展卡尔曼滤波更新
        double solve_H_time = 0.0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);  // 预测、更新
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];
        kf.change_x_imu();
        imu_only_ready = true;

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);

        /******* Publish points *******/
        if (path_en) publish_path(pubPath);
        if (scan_pub_en && first_pub) {
            publish_frame_world(pubLaserCloudFull);
            first_pub = false;
        }
        if (scan_pub_en && scan_body_pub_en) {
            publish_frame_body(pubLaserCloudFull_body);
            publish_frame_world_local(pubLaserCloudFull_world);
        }
    }
}

void mainProcessThread() {
    ros::Rate rate(5000);
    while (ros::ok()) {
        if (flg_exit) {
            break;
        }
        mainProcess();
        rate.sleep();
    }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "robots_localization");
  ros::NodeHandle nh;

  /*** System initialization ***/
  loadConfig(nh);
  if (runtime_pos_log)
  {
    fout_pose.open(root_dir + "/log/localization.txt", std::ios::out);
  }

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);

  /*** Map initialization ***/
  // string map_pcd = root_dir + "map/map.pcd";
  std::string map_pcd;
  map_pcd = pcd_path;
  std::string infoMsg = "[Robots Localization] Load Map:" + map_pcd;
  ROS_INFO(infoMsg.c_str());
  if (pcl::io::loadPCDFile<PointType>(map_pcd, *global_map) == -1)
  {
    PCL_ERROR("Couldn't read file map.pcd\n");
    return (-1);
  }
  std::cout << "map cloud width: " << global_map->width << std::endl;
  std::cout << "map cloud height: " << global_map->height << std::endl;
  if (ikdtree.Root_Node == nullptr)
  {
    ikdtree.set_downsample_param(filter_size_map_min);
    ikdtree.Build(global_map->points);
  }
  std::cout << "KDtree built! " << std::endl;

  red_prior_T << VEC_FROM_ARRAY(red_priorT);
  blue_prior_T << VEC_FROM_ARRAY(blue_priorT);
  prior_R << MAT_FROM_ARRAY(priorR);
  std::cout << "red init T: " << red_prior_T << std::endl;
  std::cout << "blue init T: " << blue_prior_T << std::endl;
  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
  YAW_RANGE[1] = 0.35;
  YAW_RANGE[2] = 6.3;

  double epsi[23] = {0.001};
  fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS,
                    epsi);
  // 将函数地址传入kf对象中，用于接收特定于系统的模型
  // 及其差异作为一个维数变化的特征矩阵进行测量。
  // 通过一个函数（h_dyn_share_in）同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）

  /*** ROS initialization ***/

  pubLaserCloudFull =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
  pubLaserCloudFull_body =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
  pubLaserCloudFull_world =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_world", 100000);
  pubLaserCloudMap =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_map", 100000);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/odometry", 100000);
  sub_pub_imu = nh.advertise<nav_msgs::Odometry>("/odometry_imu", 100000);
  pubIMUBias = nh.advertise<geometry_msgs::Twist>("/IMU_bias", 100000);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

  signal(SIGINT, SigHandle);

  std::thread mainThread(&mainProcessThread);

  ros::Subscriber sub_pcl = p_lidar->lidar_type == AVIA
                                ? nh.subscribe(lid_topic, 2, livox_pcl_cbk)
                                : nh.subscribe(lid_topic, 2, standard_pcl_cbk);
  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  ros::Subscriber sub_reloc = nh.subscribe(reloc_topic, 200000, reloc_cbk);
  ros::Subscriber sub_mode = nh.subscribe(mode_topic, 200000, mode_cbk);

  // ros::Subscriber sub_wheel = nh.subscribe(wheel_topic, 200000, wheel_cbk);

  ros::MultiThreadedSpinner spinner(
      2);  // 启用x个spinner线程订阅话题，后续可以看看异步的AsyncSpinner以及为每个subscriber指定callback队列效果会不会好些

  spinner.spin();

  return 0;
}
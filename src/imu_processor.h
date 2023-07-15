/*
 * @Author: 陈熠琳 1016598572@qq.com
 * @Date: 2023-06-24 19:20:59
 * @LastEditors: 陈熠琳 1016598572@qq.com
 * @LastEditTime: 2023-06-24 19:24:38
 * @FilePath: /rm_sentry_localization_2023/src/imu_processor.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#pragma once

#include <common_lib.h>
#include <pcl/common/transforms.h>
#include <sensor_msgs/Imu.h>

#include <fstream>
#include <sophus/so3.hpp>

#include "ikd-Tree/ikd_Tree.h"
#include "use-ikfom.h"

#define MAX_INI_COUNT (10)

using namespace common_lib;
using namespace ikd_Tree;

const bool time_list(PointType &x, PointType &y);

class IMUProcessor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUProcessor();

  ~IMUProcessor();

  void reset();

  void set_init_pose(const V3F &poseT);

  void set_extrinsic(const V3D &transl, const M3D &rot);

  void set_extrinsic(const V3D &transl);

  void set_extrinsic(const MD(4, 4) & T);

  void set_gyr_cov(const V3D &scaler);

  void set_acc_cov(const V3D &scaler);

  void set_gyr_bias_cov(const V3D &b_g);

  void set_acc_bias_cov(const V3D &b_a);

  bool init_pose(const MeasureGroup &meas,
                 esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudXYZI::Ptr map, KD_TREE<PointType> &kdtree,
                 vector<float> &YAW_RANGE);

  void process(const MeasureGroup &meas,
               esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
               PointCloudXYZI &pcl_out);

  bool process_imu_only(const sensor_msgs::Imu::Ptr imu_data,
                        esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);

  Eigen::Matrix<double, 12, 12> Q;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;

  double first_lidar_time;

  string method;
  float res, step_size, trans_eps, eculi_eps, plane_dist;
  int max_iter;

private:
  void imu_init(const MeasureGroup &meas,
                esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);

  float init_ndt_method(PointCloudXYZI::Ptr scan, M4F &predict_pose);
  float init_icp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                        M4F &predict_pose);
  float init_ppicp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                          M4F &predict_pose);

  sensor_msgs::ImuConstPtr last_imu_, last_imu_only_;
  vector<Pose6D> IMUpose;
  V3D acc_s_last, angvel_last, acc_s_last_only, angvel_last_only;
  double last_lidar_end_time_;

  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;

  float error_min_last;
  bool find_yaw;
  M4F init_pose_curr;
  M4F init_pose_last;
  bool b_first_frame_;
  V3D mean_acc;
  V3D mean_gyr;
  mutex mtx_error;

  int init_iter_num;
  std::ofstream fout_init;

  float p_valid_proportion;
  float p_num;
  float p_valid;
};



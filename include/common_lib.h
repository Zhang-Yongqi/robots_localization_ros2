#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <robots_localization/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Eigen>

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <unordered_map>

#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;

#define USE_IKFOM

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)    // Gravaty const in GuangDong/China
#define DIM_STATE (18)   // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)  // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (1)
#define NUM_MATCH_POINTS (5)
#define MAX_MEAS_DIM (10000)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

namespace common_lib
{
typedef robots_localization::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;
typedef Matrix4d M4D;
typedef Matrix4f M4F;

#define MD(a, b) Matrix<double, (a), (b)>
#define VD(a) Matrix<double, (a), 1>
#define MF(a, b) Matrix<float, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

M3D Eye3d = M3D::Identity();
M3F Eye3f = M3F::Identity();
V3D Zero3d = V3D(0, 0, 0);
V3F Zero3f = V3F(0, 0, 0);

/**************** uwb相关变量定义 ***************/
bool USE_UWB = false;
bool opt_with_uwb = false;
string uwb_topic;
string mocap_topic;
bool log_mocap_traj, log_fusion_traj;
std::ofstream fout_mocap, fout_fusion, fout_uwb_calib;
// UWB观测更新参数
int UWB_ANCHOR_NUM = 4;
std::unordered_map<int, int> anchor_id_state_index;
bool esti_uwb_scale;
bool esti_uwb_bias;
bool esti_uwb_offset;
bool esti_uwb_anchor;
bool estimate_td;
double uwb_to_imu_td;
double td_std;
double uwb_range_std = 0.5;
double uwb_chi2_threshold;
// UWB基站初始化参数
std::vector<double> uwb2imuT;
V3D IMU_T_wrt_UWB(Zero3d);
double tag_move_threshold;
int offline_calib_data_num;
std::vector<double> offline_calib_move_3d_threshold;
V3D move_3d_threshold;
std::vector<double> offline_calib_maxmin_3d_threshold;
V3D maxmin_3d_threshold;
double near_measure_dist_threshold;
int near_measure_num_threshold;
bool constrain_bias;
double bias_limit;
int calib_data_group_num;
double consistent_threshold;
int init_bias_data_num;
std::vector<double> online_calib_move_3d_threshold;
int ransac_sample_num;
double ransac_average_error_threshold;
bool use_calibrated_anchor;
std::vector<double> calibrated_anchor1;
std::vector<double> calibrated_anchor2;
std::vector<double> calibrated_anchor3;
std::vector<double> calibrated_anchor4;
/**************************************************/

/***************** zupt相关变量定义 *****************/
bool USE_ZUPT = false;
bool opt_with_zupt = false;
double zupt_duration;
int zupt_imu_data_num;
double zupt_max_acc_std;
double zupt_max_gyr_std;
double zupt_max_gyr_median;
double zupt_gyr_std;
double zupt_vel_std;
double zupt_acc_std;
double zupt_chi2_threshold;
V3D recent_avg_acc, recent_avg_gyr;
/**************************************************/

struct UWBAnchor
{
  int id;                                                         // 基站ID
  V3D position;                                                   // 基站三维坐标
  double scale, bias;                                             // 测距尺度与偏置
  V3D last_tag_position;                                          // 上一帧tag的位置
  V3D total_dist_3d;                                              // 3轴分别积累的运动里程数
  std::vector<V3D> total_dist_3d_history;                         // 所有轨迹的3轴分别积累的运动里程数
  V3D min_coor_3d;                                                // 3轴坐标的最小值
  V3D max_coor_3d;                                                // 3轴坐标的最大值
  V3D maxmin_coor_3d;                                             // 3轴坐标的极差
  std::vector<V3D> maxmin_coor_3d_history;                        // 所有轨迹的3轴坐标的极差
  int near_measure_num;                                           // 与当前基站近距离测量帧数
  std::vector<int> near_measure_num_history;                      // 所有轨迹的与当前基站近距离测量帧数
  bool initialized;                                               // 是否已初始化
  std::vector<std::vector<std::pair<V3D, double>>> measurements;  // 移动站位置和测距数据
  int group_idx;                                                  // 当前记录第几组标定数据

  UWBAnchor() : UWBAnchor(-1)
  {
  }

  UWBAnchor(int _id)
    : id(_id)
    , position(Zero3d)
    , scale(1.0)
    , bias(0.0)
    , initialized(false)
    , last_tag_position(Zero3d)
    , total_dist_3d(Zero3d)
    , near_measure_num(0)
    , maxmin_coor_3d(Zero3d)
    , group_idx(0)
  {
    constexpr double max_val = std::numeric_limits<double>::max();
    constexpr double min_val = std::numeric_limits<double>::min();

    min_coor_3d = V3D::Constant(max_val);
    max_coor_3d = V3D::Constant(min_val);

    total_dist_3d_history.resize(calib_data_group_num);
    maxmin_coor_3d_history.resize(calib_data_group_num);
    near_measure_num_history.resize(calib_data_group_num);
    measurements.resize(calib_data_group_num);
  }

  void add_measurement(const V3D& new_tag_position, double distance)
  {
    double tag_movement = (new_tag_position - last_tag_position).norm();
    if (tag_movement > tag_move_threshold)
    {
      // 更新运动里程数
      V3D delta = new_tag_position - last_tag_position;
      total_dist_3d += delta.cwiseAbs();

      // 更新坐标极值
      min_coor_3d = min_coor_3d.cwiseMin(new_tag_position);
      max_coor_3d = max_coor_3d.cwiseMax(new_tag_position);
      maxmin_coor_3d = max_coor_3d - min_coor_3d;

      // 更新近距离测量帧数
      if (distance < near_measure_dist_threshold)
        near_measure_num += 1;

      // 更新上一帧位置
      last_tag_position = new_tag_position;

      // 添加当前测量到初始化数据中
      measurements[group_idx].emplace_back(new_tag_position, distance);
    }
  }

  bool curr_group_meet_calib_condition()
  {
    return (measurements[group_idx].size() > offline_calib_data_num) &&
           (total_dist_3d.array() > move_3d_threshold.array()).all() &&
           (maxmin_coor_3d.array() > maxmin_3d_threshold.array()).all() &&
           (near_measure_num > near_measure_num_threshold);
  }

  bool all_group_meet_calib_condition()
  {
    return group_idx >= calib_data_group_num;
  }

  bool meet_calib_bias_condition()
  {
    int total_meas = 0;
    for (int i = 0; i <= group_idx; i++)
      total_meas += measurements[i].size();
    return total_meas >= init_bias_data_num;
  }

  void reset_trajectory()
  {
    // 保存
    total_dist_3d_history[group_idx] = total_dist_3d;
    maxmin_coor_3d_history[group_idx] = maxmin_coor_3d;
    near_measure_num_history[group_idx] = near_measure_num;

    // 重置
    total_dist_3d = Zero3d;

    constexpr double max_val = std::numeric_limits<double>::max();
    constexpr double min_val = std::numeric_limits<double>::min();
    min_coor_3d = V3D::Constant(max_val);
    max_coor_3d = V3D::Constant(min_val);
    maxmin_coor_3d = Zero3d;

    near_measure_num = 0;

    // 新的轨迹
    group_idx += 1;
  }

  void reset_measurements()
  {
    group_idx = 0;

    measurements.clear();
    measurements.resize(calib_data_group_num);

    total_dist_3d_history.clear();
    total_dist_3d_history.resize(calib_data_group_num, Zero3d);

    maxmin_coor_3d_history.clear();
    maxmin_coor_3d_history.resize(calib_data_group_num, Zero3d);

    near_measure_num_history.clear();
    near_measure_num_history.resize(calib_data_group_num, 0);

    total_dist_3d = Zero3d;
    constexpr double max_val = std::numeric_limits<double>::max();
    constexpr double min_val = std::numeric_limits<double>::min();
    min_coor_3d = V3D::Constant(max_val);
    max_coor_3d = V3D::Constant(min_val);
    maxmin_coor_3d = Zero3d;
    near_measure_num = 0;
  }
};
std::unordered_map<int, UWBAnchor> anchor_map;

struct UWBObservation
{
  int anchor_id;    // 基站ID
  double distance;  // 与移动站的距离
};

struct IMU
{
  double timestamp;
  V3D acc;
  V3D gyr;
};
deque<IMU> zupt_imu_buffer;

struct MeasureGroup  // Lidar data and imu dates for the curent process
{
  MeasureGroup()
  {
    lidar_beg_time = 0.0;
    this->lidar.reset(new PointCloudXYZI());
  };
  double lidar_beg_time;
  double lidar_end_time;
  PointCloudXYZI::Ptr lidar;
  deque<sensor_msgs::Imu::ConstPtr> imu;
  deque<std::pair<double, std::vector<UWBObservation>>> uwb;  // <时间戳，当前帧所有可用基站观测>
};

// 查找当前帧UWB测量中有哪些基站已经初始化完成
std::vector<UWBObservation> get_inited_anchor_meas(const std::vector<UWBObservation>& cur_uwb_meas)
{
  std::vector<UWBObservation> inited_anchor_meas;
  for (const auto& meas : cur_uwb_meas)
  {
    int id = meas.anchor_id;
    auto it = anchor_map.find(id);
    if (it != anchor_map.end() && it->second.initialized)
    {
      inited_anchor_meas.push_back(meas);
    }
  }
  return inited_anchor_meas;
}

// 当前帧UWB测量中是否包含已初始化的基站
bool has_inited_anchor_in_cur_meas(const std::vector<UWBObservation>& cur_uwb_meas)
{
  for (const auto& meas : cur_uwb_meas)
  {
    int id = meas.anchor_id;
    auto it = anchor_map.find(id);
    if (it != anchor_map.end() && it->second.initialized)
      return true;
  }
  return false;
}

struct AnchorResidual
{
  AnchorResidual(const Eigen::Vector3d& p, double d) : p_(p), d_(d)
  {
  }

  template <typename T>
  bool operator()(const T* const anchor_params, T* residual) const
  {
    const T* pos = anchor_params;  // 基站坐标
    T b = anchor_params[3];        // 测距偏置
    T diff[3];
    for (int i = 0; i < 3; ++i)
    {
      diff[i] = T(p_[i]) - pos[i];
    }
    T dist = ceres::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
    residual[0] = T(d_) - (dist + b);
    return true;
  }

private:
  const Eigen::Vector3d p_;  // tag坐标
  const double d_;           // 测量距离
};

void nonlinear_optimize_anchor(Eigen::Vector4d& anchor_estimate, const std::vector<std::pair<V3D, double>>& anchor_meas)
{
  ceres::Problem problem;

  double anchor_params[4];
  for (int i = 0; i < 4; ++i)
  {
    anchor_params[i] = anchor_estimate[i];
  }

  for (const auto& meas : anchor_meas)
  {
    const Eigen::Vector3d& p = meas.first;
    double d = meas.second;
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<AnchorResidual, 1, 4>(new AnchorResidual(p, d));
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(uwb_range_std);
    problem.AddResidualBlock(cost_function, loss_function, anchor_params);
  }
  if (constrain_bias)
  {
    problem.SetParameterLowerBound(anchor_params, 3, 0.0);
    problem.SetParameterUpperBound(anchor_params, 3, bias_limit);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;
  fout_uwb_calib << summary.BriefReport() << std::endl;

  for (int i = 0; i < 4; ++i)
  {
    anchor_estimate[i] = anchor_params[i];
  }
}

void ransac_nonlinear_optimize_anchor(Eigen::Vector4d& anchor_estimate,
                                      const std::vector<std::pair<V3D, double>>& anchor_meas)
{
  std::vector<std::pair<V3D, double>> inlier_meas = anchor_meas;
  std::vector<std::pair<V3D, double>> outlier_meas;
  std::vector<std::pair<V3D, double>> sample_meas;
  std::vector<std::pair<V3D, double>> inlier_candidate;
  std::vector<std::pair<V3D, double>> outlier_candidate;

  Eigen::Vector4d local_estimate = Eigen::Vector4d::Zero();
  Eigen::Vector4d best_estimate = Eigen::Vector4d::Zero();

  std::vector<double> errors_inlier, errors_outlier;

  int max_iter = 200;
  double score_min = std::numeric_limits<double>::max();
  double score;

  auto time_start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < max_iter; iter++)
  {
    // 1. 随机采样测量数据
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(inlier_meas.begin(), inlier_meas.end(), std::default_random_engine(seed));
    sample_meas.assign(inlier_meas.begin(),
                       inlier_meas.begin() + std::min(ransac_sample_num, static_cast<int>(inlier_meas.size())));

    // 2. 估计模型参数
    nonlinear_optimize_anchor(local_estimate, sample_meas);

    // 3. 计算代价（平均残差）
    errors_inlier.clear();
    for (const auto& meas : inlier_meas)
    {
      const Eigen::Vector3d& p = meas.first;
      double d = meas.second;
      Eigen::Vector3d diff = p - local_estimate.head<3>();
      double dist = diff.norm();
      double residual = std::abs(d - (dist + local_estimate[3]));
      errors_inlier.push_back(residual);
    }
    score = std::accumulate(errors_inlier.begin(), errors_inlier.end(), 0.0) / errors_inlier.size();

    // 计算外点残差（如果存在外点）
    errors_outlier.clear();
    if (!outlier_meas.empty())
    {
      for (const auto& meas : outlier_meas)
      {
        const Eigen::Vector3d& p = meas.first;
        double d = meas.second;
        Eigen::Vector3d diff = p - local_estimate.head<3>();
        double dist = diff.norm();
        double residual = std::abs(d - (dist + local_estimate[3]));
        errors_outlier.push_back(residual);
      }
    }

    // 4. 更新最佳模型
    if (score < score_min)
    {
      score_min = score;
      best_estimate = local_estimate;
    }
    ROS_INFO("RANSAC iteration: %d, Inlier measurements size: %d, Score: %f", iter, inlier_meas.size(), score);
    fout_uwb_calib << "RANSAC iteration: " << iter << ", Inlier measurements size: " << inlier_meas.size()
                   << ", Score: " << score << std::endl;

    // 5. 更新内外点
    // 根据残差排序确定动态阈值（90%分位数）
    std::vector<double> temp_errors(errors_inlier);
    std::sort(temp_errors.begin(), temp_errors.end());
    int threshold_index = static_cast<int>(temp_errors.size() * 0.9);
    double dynamic_threshold =
        (threshold_index > 0 && threshold_index < temp_errors.size()) ? temp_errors[threshold_index] : uwb_range_std;

    // 从内点中移除残差大的点到外点候选
    for (size_t i = 0; i < errors_inlier.size();)
    {
      if (errors_inlier[i] > dynamic_threshold)
      {
        outlier_candidate.push_back(inlier_meas[i]);
        inlier_meas.erase(inlier_meas.begin() + i);
        errors_inlier.erase(errors_inlier.begin() + i);
      }
      else
      {
        ++i;
      }
    }
    // 从外点中回收残差小的点到内点候选
    for (size_t i = 0; i < errors_outlier.size();)
    {
      if (errors_outlier[i] < dynamic_threshold)
      {
        inlier_candidate.push_back(outlier_meas[i]);
        outlier_meas.erase(outlier_meas.begin() + i);
        errors_outlier.erase(errors_outlier.begin() + i);
      }
      else
      {
        ++i;
      }
    }

    // 更新内外点集合
    if (!inlier_candidate.empty())
    {
      inlier_meas.insert(inlier_meas.end(), inlier_candidate.begin(), inlier_candidate.end());
    }
    if (!outlier_candidate.empty())
    {
      outlier_meas.insert(outlier_meas.end(), outlier_candidate.begin(), outlier_candidate.end());
    }

    // 6. 终止条件
    if (score < ransac_average_error_threshold || inlier_meas.size() < static_cast<size_t>(ransac_sample_num * 0.8))
    {
      std::cout << "Terminated early: Score = " << score << ", Inlier size = " << inlier_meas.size() << std::endl;
      fout_uwb_calib << "Terminated early: Score = " << score << ", Inlier size = " << inlier_meas.size() << std::endl;
      break;
    }

    sample_meas.clear();
    inlier_candidate.clear();
    outlier_candidate.clear();
    errors_inlier.clear();
    errors_outlier.clear();
  }

  // 使用所有内点进行最终优化
  if (!inlier_meas.empty())
  {
    nonlinear_optimize_anchor(best_estimate, inlier_meas);
  }

  anchor_estimate = best_estimate;

  auto time_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  std::cout << "Time for RANSAC: " << duration << " ms" << std::endl;
  fout_uwb_calib << "Time for RANSAC: " << duration << " ms" << std::endl;
}

bool try_to_initialize_anchor(UWBAnchor& uwb_anchor)
{
  if (!uwb_anchor.all_group_meet_calib_condition())
    return false;

  for (int i = 0; i < calib_data_group_num; i++)
  {
    ROS_INFO("traj[%d] data_num: %d, total_dist_3d: %f %f %f, maxmin_coor_3d: %f, %f, %f, near_measure_num: %d", i,
             uwb_anchor.measurements[i].size(), uwb_anchor.total_dist_3d_history[i][0],
             uwb_anchor.total_dist_3d_history[i][1], uwb_anchor.total_dist_3d_history[i][2],
             uwb_anchor.maxmin_coor_3d_history[i][0], uwb_anchor.maxmin_coor_3d_history[i][1],
             uwb_anchor.maxmin_coor_3d_history[i][2], uwb_anchor.near_measure_num_history[i]);
    fout_uwb_calib << "traj[" << i << "] data_num: " << uwb_anchor.measurements[i].size()
                   << ", total_dist_3d: " << uwb_anchor.total_dist_3d_history[i][0] << ", "
                   << uwb_anchor.total_dist_3d_history[i][1] << ", " << uwb_anchor.total_dist_3d_history[i][2]
                   << ", maxmin_coor_3d: " << uwb_anchor.maxmin_coor_3d_history[i][0] << ", "
                   << uwb_anchor.maxmin_coor_3d_history[i][1] << ", " << uwb_anchor.maxmin_coor_3d_history[i][2]
                   << ", near_measure_num : " << uwb_anchor.near_measure_num_history[i] << std::endl;
  }

  std::vector<Eigen::Vector4d> anchor_estimates(calib_data_group_num, Eigen::Vector4d::Zero());
  for (int i = 0; i < calib_data_group_num; i++)
  {
    auto anchor_meas = uwb_anchor.measurements[i];
    ransac_nonlinear_optimize_anchor(anchor_estimates[i], anchor_meas);
  }

  if (calib_data_group_num == 1)
  {
    uwb_anchor.position = anchor_estimates[0].head<3>();
    uwb_anchor.bias = anchor_estimates[0][3];
    uwb_anchor.initialized = true;
    uwb_anchor.reset_measurements();
  }
  else
  {
    // 标定一致性验证
    bool is_consistent = true;
    bool last_group_logged = false;
    for (int i = 0; i < calib_data_group_num - 1; i++)
    {
      ROS_INFO("uwb anchor %d, traj[%d] position: [%f, %f, %f], bias: %f", uwb_anchor.id, i, anchor_estimates[i][0],
               anchor_estimates[i][1], anchor_estimates[i][2], anchor_estimates[i][3]);
      fout_uwb_calib << "*********************************************************************" << std::endl;
      fout_uwb_calib << "uwb anchor " << uwb_anchor.id << ", traj[" << i << "] position: [" << anchor_estimates[i][0]
                     << ", " << anchor_estimates[i][1] << ", " << anchor_estimates[i][2]
                     << "], bias: " << anchor_estimates[i][3] << std::endl;
      fout_uwb_calib << "*********************************************************************" << std::endl;
      for (int j = i + 1; j < calib_data_group_num; j++)
      {
        if (j == calib_data_group_num - 1 && !last_group_logged)
        {
          ROS_INFO("uwb anchor %d, traj[%d] position: [%f, %f, %f], bias: %f", uwb_anchor.id, j, anchor_estimates[j][0],
                   anchor_estimates[j][1], anchor_estimates[j][2], anchor_estimates[j][3]);
          fout_uwb_calib << "*********************************************************************" << std::endl;
          fout_uwb_calib << "uwb anchor " << uwb_anchor.id << ", traj[" << j << "] position: ["
                         << anchor_estimates[j][0] << ", " << anchor_estimates[j][1] << ", " << anchor_estimates[j][2]
                         << "], bias: " << anchor_estimates[j][3] << std::endl;
          fout_uwb_calib << "*********************************************************************" << std::endl;
          last_group_logged = true;
        }
        V3D diff = anchor_estimates[i].head<3>() - anchor_estimates[j].head<3>();
        ROS_INFO("traj[%d]-[%d] calib diff: %f, %f, %f", i, j, diff.x(), diff.y(), diff.z());
        fout_uwb_calib << "traj[" << i << "]-[" << j << "] calib diff: " << diff.x() << ", " << diff.y() << ", "
                       << diff.z() << std::endl;
        if (diff.cwiseAbs().maxCoeff() > consistent_threshold)
        {
          is_consistent = false;
          ROS_WARN("Inconsistent anchor estimates: diff > 0.5m between group %d and %d", i, j);
          fout_uwb_calib << "Inconsistent anchor estimates: diff > 0.5m between group " << i << " and " << j << "\n\n"
                         << std::endl;
          break;
        }
      }
      if (!is_consistent)
        break;
    }
    if (is_consistent)
    {
      // 计算平均坐标及bias
      V3D avg_position = Zero3d;
      double avg_bias = 0.0;
      for (const auto& estimate : anchor_estimates)
      {
        avg_position += estimate.head<3>();
        avg_bias += estimate[3];
      }
      avg_position /= calib_data_group_num;
      avg_bias /= calib_data_group_num;

      uwb_anchor.position = avg_position;
      uwb_anchor.bias = avg_bias;
      uwb_anchor.initialized = true;
      uwb_anchor.reset_measurements();
    }
    else
    {
      uwb_anchor.reset_measurements();
      return false;
    }
  }
  return true;
}

bool try_to_initialize_bias(UWBAnchor& uwb_anchor)
{
  if (!uwb_anchor.meet_calib_bias_condition())
    return false;

  std::vector<std::pair<V3D, double>> anchor_meas;
  for (int i = 0; i <= uwb_anchor.group_idx; i++)
  {
    anchor_meas.insert(anchor_meas.end(), uwb_anchor.measurements[i].begin(), uwb_anchor.measurements[i].end());
  }
  V3D anchor_position = uwb_anchor.position;

  std::vector<double> bias_meas(anchor_meas.size());
  for (int i = 0; i < anchor_meas.size(); ++i)
  {
    const Eigen::Vector3d& p = anchor_meas[i].first;
    double dist_meas = anchor_meas[i].second;
    Eigen::Vector3d diff = p - anchor_position;
    double dist_pred = diff.norm();
    bias_meas[i] = dist_meas - dist_pred;
  }

  std::sort(bias_meas.begin(), bias_meas.end());

  // IQR剔除离群值
  int n = bias_meas.size();
  int q1_idx = static_cast<int>(n / 4);
  int q3_idx = static_cast<int>((3 * n) / 4);
  double q1 = bias_meas[q1_idx];
  double q3 = bias_meas[q3_idx];
  ROS_INFO("q1: %f, q3: %f, q0: %f, q4: %f", q1, q3, bias_meas[0], bias_meas.back());
  double iqr = q3 - q1;
  double lower_bound = q1 - 1.5 * iqr;
  double upper_bound = q3 + 1.5 * iqr;
  std::vector<double> filtered_bias;
  for (double b : bias_meas)
  {
    if (b >= lower_bound && b <= upper_bound)
    {
      filtered_bias.emplace_back(b);
    }
  }

  if (filtered_bias.empty())
  {
    ROS_WARN("filtered_bias empty, calib bias failed!!!");
    uwb_anchor.reset_measurements();
    return false;
  }

  double sum = 0.0;
  for (double b : filtered_bias)
  {
    sum += b;
  }

  uwb_anchor.bias = sum / filtered_bias.size();
  uwb_anchor.initialized = true;
  uwb_anchor.reset_measurements();
  return true;
}

// Get median value
template <typename T>
static T get_median(std::vector<T>& seq)
{
  if (seq.size() == 0)
  {
    return static_cast<T>(0);
  }
  std::vector<T> temp = seq;
  typename std::vector<T>::iterator it = temp.begin() + std::floor(temp.size() / 2);
  std::nth_element(temp.begin(), it, temp.end());
  return *it;
}

// Get average value
template <typename T>
static double get_average(std::vector<T>& seq)
{
  if (seq.size() == 0)
  {
    return 0.0;
  }
  double sum = 0.0;
  for (size_t i = 0; i < seq.size(); i++)
  {
    sum += static_cast<double>(seq[i]);
  }
  return sum / static_cast<double>(seq.size());
}

// Get standard deviation value
template <typename T>
static double get_standard_deviation(std::vector<T>& seq, const T& ref)
{
  if (seq.size() == 0)
  {
    return 0.0;
  }
  double sum2 = 0.0;
  for (size_t i = 0; i < seq.size(); i++)
  {
    sum2 += pow(seq[i] - ref, 2);
  }
  return sqrt(sum2 / static_cast<double>(seq.size()));
}

bool check_zupt(const deque<IMU>& imu_data, V3D& average_acc, V3D& average_gyr)
{
  if (imu_data.size() < zupt_imu_data_num)
    return false;

  std::vector<double> acc[3], gyr[3];

  for (int i = 0; i < imu_data.size(); i++)
  {
    const IMU& imu = imu_data[i];
    for (int j = 0; j < 3; j++)
    {
      acc[j].push_back(imu.acc[j]);
      gyr[j].push_back(imu.gyr[j]);
    }
  }
  double median_acc[3] = { 0 }, median_gyr[3] = { 0 };
  double std_acc[3] = { 0 }, std_gyr[3] = { 0 };
  double avg_acc[3] = { 0 }, avg_gyr[3] = { 0 };
  for (int i = 0; i < 3; i++)
  {
    median_acc[i] = get_median(acc[i]);
    median_gyr[i] = get_median(gyr[i]);
    std_acc[i] = get_standard_deviation(acc[i], median_acc[i]);
    std_gyr[i] = get_standard_deviation(gyr[i], median_gyr[i]);
    avg_acc[i] = get_average(acc[i]);
    avg_gyr[i] = get_average(gyr[i]);
  }
  average_acc << avg_acc[0], avg_acc[1], avg_acc[2];
  average_gyr << avg_gyr[0], avg_gyr[1], avg_gyr[2];

  //   ROS_INFO("std_acc: %f, %f, %f, std_gyr: %f, %f, %f, median_gyr: %f, %f, %f", std_acc[0], std_acc[1], std_acc[2],
  //            std_gyr[0], std_gyr[1], std_gyr[2], median_gyr[0], median_gyr[1], median_gyr[2]);
  for (int i = 0; i < 3; i++)
  {
    if (std_acc[i] > zupt_max_acc_std)
      return false;
    if (std_gyr[i] > zupt_max_gyr_std)
      return false;
    if (fabs(median_gyr[i]) > zupt_max_gyr_median)
      return false;
  }

  return true;
}

void interpolate_imu(sensor_msgs::Imu::ConstPtr head_imu, sensor_msgs::Imu::ConstPtr tail_imu, double mid_time,
                     sensor_msgs::Imu& mid_imu)
{
  double t_head = head_imu->header.stamp.toSec();
  double t_tail = tail_imu->header.stamp.toSec();

  double alpha = (mid_time - t_head) / (t_tail - t_head);

  mid_imu.header.stamp = ros::Time(mid_time);
  mid_imu.header.frame_id = head_imu->header.frame_id;

  mid_imu.angular_velocity.x =
      head_imu->angular_velocity.x + alpha * (tail_imu->angular_velocity.x - head_imu->angular_velocity.x);
  mid_imu.angular_velocity.y =
      head_imu->angular_velocity.y + alpha * (tail_imu->angular_velocity.y - head_imu->angular_velocity.y);
  mid_imu.angular_velocity.z =
      head_imu->angular_velocity.z + alpha * (tail_imu->angular_velocity.z - head_imu->angular_velocity.z);

  mid_imu.linear_acceleration.x =
      head_imu->linear_acceleration.x + alpha * (tail_imu->linear_acceleration.x - head_imu->linear_acceleration.x);
  mid_imu.linear_acceleration.y =
      head_imu->linear_acceleration.y + alpha * (tail_imu->linear_acceleration.y - head_imu->linear_acceleration.y);
  mid_imu.linear_acceleration.z =
      head_imu->linear_acceleration.z + alpha * (tail_imu->linear_acceleration.z - head_imu->linear_acceleration.z);
}

struct StatesGroup
{
  StatesGroup()
  {
    this->rot_end = M3D::Identity();
    this->pos_end = Zero3d;
    this->vel_end = Zero3d;
    this->bias_g = Zero3d;
    this->bias_a = Zero3d;
    this->gravity = Zero3d;
    this->cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
    this->cov.block<9, 9>(9, 9) = MD(9, 9)::Identity() * 0.00001;
  };

  StatesGroup(const StatesGroup& b)
  {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->cov = b.cov;
  };

  StatesGroup& operator=(const StatesGroup& b)
  {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->cov = b.cov;
    return *this;
  };

  StatesGroup operator+(const Matrix<double, DIM_STATE, 1>& state_add)
  {
    StatesGroup a;
    a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
    a.cov = this->cov;
    return a;
  };

  StatesGroup& operator+=(const Matrix<double, DIM_STATE, 1>& state_add)
  {
    this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->vel_end += state_add.block<3, 1>(6, 0);
    this->bias_g += state_add.block<3, 1>(9, 0);
    this->bias_a += state_add.block<3, 1>(12, 0);
    this->gravity += state_add.block<3, 1>(15, 0);
    return *this;
  };

  Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b)
  {
    Matrix<double, DIM_STATE, 1> a;
    M3D rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0) = Log(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(15, 0) = this->gravity - b.gravity;
    return a;
  };

  void resetpose()
  {
    this->rot_end = M3D::Identity();
    this->pos_end = Zero3d;
    this->vel_end = Zero3d;
  }

  M3D rot_end;                               // the estimated attitude (rotation matrix) at the end lidar
                                             // point
  V3D pos_end;                               // the estimated position at the end lidar point (world frame)
  V3D vel_end;                               // the estimated velocity at the end lidar point (world frame)
  V3D bias_g;                                // gyroscope bias
  V3D bias_a;                                // accelerator bias
  V3D gravity;                               // the estimated gravity acceleration
  Matrix<double, DIM_STATE, DIM_STATE> cov;  // states covariance
};

template <typename T>
T rad2deg(T radians)
{
  return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees)
{
  return degrees * PI_M / 180.0;
}

template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1>& a, const Matrix<T, 3, 1>& g, const Matrix<T, 3, 1>& v,
                const Matrix<T, 3, 1>& p, const Matrix<T, 3, 3>& R)
{
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++)
  {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++)
      rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return move(rot_kp);
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template <typename T>
bool esti_normvector(Matrix<T, 3, 1>& normvec, const PointVector& point, const T& threshold, const int& point_num)
{
  MatrixXf A(point_num, 3);
  MatrixXf b(point_num, 1);
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < point_num; j++)
  {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }
  normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < point_num; j++)
  {
    if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
    {
      return false;
    }
  }

  normvec.normalize();
  return true;
}

float calc_dist(PointType p1, PointType p2)
{
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
  return d;
}
/**
 * 拟合法向量，返回归一化法向量
 * @tparam T
 * @param pca_result 输出法向量
 * @param point 近邻点
 * @param threshold 平面距离原点阈值
 * @return
 */
template <typename T>
bool esti_plane(Matrix<T, 4, 1>& pca_result, const PointVector& point, const T& threshold)
{
  Matrix<T, NUM_MATCH_POINTS, 3> A;
  Matrix<T, NUM_MATCH_POINTS, 1> b;
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < NUM_MATCH_POINTS; j++)
  {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);  // 求解Ax=b，即ax+by+cz=1
  // 由于是截距式平面方程，(a,b,c)就是一个法向量

  T n = normvec.norm();  // 向量的范数，长度
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;

  for (int j = 0; j < NUM_MATCH_POINTS; j++)
  {
    if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) >
        threshold)
    {
      return false;
    }
  }
  return true;
}

inline V3F rotationToEulerAngles(const M3F& R)
{
  V3F n = R.col(0);
  V3F o = R.col(1);
  V3F a = R.col(2);

  // 这里各轴角范围仍为-pi~pi？
  V3F rpy(3);
  float y = atan2(n(1), n(0));
  float p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  float r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));

  rpy(0) = y;
  rpy(1) = p;
  rpy(2) = r;

  return rpy;
}
}  // namespace common_lib
#endif
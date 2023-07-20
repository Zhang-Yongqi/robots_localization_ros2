#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#include "imu_processor.h"

using namespace common_lib;
using namespace ikd_Tree;

const bool
time_list(PointType &x, PointType &y)
{
  return (x.curvature < y.curvature);
};

IMUProcessor::IMUProcessor() : b_first_frame_(true)
{
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  error_min_last = 1000000.0;
  find_yaw = false;
  init_pose_curr = M4F::Zero();
  init_pose_last = M4F::Zero();
  fout_init.open((string)ROOT_DIR + "/log/initialization.txt", std::ios::out);
  p_valid_proportion = 0.0f;
  p_num = 0.0f;
  p_valid = 0.0f;
}

IMUProcessor::~IMUProcessor() {}

void IMUProcessor::reset()
{
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  acc_s_last = Zero3d;
  angvel_last = Zero3d;
  init_iter_num = 1;
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  last_imu_only_.reset(new sensor_msgs::Imu());
  Q = process_noise_cov();
  p_valid_proportion = 0.0f;
  p_num = 0.0f;
  p_valid = 0.0f;
  b_first_frame_ = true;

  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
}

void IMUProcessor::set_init_pose(const V3F &poseT)
{
  init_pose_last.block<3, 3>(0, 0) = Eye3f;
  init_pose_last.block<3, 1>(0, 3) = poseT;
}

void IMUProcessor::set_extrinsic(const MD(4, 4) & T)
{
  Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
  Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void IMUProcessor::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void IMUProcessor::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void IMUProcessor::set_gyr_cov(const V3D &scaler) { cov_gyr_scale = scaler; }

void IMUProcessor::set_acc_cov(const V3D &scaler) { cov_acc_scale = scaler; }

void IMUProcessor::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void IMUProcessor::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

float IMUProcessor::init_ndt_method(PointCloudXYZI::Ptr scan,
                                    M4F &predict_pose)
{
  return 0.0;
}

float IMUProcessor::init_icp_method(KD_TREE<PointType> &kdtree,
                                    PointCloudXYZI::Ptr scan,
                                    M4F &predict_pose)
{
  PointCloudXYZI::Ptr trans_cloud(new PointCloudXYZI());
  int knn_num = 1;
  M3F rotation_matrix;
  V3F translation;
  rotation_matrix = predict_pose.block<3, 3>(0, 0);
  translation = predict_pose.block<3, 1>(0, 3);
  float whole_dist = 0.0;

  for (int iter = 0; iter < max_iter; iter++)
  {
    pcl::transformPointCloud(*scan, *trans_cloud, predict_pose);
    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> b;
    H.setZero();
    b.setZero();

    whole_dist = 0.0;
    int point_num = trans_cloud->size();
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < point_num; i++)
    {
      auto ori_point = scan->points[i];
      // pcl::isFinite()用来检查点云中是否有NaN值
      if (!pcl::isFinite(ori_point))
        continue;
      auto trans_point = trans_cloud->points[i];
      std::vector<float> dist;
      // Eigen::aligned_allocator<PointType> 是一个内存分配器，它用于为存储在
      // std::vector 中的 PointType 对象分配内存。
      std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
      // dist是临近点到搜索点trans_point的距离
      kdtree.Nearest_Search(trans_point, knn_num, points_near, dist);

      Eigen::Vector3f closest_point =
          Eigen::Vector3f(points_near[0].x, points_near[0].y, points_near[0].z);
      // 临近点距离的向量
      Eigen::Vector3f error_dist =
          Eigen::Vector3f(trans_point.x, trans_point.y, trans_point.z) -
          closest_point;
      whole_dist += (trans_point.x - closest_point[0]) *
                    (trans_point.x - closest_point[0]),
          (trans_point.y - closest_point[1]) *
              (trans_point.y - closest_point[1]),
          (trans_point.z - closest_point[2]) *
              (trans_point.z - closest_point[2]);

      Eigen::Matrix<float, 3, 6> J(Eigen::Matrix<float, 3, 6>::Zero());
      J.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
      J.block<3, 3>(0, 3) =
          -rotation_matrix * Sophus::SO3f::hat(Eigen::Vector3f(
                                 ori_point.x, ori_point.y, ori_point.z));
      H += J.transpose() * J;
      b += -J.transpose() * error_dist;
    }

    // H.ldlt()返回一个对角线矩阵，它是用H的一个特征分解所得，solve(b)方法调用了对角线矩阵的求逆
    // H.ldlt().solve(b)是通过使用特征分解求线性方程组Hx=b
    Eigen::Matrix<float, 6, 1> delta_x = H.ldlt().solve(b);
    translation += delta_x.block<3, 1>(0, 0);
    rotation_matrix *= Sophus::SO3f::exp(delta_x.block<3, 1>(3, 0)).matrix();
    predict_pose.block<3, 3>(0, 0) = rotation_matrix;
    predict_pose.block<3, 1>(0, 3) = translation;
  }

  pcl::transformPointCloud(*scan, *trans_cloud, predict_pose);
  p_num = (float)trans_cloud->size();
  p_valid = 0.0;
  for (size_t i = 0; i < trans_cloud->size(); i++)
  {
    auto ori_point = scan->points[i];
    if (!pcl::isFinite(ori_point))
      continue;
    auto trans_point = trans_cloud->points[i];
    std::vector<float> dist;
    std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
    kdtree.Nearest_Search(trans_point, 1, points_near, dist);
    Eigen::Vector3f closest_point =
        Eigen::Vector3f(points_near[0].x, points_near[0].y, points_near[0].z);
    float p_dist =
        (trans_point.x - closest_point[0]) *
            (trans_point.x - closest_point[0]) +
        (trans_point.y - closest_point[1]) *
            (trans_point.y - closest_point[1]) +
        (trans_point.z - closest_point[2]) * (trans_point.z - closest_point[2]);
    if (p_dist < 1.0)
    {
      p_valid += 1.0;
    }
  }
  p_valid_proportion = p_valid / p_num;

  fout_init << "whole dist: " << whole_dist << std::endl;
  std::cout << "whole dist: " << whole_dist << std::endl;
  return whole_dist;
}

float IMUProcessor::init_ppicp_method(KD_TREE<PointType> &kdtree,
                                      PointCloudXYZI::Ptr scan,
                                      M4F &predict_pose)
{
  PointCloudXYZI::Ptr trans_cloud(new PointCloudXYZI());
  M3F rotation_matrix;
  V3F translation;
  rotation_matrix = predict_pose.block<3, 3>(0, 0);
  translation = predict_pose.block<3, 1>(0, 3);
  float whole_dist = 0.0;

  for (int iter = 0; iter < max_iter; iter++)
  {
    pcl::transformPointCloud(*scan, *trans_cloud, predict_pose);
    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> b;
    H.setZero();
    b.setZero();

    whole_dist = 0.0;
    int point_num = trans_cloud->size();
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < point_num; i++)
    {
      auto ori_point = scan->points[i];
      if (!pcl::isFinite(ori_point))
        continue;
      auto trans_point = trans_cloud->points[i];
      std::vector<float> dist;
      std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
      kdtree.Nearest_Search(trans_point, NUM_MATCH_POINTS, points_near, dist);

      VF(4)
      abcd;
      esti_plane(abcd, points_near, plane_dist);
      float error_dist = abcd[0] * trans_point.x + abcd[1] * trans_point.y +
                         abcd[2] * trans_point.z + abcd[3];
      if (error_dist < 0.0)
      {
        mtx_error.lock();
        whole_dist += -error_dist;
        mtx_error.unlock();
      }
      else
      {
        mtx_error.lock();
        whole_dist += error_dist;
        mtx_error.unlock();
      }

      Eigen::Matrix<float, 1, 6> J(Eigen::Matrix<float, 1, 6>::Zero());
      J.block<1, 3>(0, 0) << abcd[0], abcd[1], abcd[2];
      Eigen::Matrix<float, 3, 3> tmp =
          -rotation_matrix * Sophus::SO3f::hat(Eigen::Vector3f(
                                 ori_point.x, ori_point.y, ori_point.z));
      J.block<1, 3>(0, 3) << abcd[0] * tmp(0, 0) + abcd[1] * tmp(1, 0) +
                                 abcd[2] * tmp(2, 0),
          abcd[0] * tmp(0, 1) + abcd[1] * tmp(1, 1) + abcd[2] * tmp(2, 1),
          abcd[0] * tmp(0, 2) + abcd[1] * tmp(1, 2) + abcd[2] * tmp(2, 2);

      H += J.transpose() * J;
      b += -J.transpose() * error_dist;
    }

    Eigen::Matrix<float, 6, 1> delta_x = H.ldlt().solve(b);
    translation += delta_x.block<3, 1>(0, 0);
    rotation_matrix *= Sophus::SO3f::exp(delta_x.block<3, 1>(3, 0)).matrix();
    predict_pose.block<3, 3>(0, 0) = rotation_matrix;
    predict_pose.block<3, 1>(0, 3) = translation;

    if (iter == max_iter - 1) {
      p_num = (float)trans_cloud->size();
      p_valid = 0.0;
      for (size_t i = 0; i < trans_cloud->size(); i++) {
        auto ori_point = scan->points[i];
        if (!pcl::isFinite(ori_point)) continue;
        auto trans_point = trans_cloud->points[i];
        std::vector<float> dist;
        std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
        kdtree.Nearest_Search(trans_point, 1, points_near, dist);
        Eigen::Vector3f closest_point = Eigen::Vector3f(points_near[0].x, points_near[0].y, points_near[0].z);
        float p_dist = (trans_point.x - closest_point[0]) * (trans_point.x - closest_point[0]) +
                       (trans_point.y - closest_point[1]) * (trans_point.y - closest_point[1]) +
                       (trans_point.z - closest_point[2]) * (trans_point.z - closest_point[2]);
        if (p_dist < 1.0) {
            p_valid += 1.0;
        }
      }
      p_valid_proportion = p_valid / p_num;
    }
  }

  fout_init << "whole dist: " << whole_dist << std::endl;
  std::cout << "whole dist: " << whole_dist << std::endl;
  return whole_dist;
}

/** 1. initialize the gravity, gyro bias, acc and gyro covariance
 ** 2. normalize the acceleration measurenments to unit gravity **/
void IMUProcessor::imu_init(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  V3D cur_acc, cur_gyr;
  if (b_first_frame_)
  {
    reset();
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;
    cov_acc = cov_acc * (N - 1.0) / N +
              (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / (N - 1.0);
    cov_gyr = cov_gyr * (N - 1.0) / N +
              (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / (N - 1.0);
    //.cwiseProduct()为对应系数相乘
    // https://blog.csdn.net/weixin_44479136/article/details/90510374
    // 均值 & 方差迭代计算公式
    N++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
  // 从common_lib.h中拿到重力，并与加速度测量均值的单位重力求出S2的旋转矩阵类型的重力加速度

  init_state.bg = mean_gyr; // 角速度测量均值作为陀螺仪偏差
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  init_state.pos = vect3(init_pose_curr.block<3, 1>(0, 3).cast<double>());
  init_state.rot = SO3(init_pose_curr.block<3, 3>(0, 0).cast<double>());
  init_state.vel = vect3(V3D(0.0, 0.0, 0.0));
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;      // 外参R
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;  // 外参t
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001; // 陀螺仪偏差
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;  // 加速度计偏差
  init_P(21, 21) = init_P(22, 22) = 0.00001;                 // 重力
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
  last_imu_only_ = meas.imu.back();
}

bool IMUProcessor::init_pose(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
    PointCloudXYZI::Ptr map, KD_TREE<PointType> &kdtree,
    vector<float> &YAW_RANGE)
{
  std::cout << "begin to init pose " << std::endl;
  if (meas.imu.empty())
  {
    std::cout << "no imu meas" << std::endl;
    return false;
  };
  ROS_ASSERT(meas.lidar != nullptr);

  double t1 = omp_get_wtime();
  float error_min = 1000000.0, p_valid_proportion_max = 0.0;
  M4F prior_with_min_error = M4F::Zero();
  if (error_min_last > 100 && find_yaw) {
    M4F prior_with_yaw = init_pose_last;
    error_min_last = init_ppicp_method(kdtree, meas.lidar, prior_with_yaw);
    prior_with_min_error = prior_with_yaw;
    p_valid_proportion_max = p_valid_proportion;
  } else if (!find_yaw) {
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)((YAW_RANGE[2] - YAW_RANGE[0]) / YAW_RANGE[1]); i++) {
      float yaw = YAW_RANGE[0] + i * YAW_RANGE[1];
      // std::cout << "iter: " << i << ", yaw: " << yaw << std::endl;

      float error = 0.0;
      M4F prior_with_yaw = M4F::Zero();
      M3F rotation_yaw = M3F::Zero();
      rotation_yaw << std::cos(yaw), -std::sin(yaw), 0.0, std::sin(yaw), std::cos(yaw), 0.0, 0.0, 0.0, 1.0;
      prior_with_yaw.block<3, 1>(0, 3) = init_pose_last.block<3, 1>(0, 3);
      prior_with_yaw.block<3, 3>(0, 0) = rotation_yaw * init_pose_last.block<3, 3>(0, 0);

      // if (method == "NDT") {
      //   error = init_ndt_method(meas.lidar, prior_with_yaw);
      // }
      // else if (method == "ICP") {
      //   error = init_icp_method(kdtree, meas.lidar, prior_with_yaw);
      // }
      // else if (method == "PPICP") {
      error = init_ppicp_method(kdtree, meas.lidar, prior_with_yaw);
      // }
      // else
      // {
      //   std::cerr << "Not valid method!" << std::endl;
      //   return false;
      // }
#ifdef MP_EN
#pragma omp critical
#endif
      {
        if (error < error_min) {
            error_min = error;
            prior_with_min_error = prior_with_yaw;
            p_valid_proportion_max = p_valid_proportion;
        }
      }
      // if (error_min < 100)
      // {
      //   break;
      // }
    }
    find_yaw = true;
    error_min_last = error_min;
  }
  init_pose_curr = prior_with_min_error;
  double t2 = omp_get_wtime();
  fout_init << "Init align time cost " << t2 - t1 << "s. " << std::endl
            << "Current pos:  " << std::endl
            << init_pose_curr.block<3, 1>(0, 3) << std::endl
            << "Current rot:  " << std::endl
            << init_pose_curr.block<3, 3>(0, 0) << std::endl
            << "Last pos:  " << std::endl
            << init_pose_last.block<3, 1>(0, 3) << std::endl
            << "Last rot:  " << std::endl
            << init_pose_last.block<3, 3>(0, 0) << std::endl
            << std::endl;
  std::cout << "Init align time cost " << t2 - t1 << "s. " << std::endl
            << "Current pos:  " << std::endl
            << init_pose_curr.block<3, 1>(0, 3) << std::endl
            << "Current rot:  " << std::endl
            << init_pose_curr.block<3, 3>(0, 0) << std::endl
            << "Last pos:  " << std::endl
            << init_pose_last.block<3, 1>(0, 3) << std::endl
            << "Last rot:  " << std::endl
            << init_pose_last.block<3, 3>(0, 0) << std::endl
            << std::endl;

  V3F delta_rvec, delta_tvec;
  delta_rvec =
      rotationToEulerAngles(init_pose_curr.block<3, 3>(0, 0) *
                            init_pose_last.block<3, 3>(0, 0).inverse());
  delta_tvec =
      init_pose_curr.block<3, 1>(0, 3) - init_pose_last.block<3, 1>(0, 3);
  fout_init << "delta_tvec: " << delta_tvec.norm() << ", "
            << "delta_rvec: " << delta_rvec.norm() << std::endl;
  if (delta_tvec.norm() < 0.1 && delta_rvec.norm() < 0.1 && init_pose_curr(0, 3) < 30.0 &&
      init_pose_curr(0, 3) > -2.0 && init_pose_curr(1, 3) < 17.0 && init_pose_curr(1, 3) > -2.0 &&
      init_pose_curr(2, 3) < 5.0 && init_pose_curr(2, 3) > -2.0 && p_valid_proportion_max > 0.5) {
    /// The very first lidar frame
    imu_init(meas, kf_state, init_iter_num);
    // last_imu_ = meas.imu.back();
    // last_imu_only_ = meas.imu.back(); imu_init里赋值过了

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO(
          "Initialization Done: pos: %.4f %.4f %.4f"
          "Gravity: %.4f %.4f %.4f %.4f; "
          "mean_acc: %.4f %.4f %.4f; "
          "mean_gyr: %.4f %.4f %.4f; "
          "bias_acc covariance: %.4f %.4f %.4f; "
          "bias_gyr covariance: %.4f %.4f %.4f; "
          "acc covarience: %.8f %.8f %.8f; "
          "gyr covarience: %.8f %.8f %.8f",
          imu_state.pos[0], imu_state.pos[1], imu_state.pos[2],
          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2],
          mean_acc.norm(), mean_acc[0], mean_acc[1], mean_acc[2], mean_gyr[0],
          mean_gyr[1], mean_gyr[2], cov_bias_acc[0], cov_bias_acc[1],
          cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2],
          cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
          cov_gyr[2]);
      fout_init << "Initialization Done: pos:" << imu_state.pos[0] << ", "
                << imu_state.pos[1] << ", " << imu_state.pos[2] << std::endl
                << "Gravity: " << imu_state.grav[0] << ", " << imu_state.grav[1] << ", "
                << imu_state.grav[2] << std::endl
                << "mean_acc: " << mean_acc[0] << ", "
                << mean_acc[1] << ", " << mean_acc[2] << std::endl
                << "mean_gyr: " << mean_gyr[0] << ", " << mean_gyr[1] << ", " << mean_gyr[2] << std::endl
                << "bias_acc covariance: " << cov_bias_acc[0] << ", " << cov_bias_acc[1]
                << ", " << cov_bias_acc[2] << std::endl
                << "bias_gyr covariance: " << cov_bias_gyr[0] << ", " << cov_bias_gyr[1] << ", " << cov_bias_gyr[2]
                << std::endl
                << "acc covarience: " << cov_acc[0] << ", " << cov_acc[1]
                << ", " << cov_acc[2] << std::endl
                << "gyr covarience: " << cov_gyr[0]
                << ", " << cov_gyr[1] << ", " << cov_gyr[2] << std::endl;
      return true;
    }
  }
  std::cout << "init pose last change" << std::endl;
  init_pose_last = init_pose_curr;
  p_valid_proportion_max = 0.0f;
  p_valid_proportion = 0.0f;
  p_num = 0.0f;
  p_valid = 0.0f;
  return false;
}

// 正向传播 反向传播 去畸变
void IMUProcessor::process(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
    PointCloudXYZI &pcl_out)
{
  if (meas.imu.empty())
  {
    return;
  };
  ROS_ASSERT(meas.lidar != nullptr);

  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout << "[IMU Process]:Process " << pcl_out.size() << " lidar points from
  // "
  //      << pcl_beg_time << " to " << pcl_end_time << ", " << meas.imu.size()
  //      << " imu msgs from " << imu_beg_time << " to " << imu_end_time <<
  //      endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel,
                               imu_state.pos,
                               imu_state.rot.toRotationMatrix()));
  // 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，
  // 上一帧速度，上一帧位置，上一帧旋转矩阵

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  input_ikfom in;
  // 遍历本次估计的所有IMU测量并且进行积分，离散中值法 前向传播
  fout_init << std::endl;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    if (tail->header.stamp.toSec() < last_lidar_end_time_)
      continue;

    // 中值
    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 *
                   (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // 通过重力数值对加速度进行一下微调
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

    // 如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现这种情况)
    if (head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    fout_init << "dt: " << dt << std::endl;

    // 原始测量的中值作为输入
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    // 过程噪声协方差矩阵
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    if (dt < 0.1)  // fastlio2里没有该判断
    {
      kf_state.predict(dt, Q, in);
    }

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel,
                                 imu_state.pos,
                                 imu_state.rot.toRotationMatrix()));
  }
  fout_init << std::endl;

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // 判断雷达结束时间是否晚于IMU，最后一个IMU时刻可能早于雷达末尾 也可能晚于雷达末尾
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end())
    return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    // 1：P_I=imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I->将t_n时刻雷达系下的点P_i
    // 转换到imu系，得到点P_I
    // 2：P_w=R_i *P_I+pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt->在t_n时刻下将点P_I转到世界得到点P_w
    // 3：P_ie=imu_state.rot.conjugate()*P_w - imu_state.pos
    // ->将将t_n时刻的点P_w转到最后时刻t_e下的imu坐标系得到点P_ie
    // 4：P_compensate=imu_state.offset_R_L_I.conjugate() *(P_ie -imu_state.offset_T_L_I)
    // ->在t_e时刻下，将点P_ie转的雷达坐标系得到点P_compensate，完成运动补偿
    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
       * represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));

      // 点所在时刻的位置(雷达坐标系下)
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      // 从点所在的世界位置-雷达末尾世界位置，匀加速模型，加速度影响大吗？
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
               imu_state.pos);
      ///////////////////////////////////////////////////////////////////////
      // imu_state.offset_R_L_I是从雷达到惯性的旋转矩阵 简单记为I^R_L
      // imu_state.offset_T_L_I是惯性系下雷达坐标系原点的位置简单记为I^t_L
      // 下面去畸变补偿的公式这里倒推一下
      // e代表end时刻
      // P_compensate是点在末尾时刻在雷达系的坐标 简记为L^P_e
      // 将右侧矩阵乘过来并加上右侧平移
      // 左边变为I^R_L * L^P_e + I^t_L= I^P_e 也就是end时刻点在IMU系下的坐标
      ///////////////////////////////////////////////////////////////////////
      // 右边剩下imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I *
      // P_i + imu_state.offset_T_L_I) + T_ei
      // imu_state.rot.conjugate()是结束时刻IMU到世界坐标系的旋转矩阵的转置
      // 也就是(W^R_i_e)^T
      // T_ei展开是pos_imu + vel_imu * dt + 0.5 * acc_imu * dt
      // * dt - imu_state.pos
      // 也就是点所在时刻IMU在世界坐标系下的位置 -
      // end时刻IMU在世界坐标系下的位置 W^t_I-W^t_I_e
      ///////////////////////////////////////////////////////////////////////
      // 现在等式两边变为 I^P_e =
      // (W^R_i_e)^T * (R_i * (imu_state.offset_R_L_I
      // * P_i + imu_state.offset_T_L_I) + W^t_I - W^t_I_e
      // (W^R_i_e) * I^P_e + W^t_I_e = (R_i * (imu_state.offset_R_L_I * P_i +
      // imu_state.offset_T_L_I) + W^t_I
      // 世界坐标系也无所谓时刻了 因为只有一个世界坐标系 两边变为
      // W^P = R_i * I^P+ W^t_I
      // W^P = W^P
      ///////////////////////////////////////////////////////////////////////
      V3D P_compensate =
          imu_state.offset_R_L_I.conjugate() * //.conjugate()取旋转矩阵的转置
          (imu_state.rot.conjugate() *
               (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) +
                T_ei) -
           imu_state.offset_T_L_I); // not accurate!

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

bool IMUProcessor::process_imu_only(
    const sensor_msgs::Imu::Ptr imu_data,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state)
{
  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x_imu();

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  input_ikfom in;

  angvel_avr << 0.5 * (last_imu_only_->angular_velocity.x +
                       imu_data->angular_velocity.x),
      0.5 * (last_imu_only_->angular_velocity.y + imu_data->angular_velocity.y),
      0.5 * (last_imu_only_->angular_velocity.z + imu_data->angular_velocity.z);
  acc_avr << 0.5 * (last_imu_only_->linear_acceleration.x + imu_data->linear_acceleration.x),
      0.5 * (last_imu_only_->linear_acceleration.y + imu_data->linear_acceleration.y),
      0.5 * (last_imu_only_->linear_acceleration.z + imu_data->linear_acceleration.z);

  // 通过重力数值对加速度进行一下微调
  acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

  dt = imu_data->header.stamp.toSec() - last_imu_only_->header.stamp.toSec();
  in.acc = acc_avr;
  in.gyro = angvel_avr;
  if (dt < 0.1)
  {
    kf_state.predict_imu_only(dt, in);
  }

  /* save the poses at each IMU measurements */
  imu_state = kf_state.get_x_imu();
  angvel_last_only = angvel_avr - imu_state.bg;
  acc_s_last_only = imu_state.rot * (acc_avr - imu_state.ba);
  for (int i = 0; i < 3; i++)
  {
    acc_s_last_only[i] += imu_state.grav[i];
  }

  last_imu_only_ = imu_data;

  return true;
}
#pragma clang diagnostic pop
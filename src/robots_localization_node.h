#pragma once

#include <ikd-Tree/ikd_Tree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf2_ros/transform_broadcaster.h>
#include <unistd.h>

#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <mutex>
#include <nav_msgs/msg/path.hpp>
#include <thread>

#include "imu_processor.h"
#include "lidar_processor.h"
#include "nlink_message/msg/linktrack_nodeframe2.hpp"
#include "scan_aligner.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)
#define DET_RANGE (300.0f)
#define MOV_THRESHOLD (1.5f)

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_global;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_world;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pubElevationMap;
rclcpp::TimerBase::SharedPtr elevation_map_timer;
rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pubIMUBias;
rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;
rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
rclcpp::Subscription<nlink_message::msg::LinktrackNodeframe2>::SharedPtr sub_uwb;
rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_mocap;

string root_dir = ROOT_DIR;
string ns, lid_topic, imu_topic, reloc_topic, mode_topic, pcd_path;

bool time_sync_en = false;
bool path_en = false, scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false, mapping_en = false, elevation_publish_en = false;
bool extrinsic_est_en = true, runtime_pos_log = false, pcd_save_en = false, flg_exit = false, flg_EKF_inited = false;
bool lidar_pushed, flg_first_scan = true, initialized = false, initializing_pose = false, first_pub = true;

double elevation_resolution = 0.1, elevation_offset_z = 0.75;
double time_diff_lidar_to_imu = 0.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double fov_deg = 0.0, filter_size_surf_min = 0.0, filter_size_map_min = 0.0, cube_len = 0.0;
double last_timestamp_lidar = 0.0, last_timestamp_imu = -1.0, last_timestamp_imu_back = -1.0, last_timestamp_uwb = 0.0;
double lidar_end_time = 0.0, first_lidar_time = 0.0;
double total_residual = 0.0, res_mean_last = 0.0;
float det_range = 300.0f;

int NUM_MAX_ITERATIONS = 0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0, pcd_save_interval = -1;
int feats_down_size = 0;

bool need_reloc = false, imu_only_ready = false, initT_flag = false;
int init_num = 0;
float point_num = 0.0f, point_valid_num = 0.0f, point_valid_proportion = 0.0f;
V3F reloc_initT(Zero3f);

vector<double> elevation_size(2, 0.0);
vector<float> priorT(3, 0.0);
vector<float> YAW_RANGE(3, 0.0);
vector<float> priorR(9, 0.0);
V3F prior_T(Zero3f);
M3F prior_R(Eye3f);
bool pose_inited = false;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> imu2robotT(3, 0.0);
vector<double> imu2robotR(9, 0.0);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
V3D Robot_T_wrt_IMU(Zero3d);
M3D Robot_R_wrt_IMU(Eye3d);

pcl::VoxelGrid<PointType> downSizeFilterSurf;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
state_ikfom state_point_imu;
vect3 pos_lid;

mutex mtx_buffer;
condition_variable sig_buffer;
deque<double> time_buffer;                                         // 激光雷达数据
deque<PointCloudXYZI::Ptr> lidar_buffer;                           // 雷达数据队列
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;                 // IMU数据队列
deque<std::pair<double, std::vector<UWBObservation>>> uwb_buffer;  // UWB数据队列

// PointCloudXYZI: 点云坐标 + 信号强度形式
PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  // 雷达坐标系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());        // 世界坐标系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));  // 雷达滤波
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));  // 存放法向

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
vector<PointVector> Nearest_Points;
vector<BoxPointType> cub_needrm;
float res_last[100000] = { 0.0 };
bool point_selected_surf[100000] = { 0 };

shared_ptr<LidarProcessor> p_lidar(new LidarProcessor());
shared_ptr<IMUProcessor> p_imu(new IMUProcessor());

nav_msgs::msg::Path path;
geometry_msgs::msg::PoseStamped msg_body_pose;
geometry_msgs::msg::Quaternion geoQuat;
nav_msgs::msg::Odometry odomAftMapped;
nav_msgs::msg::Odometry odomAftMappedIMU;
geometry_msgs::msg::TwistStamped odomIMUBias;  // 用来传IMU的偏置

std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

std::ofstream fout_pose;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr sub_pub_imu;

double timediff_lidar_wrt_imu = 0.0;  // lidar imu 时间差
bool timediff_set_flg = false;        // 是否已经计算了时间差

int pcd_index = 0;
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

int add_point_size = 0, kdtree_delete_counter = 0;
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

const bool time_list(PointType& x, PointType& y);

void SigHandle(int sig);

// pi:激光雷达坐标系
// 函数功能：激光雷达坐标点转到世界坐标系
// state_point.offset_R_L_I*p_body + state_point.offset_T_L_I:转到IMU坐标系
// state_point.rot: IMU坐标系到世界坐标系的旋转
void pointBodyToWorld(PointType const* const pi, PointType* const po);

// 激光雷达坐标系到IMU坐标系
void pointBodyLidarToIMU(PointType const* const pi, PointType* const po);

void pointBodyLidarToRobot(PointType const* const pi, PointType* const po);

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1>& pi, Matrix<T, 3, 1>& po);

void RGBpointBodyToWorld(PointType const* const pi, PointType* const po);

void RGBpointBodyLidarToIMU(PointType const* const pi, PointType* const po);

template <typename T>
void set_posestamp(T& out);

// 通过pubOdomAftMapped发布位姿odomAftMapped，同时计算协方差存在kf中，同tf计算位姿
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& pubOdomAftMapped);

void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& pubPath);

void publish_frame_global(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_global);

// 发布feats_undistort转到机器人下的laserCloudIMUBody
void publish_frame_body(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_body);

void publish_frame_world_local(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_world);

void publish_elevation_map(const rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr& pubElevationMap);

// 观测模型
void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
    if (opt_with_uwb) {
        std::vector<UWBObservation> cur_uwb_meas = Measures.uwb.front().second;
        std::vector<UWBObservation> inited_anchor_meas = get_inited_anchor_meas(cur_uwb_meas);
        int update_anchor_num = inited_anchor_meas.size();
        ekfom_data.z = MatrixXd::Zero(update_anchor_num, 1);
        ekfom_data.h_x = MatrixXd::Zero(update_anchor_num, 47);
        ekfom_data.h.resize(update_anchor_num);
        ekfom_data.R = MatrixXd::Identity(update_anchor_num, update_anchor_num);
        MatrixXd additional_td_R = MatrixXd::Zero(update_anchor_num, update_anchor_num);
        ekfom_data.h_v = MatrixXd::Identity(update_anchor_num, update_anchor_num);
        for (int i = 0; i < update_anchor_num; ++i) {
            double dist_meas = inited_anchor_meas[i].distance;
            vect5 anchor_state;
            int cur_anchor_id = inited_anchor_meas[i].anchor_id;
            switch (cur_anchor_id) {
                case 1:
                    anchor_state = s.anchor1;
                    break;
                case 2:
                    anchor_state = s.anchor2;
                    break;
                case 3:
                    anchor_state = s.anchor3;
                    break;
                case 4:
                    anchor_state = s.anchor4;
                default:
                    break;
            }
            // s * ||w^p_i + w^R_i * i^p_t - w^p_a|| + b
            V3D anchor_position(anchor_state[0], anchor_state[1], anchor_state[2]);
            // 考虑了uwb与imu之间的时延，td时间内tag多平移了v*td
            // 如果不估计td，默认td一直为0.0，不会影响原本残差和雅可比的计算
            double dist_pred =
                anchor_state[3] *
                    (s.pos + s.rot * s.offset_T_I_U + s.vel * s.td[0] - anchor_position).norm() +
                anchor_state[4];
            double res = dist_meas - dist_pred;

            // residual
            ekfom_data.h(i) = res;

            // Jacobian，注：FAST_LIO中的雅克比不是观测对状态的偏导，而是残差对状态的偏导
            V3D scaled_direction_vec =
                anchor_state[3] * (s.pos + s.rot * s.offset_T_I_U + s.vel * s.td[0] - anchor_position) /
                (s.pos + s.rot * s.offset_T_I_U + s.vel * s.td[0] - anchor_position).norm();
            int start_index = anchor_id_state_index[cur_anchor_id];
            // 1.对位置 (dh_dpos)
            ekfom_data.h_x.block<1, 3>(i, 0) = -scaled_direction_vec.transpose();
            std::cout << "ekfom_data.h_x.block<1, 3>(" << i << ", 0): " << scaled_direction_vec.transpose()
                      << std::endl;
            // 2.对旋转 (dh_drot)
            M3D crossmat;
            crossmat << SKEW_SYM_MATRX(s.offset_T_I_U);
            ekfom_data.h_x.block<1, 3>(i, 3) =
                scaled_direction_vec.transpose() * s.rot.toRotationMatrix() * crossmat;
            if (esti_uwb_offset) {
                // 3.对外参 (dh_doffset_T_I_U)
                ekfom_data.h_x.block<1, 3>(i, 23) =
                    -scaled_direction_vec.transpose() * s.rot.toRotationMatrix();
            }
            if (esti_uwb_anchor) {
                // 4.对基站坐标 (dh_danchorposition)
                ekfom_data.h_x.block<1, 3>(i, start_index) = scaled_direction_vec.transpose();
            }
            if (esti_uwb_scale) {
                // 5.对测距尺度 (dh_danchorscale)
                ekfom_data.h_x(i, start_index + 3) =
                    -(s.pos + s.rot * s.offset_T_I_U + s.vel * s.td[0] - anchor_position).norm();
            }
            if (esti_uwb_bias) {
                // 6.对测距偏置 (dh_danchorbias)
                ekfom_data.h_x(i, start_index + 4) = -1.0;
            }
            if (estimate_td) {
                // 估计了uwb与imu之间的时延，观测矩阵应该包含观测对td的雅可比项
                // std::cout << s.vel.transpose() << " " << scaled_direction_vec.transpose() << " "
                //           << s.vel.transpose() * scaled_direction_vec << std::endl;
                // 7.对时延 (dh_dtd)
                ekfom_data.h_x(i, 46) = -static_cast<double>(s.vel.transpose() * scaled_direction_vec);
                std::cout << "ekfom_data.h_x(" << i << ", 46): " << ekfom_data.h_x(i, 46) << std::endl;
                additional_td_R(i, i) = td_std * td_std * ekfom_data.h_x(i, 46) * ekfom_data.h_x(i, 46);
            }

            // chi-squared test
            // S = H_x * P * H_x^T + uwb_range_std * uwb_range_std, chi2 = res * S^{-1} * res
            MatrixXd cur_hx = ekfom_data.h_x.block<1, 47>(i, 0);
            auto P = kf.get_P();
            auto dist_cov = cur_hx * P * cur_hx.transpose();
            double S = dist_cov(0, 0) + uwb_range_std * uwb_range_std;
            double chi2 = res * res / S;
            if (chi2 > uwb_chi2_threshold) {  // 将测量矩阵置0，不利用该uwb进行状态更新
                ekfom_data.h_x.block<1, 47>(i, 0) = MatrixXd::Zero(1, 47);
                std::cout << "big error: " << chi2 << ", avoid update!!! dist_meas: " << dist_meas
                          << ", dist_pred: " << dist_pred << ", res: " << res << ", S: " << S << std::endl;
            }
        }
        // covariance, 残差协方差S = H * P * H^T + sigma_r^2 * I + sigma_td^2 * H_td * H_td^T,
        // 即td被建模为符合高斯分布，标准差为td_std
        ekfom_data.R = uwb_range_std * uwb_range_std * ekfom_data.R + additional_td_R;
        return;
    }

    if (opt_with_zupt) {
        ekfom_data.z = MatrixXd::Zero(6, 1);
        ekfom_data.h_x = MatrixXd::Zero(6, 47);
        ekfom_data.h.resize(6);
        ekfom_data.R = MatrixXd::Identity(6, 6);
        ekfom_data.h_v = MatrixXd::Identity(6, 6);
        // residual
        V3D gyr_res = recent_avg_gyr - s.bg;
        V3D vel_res = -s.vel;
        ekfom_data.h[0] = gyr_res[0];
        ekfom_data.h[1] = gyr_res[1];
        ekfom_data.h[2] = gyr_res[2];
        ekfom_data.h[3] = vel_res[0];
        ekfom_data.h[4] = vel_res[1];
        ekfom_data.h[5] = vel_res[2];
        // jacobian
        ekfom_data.h_x.block<3, 3>(0, 15) = -Matrix3d::Identity();
        ekfom_data.h_x.block<3, 3>(3, 12) = -Matrix3d::Identity();
        // covariance
        ekfom_data.R.block<3, 3>(0, 0) = zupt_gyr_std * zupt_gyr_std * Matrix3d::Identity();
        ekfom_data.R.block<3, 3>(3, 3) = zupt_vel_std * zupt_vel_std * Matrix3d::Identity();

        // chi-square test
        MatrixXd cur_hx = ekfom_data.h_x;
        VectorXd res = ekfom_data.h;
        MatrixXd P = kf.get_P();
        MatrixXd S = cur_hx * P * cur_hx.transpose();
        S.block<3, 3>(0, 0).diagonal() += zupt_gyr_std * zupt_gyr_std * Eigen::VectorXd::Ones(3);
        S.block<3, 3>(3, 3).diagonal() += zupt_vel_std * zupt_vel_std * Eigen::VectorXd::Ones(3);
        double chi2 = res.dot(S.llt().solve(res));
        if (chi2 > zupt_chi2_threshold) {
            ekfom_data.h_x = MatrixXd::Zero(6, 47);
        }

        return;
    }

    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;  // 残差和

    point_valid_num = 0.0;
    point_num = 0.0;
    // 最近邻面搜索，以及残差计算
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    /** closest surface search and residual computation **/
    // 遍历所有特征点，判断每个点的对应邻域是否符合平面点的假设
    for (int i = 0; i < feats_down_size; i++) {
        point_num += 1.0;
        // feats_down_body: 网格滤波器之后的激光点
        PointType& point_body = feats_down_body->points[i];
        // feats_down_world: 世界坐标系下的激光点
        PointType& point_world = feats_down_world->points[i];

        V3D p_body(point_body.x, point_body.y, point_body.z);
        /* transform to world frame */
        // 激光雷达坐标系->IMU坐标系->世界坐标系
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;  // 信号强度

        // NUM_MATCH_POINTS: 5
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto& points_near = Nearest_Points[i];

        if (ekfom_data.converge) {
            /** Find the closest surfaces in the map **/
            // 在地图中找到与之最邻近的平面，world系下从ikdtree找NUM_MATCH_POINTS个最近点用于平面拟合
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，
            // 则认为该点不是有效点
            // 判断是否是有效匹配点，与LOAM系列类似，要求特征点最近邻的地图点数量大于阈值A，距离小于阈值B
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS        ? false
                                     : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                  : true;
        }
        if (!point_selected_surf[i]) continue;  // 如果该点不是有效点

        VF(4) pabcd;                     // 法向量
        point_selected_surf[i] = false;  // 二次筛选平面点
        // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f)) {  // 计算平面法向量
            // 根据它计算过程推测points_near的原点应该是这几个点中的一个，拟合了平面之后原点也就近似在平面
            // 上了，这样下面算出来的投影就是点到平面的距离。
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                        pabcd(3);  // 计算点到平面的距离
            // 发射距离越长，测量误差越大，归一化，消除雷达点发射距离的影响
            // p_body是激光雷达坐标系下的点
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());  // 判断残差阈值

            if (s > 0.9) {  // 如果残差大于阈值，则认为该点是有效点
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

    // 根据point_selected_surf状态判断哪些点是可用的
    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++) {
        if (point_selected_surf[i]) {  // 只保留有效的特征点
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];  // 计算总残差
            effct_feat_num++;
        }
    }
    if (effct_feat_num < 1) {
        ekfom_data.valid = false;
        std::cout << "No Effective Points! \n" << std::endl;
        return;
    }
    res_mean_last = total_residual / effct_feat_num;  // 残差均值 （距离）

    /* Computation of Measuremnt Jacobian matrix H and measurents vector */
    // 测量雅可比矩阵H和测量向量的计算 H=J*P*J'
    // h_x是观测h相对于状态x的jacobian，尺寸为特征点数x12
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // (23)
    ekfom_data.h.resize(effct_feat_num);                  // 有效方程个数

    // 求观测值与误差的雅克比矩阵，如论文式14以及式12、13
    for (int i = 0; i < effct_feat_num; i++) {
        // 拿到有效点的坐标
        const PointType& laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        // 转换到IMU坐标系下
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType& norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);  // 对应局部法相量, world系下

        /*** calculate the Measuremnt Jacobian matrix H ***/
        // conjugate()用于计算四元数的共轭，表示旋转的逆
        V3D C(s.rot.conjugate() * norm_vec);  // 世界坐标系的法向量旋转到IMU坐标系
        V3D A(point_crossmat * C);            // IMU坐标系下原点到点云点距离在法向上的投影

        if (extrinsic_est_en) {  // extrinsic_est_en: IMU,lidar外参在线更新
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // Lidar坐标系下点向量在法向上的投影
            // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A),
                VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        } else {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
}

class RobotsLocalizationNode : public rclcpp::Node {
   public:
    RobotsLocalizationNode() : Node("robots_localization_node") {
        loadConfig();
        if (runtime_pos_log) {
            fout_pose.open(root_dir + "/log/" + p_imu->timeStr + "_Pose.txt", std::ios::out | std::ios::app);
        }

        if (log_mocap_traj) {
            fout_mocap.open(root_dir + "/log/uwb_fusion/" + p_imu->timeStr + "_Mocap.txt",
                            std::ios::out | std::ios::app);
            fout_mocap << "# timestamp(s) tx ty tz qx qy qz qw" << std::endl;
        }

        if (log_fusion_traj) {
            if (USE_UWB)
                fout_fusion.open(root_dir + "/log/uwb_fusion/" + p_imu->timeStr + "_with_uwb.txt",
                                 std::ios::out | std::ios::app);
            else
                fout_fusion.open(root_dir + "/log/uwb_fusion/" + p_imu->timeStr + "_without_uwb.txt",
                                 std::ios::out | std::ios::app);
            fout_fusion << "# timestamp(s) tx ty tz qx qy qz qw" << std::endl;
        }

        if (USE_UWB && !use_calibrated_anchor) {
            fout_uwb_calib.open(root_dir + "/log/uwb_calib/" + p_imu->timeStr + "_Anchor_Calib.txt",
                                std::ios::out | std::ios::app);
        }

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        /*** Map initialization ***/
        if (!mapping_en) {
            // string map_pcd = root_dir + "map/map.pcd";
            std::string map_pcd;
            map_pcd = pcd_path;
            std::string infoMsg = "[Robots Localization] Load Map:" + map_pcd;
            std::cout << infoMsg << std::endl;
            if (pcl::io::loadPCDFile<PointType>(map_pcd, *global_map) == -1) {
                PCL_ERROR("Couldn't read file map.pcd\n");
                rclcpp::shutdown();
            }

            // 去除NaN
            std::vector<int> indices;
            global_map->is_dense = false;
            pcl::removeNaNFromPointCloud(*global_map, *global_map, indices);

            std::cout << "map cloud width: " << global_map->width << std::endl;
            std::cout << "map cloud height: " << global_map->height << std::endl;
            if (ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(filter_size_map_min);
                ikdtree.Build(global_map->points);
            }
            std::cout << "KDtree built! " << std::endl;
        }

        prior_T << VEC_FROM_ARRAY(priorT);
        prior_R << MAT_FROM_ARRAY(priorR);
        std::cout << "init T: " << prior_T << std::endl;
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        Robot_T_wrt_IMU << VEC_FROM_ARRAY(imu2robotT);
        Robot_R_wrt_IMU << MAT_FROM_ARRAY(imu2robotR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        YAW_RANGE[1] = 0.35;
        YAW_RANGE[2] = 6.3;
        double epsi[47] = {0.001};
        fill(epsi, epsi + 47, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        // uwb相关参数初始化
        if (USE_UWB) initializeUWB();

        // 将函数地址传入kf对象中，用于接收特定于系统的模型
        // 及其差异作为一个维数变化的特征矩阵进行测量。
        // 通过一个函数（h_dyn_share_in）同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）

        /*** ROS initialization ***/

        pubLaserCloudFull_global =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 20);
        pubLaserCloudFull_body =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", 20);
        pubLaserCloudFull_world =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_world", 20);
        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("laser_map", 20);
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("odometry", 20);
        sub_pub_imu = this->create_publisher<nav_msgs::msg::Odometry>("odometry_imu", 20);
        pubIMUBias = this->create_publisher<geometry_msgs::msg::TwistStamped>("IMU_bias", 20);
        pubPath = this->create_publisher<nav_msgs::msg::Path>("path", 20);
        pubElevationMap = this->create_publisher<std_msgs::msg::Float32MultiArray>("elevation_map", 10);
        if (elevation_publish_en) {
            elevation_map_timer = this->create_wall_timer(
                std::chrono::milliseconds(20), 
            [this]() { 
            if (initialized && imu_only_ready && !need_reloc) {
                publish_elevation_map(pubElevationMap); 
            }
        });
        }
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        signal(SIGINT, SigHandle);
        if (p_lidar->lidar_type == AVIA) {
            sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, 1, std::bind(&RobotsLocalizationNode::livox_pcl_cbk, this, std::placeholders::_1));
        } else {
            sub_pcl =
                this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, 1, std::bind(&RobotsLocalizationNode::standard_pcl_cbk, this, std::placeholders::_1));
        }
        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200000, std::bind(&RobotsLocalizationNode::imu_cbk, this, std::placeholders::_1));
        if (USE_UWB)
            sub_uwb = this->create_subscription<nlink_message::msg::LinktrackNodeframe2>(uwb_topic, 200000,
                                                                                         std::bind(&RobotsLocalizationNode::uwb_cbk, this, std::placeholders::_1));
        if (log_mocap_traj)
            sub_mocap =
                this->create_subscription<geometry_msgs::msg::PoseStamped>(mocap_topic, 20000, std::bind(&RobotsLocalizationNode::mocap_cbk, this, std::placeholders::_1));

        std::thread mainThread(&RobotsLocalizationNode::mainProcessThread, this);
        mainThread.detach();
    }
    void points_cache_collect();
    void lasermap_fov_segment();
    void map_incremental();
    void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr& msg);
    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);
    void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr& msg_in);
    void uwb_cbk(const nlink_message::msg::LinktrackNodeframe2::ConstSharedPtr& msg);
    void mocap_cbk(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg);
    bool sync_packages(MeasureGroup& meas);
    void loadConfig();
    void mainProcess();
    void mainProcessThread();
    void initializeUWB();
};
#pragma once

#include <ikd-Tree/ikd_Tree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <tf2_ros/transform_broadcaster.h>
#include <unistd.h>

#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <mutex>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/grid_cells.hpp>
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
rclcpp::Publisher<nav_msgs::msg::GridCells>::SharedPtr pubGlobalElevationMap;
rclcpp::Publisher<nav_msgs::msg::GridCells>::SharedPtr pubElevationMap;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pubElevation;
rclcpp::TimerBase::SharedPtr elevation_map_timer;
rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pubIMUBias;
rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;
rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
rclcpp::Subscription<nlink_message::msg::LinktrackNodeframe2>::SharedPtr sub_uwb;
rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_mocap;
rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initial_pose;

string root_dir = ROOT_DIR;
string ns, lid_topic, imu_topic, reloc_topic, mode_topic, pcd_path;

bool time_sync_en = false;
bool path_en = false, scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false, mapping_en = false, elevation_publish_en = false, elevation_viz_pub_en = false;
bool extrinsic_est_en = true, runtime_pos_log = false, pcd_save_en = false, flg_exit = false, flg_EKF_inited = false;
bool lidar_pushed, flg_first_scan = true, initialized = false, initializing_pose = false, first_pub = true;

double elevation_resolution = 0.1, elevation_offset_z = 0.75, global_elevation_resolution = 0.01, global_elevation_max_x = 8.0, global_elevation_max_y = 3.0, global_elevation_max_z = 2.0;
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
M3F reloc_initR = Eye3f;

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
pcl::PassThrough<PointType> passFilterX;
pcl::PassThrough<PointType> passFilterY;
pcl::PassThrough<PointType> passFilterZ;
pcl::StatisticalOutlierRemoval<PointType> sorFilter;

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
PointCloudXYZI::Ptr filtered_map(new PointCloudXYZI());
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

struct GlobalElevationMap {
    std::vector<float> data;
    std::vector<bool> valid;
    float origin_x, origin_y;
    float resolution;
    int size_x, size_y;
    
    bool getElevation(float x, float y, float& elevation) const {
        int ix = static_cast<int>((x - origin_x) / resolution);
        int iy = static_cast<int>((y - origin_y) / resolution);
        if (ix < 0 || ix >= size_x || iy < 0 || iy >= size_y) return false;
        int idx = iy * size_x + ix;
        if (!valid[idx]) return false;
        elevation = data[idx];
        return true;
    }
};

GlobalElevationMap global_elevation_map;

void buildGlobalElevationMap(const PointCloudXYZI::Ptr& cloud, 
                              GlobalElevationMap& elev_map,
                              float resolution = 0.1f);

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

void publish_elevation(const rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr& pubElevation,
                       const rclcpp::Publisher<nav_msgs::msg::GridCells>::SharedPtr& pubElevationMap);

// 观测模型
void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

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
            if (elevation_publish_en) {
                PointCloudXYZI::Ptr tmp_map_x(new PointCloudXYZI());
                PointCloudXYZI::Ptr tmp_map_xy(new PointCloudXYZI());
                PointCloudXYZI::Ptr tmp_map_xyz(new PointCloudXYZI());
                passFilterX.setInputCloud(global_map);
                passFilterX.setFilterFieldName("x");
                passFilterX.setFilterLimits(-global_elevation_max_x, global_elevation_max_x);
                passFilterX.filter(*tmp_map_x);
                passFilterY.setInputCloud(tmp_map_x);
                passFilterY.setFilterFieldName("y");
                passFilterY.setFilterLimits(-global_elevation_max_y, global_elevation_max_y);
                passFilterY.filter(*tmp_map_xy);
                passFilterZ.setInputCloud(tmp_map_xy);
                passFilterZ.setFilterFieldName("z");
                passFilterZ.setFilterLimits(-global_elevation_max_z, global_elevation_max_z);
                passFilterZ.filter(*tmp_map_xyz);
                sorFilter.setInputCloud(tmp_map_xyz);
                sorFilter.setMeanK(10);
                sorFilter.setStddevMulThresh(2.0);
                sorFilter.filter(*filtered_map);
                buildGlobalElevationMap(filtered_map, global_elevation_map, global_elevation_resolution);
            }
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
        rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
        qos_profile.transient_local();
        pubLaserCloudFull_global =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", qos_profile);
        pubLaserCloudFull_body =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", 20);
        pubLaserCloudFull_world =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_world", 20);
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("odometry", 20);
        sub_pub_imu = this->create_publisher<nav_msgs::msg::Odometry>("odometry_imu", 20);
        pubIMUBias = this->create_publisher<geometry_msgs::msg::TwistStamped>("IMU_bias", 20);
        pubPath = this->create_publisher<nav_msgs::msg::Path>("path", 20);
        pubGlobalElevationMap = this->create_publisher<nav_msgs::msg::GridCells>("elevation", qos_profile);
        pubElevationMap = this->create_publisher<nav_msgs::msg::GridCells>("elevation_body", 20);
        pubElevation = this->create_publisher<std_msgs::msg::Float32MultiArray>("elevation_data", 20);
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        if (!mapping_en && elevation_publish_en) {
            if (global_elevation_map.data.empty()) {
                std::cout << "Global elevation map is empty!" << std::endl;
                rclcpp::shutdown();
            } 
            nav_msgs::msg::GridCells msg;
            msg.header.frame_id = "world";
            msg.header.stamp = this->now();
            msg.cell_width = global_elevation_map.resolution;
            msg.cell_height = global_elevation_map.resolution;
            for (int iy = 0; iy < global_elevation_map.size_y; ++iy) {
                for (int ix = 0; ix < global_elevation_map.size_x; ++ix) {
                    int idx = iy * global_elevation_map.size_x + ix;
                    if (!global_elevation_map.valid[idx]) continue;

                    geometry_msgs::msg::Point pt;
                    pt.x = global_elevation_map.origin_x + (ix + 0.5f) * global_elevation_map.resolution;
                    pt.y = global_elevation_map.origin_y + (iy + 0.5f) * global_elevation_map.resolution;
                    pt.z = global_elevation_map.data[idx];
                    msg.cells.push_back(pt);
                }
            }
            pubGlobalElevationMap->publish(msg);

            elevation_map_timer = this->create_wall_timer(
                std::chrono::milliseconds(20), 
            [this]() { 
            if (initialized && imu_only_ready && !need_reloc) {
                    publish_elevation(pubElevation, pubElevationMap);
                }
            });
        }

        signal(SIGINT, SigHandle);
        if (p_lidar->lidar_type == AVIA) {
            sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, 1, std::bind(&RobotsLocalizationNode::livox_pcl_cbk, this, std::placeholders::_1));
        } else {
            sub_pcl =
                this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, 1, std::bind(&RobotsLocalizationNode::standard_pcl_cbk, this, std::placeholders::_1));
        }
        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200, std::bind(&RobotsLocalizationNode::imu_cbk, this, std::placeholders::_1));
        if (USE_UWB)
            sub_uwb = this->create_subscription<nlink_message::msg::LinktrackNodeframe2>(uwb_topic, 200,
                                                                                         std::bind(&RobotsLocalizationNode::uwb_cbk, this, std::placeholders::_1));
        if (log_mocap_traj)
            sub_mocap =
                this->create_subscription<geometry_msgs::msg::PoseStamped>(mocap_topic, 200, std::bind(&RobotsLocalizationNode::mocap_cbk, this, std::placeholders::_1));

        sub_initial_pose = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10, std::bind(&RobotsLocalizationNode::initial_pose_cbk, this, std::placeholders::_1));

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
    void initial_pose_cbk(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& msg);
    bool sync_packages(MeasureGroup& meas);
    void loadConfig();
    void mainProcess();
    void mainProcessThread();
    void initializeUWB();
};
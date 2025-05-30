#include "robots_localization_node.h"

const bool time_list(PointType& x, PointType& y) { return (x.curvature < y.curvature); }

void SigHandle(int sig)
{
  flg_exit = true;
  std::cout << "catch sig" << sig << std::endl;
  sig_buffer.notify_all();
}

// pi:激光雷达坐标系
// 函数功能：激光雷达坐标点转到世界坐标系
// state_point.offset_R_L_I*p_body + state_point.offset_T_L_I:转到IMU坐标系
// state_point.rot: IMU坐标系到世界坐标系的旋转
void pointBodyToWorld(PointType const* const pi, PointType* const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

// 激光雷达坐标系到IMU坐标系
void pointBodyLidarToIMU(PointType const* const pi, PointType* const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void pointBodyLidarToRobot(PointType const* const pi, PointType* const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_robot(Robot_R_wrt_IMU.inverse() * (state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I) -
                   Robot_T_wrt_IMU);

  po->x = p_body_robot(0);
  po->y = p_body_robot(1);
  po->z = p_body_robot(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1>& pi, Matrix<T, 3, 1>& po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const* const pi, PointType* const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const* const pi, PointType* const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

template <typename T>
void set_posestamp(T& out) {
    V3D pos_robot = state_point.pos + state_point.rot * Robot_T_wrt_IMU;
    out.pose.position.x = pos_robot(0);
    out.pose.position.y = pos_robot(1);
    out.pose.position.z = pos_robot(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

// 通过pubOdomAftMapped发布位姿odomAftMapped，同时计算协方差存在kf中，同tf计算位姿
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& pubOdomAftMapped) {
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = ns;
    odomAftMapped.header.stamp.sec = static_cast<int32_t>(lidar_end_time);
    odomAftMapped.header.stamp.nanosec =
        static_cast<uint32_t>((lidar_end_time - odomAftMapped.header.stamp.sec) * 1e9);
    set_posestamp(odomAftMapped.pose);  // 设置位置，欧拉角

    odomAftMapped.twist.twist.linear.x = state_point.vel(0);
    odomAftMapped.twist.twist.linear.y = state_point.vel(1);
    odomAftMapped.twist.twist.linear.z = state_point.vel(2);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "world";
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = ns;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_broadcaster->sendTransform(trans);

    odomIMUBias.twist.linear.x = state_point.ba(0);
    odomIMUBias.twist.linear.y = state_point.ba(1);
    odomIMUBias.twist.linear.z = state_point.ba(2);
    // 利用twist的空位发布bg
    odomIMUBias.twist.angular.x = state_point.bg(0);
    odomIMUBias.twist.angular.y = state_point.bg(1);
    odomIMUBias.twist.angular.z = state_point.bg(2);
    pubIMUBias->publish(odomIMUBias);

    if (runtime_pos_log) {
        fout_pose << lidar_end_time << ", " << odomAftMapped.pose.pose.position.x << ", "
                  << odomAftMapped.pose.pose.position.y << ", " << odomAftMapped.pose.pose.position.z << ", "
                  << odomAftMapped.pose.pose.orientation.x << ", " << odomAftMapped.pose.pose.orientation.y
                  << ", " << odomAftMapped.pose.pose.orientation.z << ", "
                  << odomAftMapped.pose.pose.orientation.w << std::endl;
    }
}

void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& pubPath) {
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp.sec = static_cast<int32_t>(lidar_end_time);
    msg_body_pose.header.stamp.nanosec =
        static_cast<uint32_t>((lidar_end_time - msg_body_pose.header.stamp.sec) * 1e9);
    msg_body_pose.header.frame_id = "world";
    path.header.frame_id = "world";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void publish_frame_global(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_global) {
    PointCloudXYZI::Ptr laserCloudGobal(mapping_en ? feats_undistort : global_map);
    int size = laserCloudGobal->points.size();
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    if (mapping_en) {
        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&feats_undistort->points[i], &laserCloudGobal->points[i]);
        }
    }
    pcl::toROSMsg(*laserCloudGobal, laserCloudmsg);
    laserCloudmsg.header.stamp.sec = static_cast<int32_t>(lidar_end_time);
    laserCloudmsg.header.stamp.nanosec =
        static_cast<uint32_t>((lidar_end_time - laserCloudmsg.header.stamp.sec) * 1e9);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudFull_global->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 发布feats_undistort转到机器人下的laserCloudIMUBody
void publish_frame_body(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_body) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) {
        pointBodyLidarToRobot(&laserCloudFullRes->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp.sec = static_cast<int32_t>(lidar_end_time);
    laserCloudmsg.header.stamp.nanosec =
        static_cast<uint32_t>((lidar_end_time - laserCloudmsg.header.stamp.sec) * 1e9);
    laserCloudmsg.header.frame_id = ns;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_frame_world_local(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pubLaserCloudFull_world) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) {
        pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudIMUWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUWorld, laserCloudmsg);
    laserCloudmsg.header.stamp.sec = static_cast<int32_t>(lidar_end_time);
    laserCloudmsg.header.stamp.nanosec =
        static_cast<uint32_t>((lidar_end_time - laserCloudmsg.header.stamp.sec) * 1e9);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudFull_world->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en && mapping_en) {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) +
                                  string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void RobotsLocalizationNode::points_cache_collect() {
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

void RobotsLocalizationNode::lasermap_fov_segment() {
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}

void RobotsLocalizationNode::map_incremental() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector& points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        } else {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

// livox激光雷达回调函数
void RobotsLocalizationNode::livox_pcl_cbk(const robots_localization::msg::CustomMsg::ConstPtr& msg) {
    if (initializing_pose) {
        return;
    }
    mtx_buffer.lock();
    scan_count++;
    double time_stamp =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    if (time_stamp < last_timestamp_lidar) {
        std::cout << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    last_timestamp_lidar = time_stamp;

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
void RobotsLocalizationNode::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstPtr& msg) {
    mtx_buffer.lock();
    scan_count++;
    double time_stamp =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    if (time_stamp < last_timestamp_lidar) {
        std::cout << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_lidar->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(time_stamp);
    last_timestamp_lidar = time_stamp;

    mtx_buffer.unlock();
    sig_buffer.notify_all();  // 唤醒所有线程
}

// 接收IMU数据回调函数
// ConstPtr: 智能指针
void RobotsLocalizationNode::imu_cbk(const sensor_msgs::msg::Imu::ConstPtr& msg_in) {
    if (initializing_pose) {
        return;
    }

    publish_count++;
    sensor_msgs::msg::Imu::Ptr msg(new sensor_msgs::msg::Imu(*msg_in));
    double time_stamp = static_cast<double>(msg_in->header.stamp.sec) +
                        static_cast<double>(msg_in->header.stamp.nanosec) * 1e-9;
    double final_time_sec = time_stamp - time_diff_lidar_to_imu;
    msg->header.stamp.sec = static_cast<int32_t>(final_time_sec);
    msg->header.stamp.nanosec = static_cast<uint32_t>((final_time_sec - msg->header.stamp.sec) * 1e9);
    // 将IMU和激光雷达点云的时间戳对齐（livox）
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
        double sync_time_sec = time_stamp + timediff_lidar_wrt_imu;
        msg->header.stamp.sec = static_cast<int32_t>(sync_time_sec);
        msg->header.stamp.nanosec = static_cast<uint32_t>((sync_time_sec - msg->header.stamp.sec) * 1e9);
    }
    // 将IMU和激光雷达点云的时间戳对齐（else）

    double timestamp =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    if (USE_ZUPT) {
        // clear old zupt imu data
        if (!zupt_imu_buffer.empty()) {
            if ((timestamp - zupt_imu_buffer.front().timestamp) > zupt_duration) zupt_imu_buffer.pop_front();
        }
        // add new zupt imu data
        IMU zupt_imu;
        zupt_imu.timestamp = timestamp;
        zupt_imu.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
        zupt_imu.gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
        zupt_imu_buffer.push_back(zupt_imu);
    }

    // 上锁
    mtx_buffer.lock();
    if (timestamp < last_timestamp_imu) {
        std::cout << "imu loop back, clear buffer" << std::endl;
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
            odomAftMappedIMU.header.frame_id = "world";
            odomAftMappedIMU.child_frame_id = ns;
            odomAftMappedIMU.header.stamp.sec = static_cast<int32_t>(last_timestamp_imu);
            double nanosec = last_timestamp_imu - odomAftMappedIMU.header.stamp.sec;
            odomAftMappedIMU.header.stamp.nanosec = static_cast<uint32_t>(nanosec * 1e9);
            V3D pos_robot = state_point_imu.pos + state_point_imu.rot * Robot_T_wrt_IMU;
            odomAftMappedIMU.pose.pose.position.x = pos_robot(0);
            odomAftMappedIMU.pose.pose.position.y = pos_robot(1);
            odomAftMappedIMU.pose.pose.position.z = pos_robot(2);
            odomAftMappedIMU.twist.twist.linear.x = state_point_imu.vel(0);
            odomAftMappedIMU.twist.twist.linear.y = state_point_imu.vel(1);
            odomAftMappedIMU.twist.twist.linear.z = state_point_imu.vel(2);
            SO3 rot_robot = state_point_imu.rot * Robot_R_wrt_IMU;
            odomAftMappedIMU.pose.pose.orientation.x = rot_robot.coeffs()[0];
            odomAftMappedIMU.pose.pose.orientation.y = rot_robot.coeffs()[1];
            odomAftMappedIMU.pose.pose.orientation.z = rot_robot.coeffs()[2];
            odomAftMappedIMU.pose.pose.orientation.w = rot_robot.coeffs()[3];

            sub_pub_imu->publish(odomAftMappedIMU);

            if (log_fusion_traj) {
                // timestamp
                fout_fusion.precision(5);
                fout_fusion.setf(std::ios::fixed, std::ios::floatfield);
                fout_fusion << last_timestamp_imu << " ";

                // pose
                fout_fusion.precision(6);
                fout_fusion << pos_robot(0) << " " << pos_robot(1) << " " << pos_robot(2) << " "
                            << rot_robot.coeffs()[0] << " " << rot_robot.coeffs()[1] << " "
                            << rot_robot.coeffs()[2] << " " << rot_robot.coeffs()[3] << std::endl;
            }

            last_timestamp_imu_back = last_timestamp_imu;
        }
    }
}

void RobotsLocalizationNode::uwb_cbk(const nlink_message::msg::LinktrackNodeframe2::ConstPtr& msg) {
    // 初始化之后才能获得每个UWB时刻的IMU位姿
    if (!initialized) return;

    double timestamp =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    mtx_buffer.lock();

    if (timestamp < last_timestamp_uwb) {
        std::cout << "uwb loop back, clear buffer" << std::endl;
        uwb_buffer.clear();
    }

    last_timestamp_uwb = timestamp;

    std::pair<double, std::vector<UWBObservation>> cur_uwb_meas;
    cur_uwb_meas.first = timestamp;
    std::vector<UWBObservation> uwb_meas;
    for (int i = 0; i < msg->nodes.size(); ++i) {
        UWBObservation meas;
        meas.anchor_id = msg->nodes[i].id;
        meas.distance = msg->nodes[i].dis;
        uwb_meas.emplace_back(meas);
    }
    cur_uwb_meas.second = uwb_meas;

    uwb_buffer.push_back(cur_uwb_meas);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void RobotsLocalizationNode::mocap_cbk(const geometry_msgs::msg::PoseStamped::ConstPtr& msg) {
    // timestamp
    fout_mocap.precision(5);
    fout_mocap.setf(std::ios::fixed, std::ios::floatfield);
    double time_stamp =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    fout_mocap << time_stamp << " ";

    // pose
    fout_mocap.precision(6);
    fout_mocap << msg->pose.position.x << " " << msg->pose.position.y << " " << msg->pose.position.z << " "
               << msg->pose.orientation.x << " " << msg->pose.orientation.y << " " << msg->pose.orientation.z
               << " " << msg->pose.orientation.w << std::endl;
}

// 将两帧激光雷达点云数据时间内的IMU数据和UWB数据从缓存队列中取出，进行时间对齐，并保存到meas中
// 输入数据：lidar_buffer, imu_buffer, uwb_buffer
// 输出数据：MeasureGroup
// 备注：必须同时有IMU数据和lidar数据
bool RobotsLocalizationNode::sync_packages(MeasureGroup& meas) {
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    if (USE_UWB && uwb_buffer.empty()) {
        opt_with_uwb = false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();  // lidar指针指向最旧的lidar数据

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
    if (last_timestamp_imu < lidar_end_time) {  // imu落后lidar
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = static_cast<double>(imu_buffer.front()->header.stamp.sec) +
                      static_cast<double>(imu_buffer.front()->header.stamp.nanosec) * 1e-9;  // 最旧IMU时间
    meas.imu.clear();

    while ((!imu_buffer.empty()) &&
           (imu_time < lidar_end_time))  // 记录imu数据，imu时间小于当前帧lidar结束时间
    {
        imu_time = static_cast<double>(imu_buffer.front()->header.stamp.sec) +
                   static_cast<double>(imu_buffer.front()->header.stamp.nanosec) * 1e-9;
        if (imu_time < meas.lidar_beg_time -
                           (meas.lidar->points.front().curvature + meas.lidar->points.back().curvature) /
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

    /*** push uwb data, and pop from uwb buffer ***/
    if (USE_UWB && !uwb_buffer.empty()) {
        meas.uwb.clear();
        double uwb_time = uwb_buffer.front().first;
        while ((!uwb_buffer.empty()) && (uwb_time < lidar_end_time)) {
            uwb_time = uwb_buffer.front().first;
            if (uwb_time > lidar_end_time) break;
            meas.uwb.push_back(uwb_buffer.front());
            uwb_buffer.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

// 观测模型
void RobotsLocalizationNode::h_share_model(state_ikfom& s,
                                           esekfom::dyn_share_datastruct<double>& ekfom_data) {
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
  for (int i = 0; i < feats_down_size; i++)
  {
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

    if (ekfom_data.converge)
    {
      /** Find the closest surfaces in the map **/
      // 在地图中找到与之最邻近的平面，world系下从ikdtree找NUM_MATCH_POINTS个最近点用于平面拟合
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，
      // 则认为该点不是有效点
      // 判断是否是有效匹配点，与LOAM系列类似，要求特征点最近邻的地图点数量大于阈值A，距离小于阈值B
      point_selected_surf[i] =
          points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
    }
    if (!point_selected_surf[i])
      continue;  // 如果该点不是有效点

    VF(4) pabcd;                     // 法向量
    point_selected_surf[i] = false;  // 二次筛选平面点
    // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
    if (esti_plane(pabcd, points_near, 0.1f))
    {  // 计算平面法向量
      // 根据它计算过程推测points_near的原点应该是这几个点中的一个，拟合了平面之后原点也就近似在平面
      // 上了，这样下面算出来的投影就是点到平面的距离。
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                  pabcd(3);  // 计算点到平面的距离
      // 发射距离越长，测量误差越大，归一化，消除雷达点发射距离的影响
      // p_body是激光雷达坐标系下的点
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());  // 判断残差阈值

      if (s > 0.9)
      {  // 如果残差大于阈值，则认为该点是有效点
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
  for (int i = 0; i < feats_down_size; i++)
  {
    if (point_selected_surf[i])
    {  // 只保留有效的特征点
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      total_residual += res_last[i];  // 计算总残差
      effct_feat_num++;
    }
  }
  if (effct_feat_num < 1)
  {
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
  for (int i = 0; i < effct_feat_num; i++)
  {
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

    if (extrinsic_est_en)
    {                                                             // extrinsic_est_en: IMU,lidar外参在线更新
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // Lidar坐标系下点向量在法向上的投影
      // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B),
          VEC_FROM_ARRAY(C);
    }
    else
    {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
}

void RobotsLocalizationNode::loadConfig() {
    // 参数加载
    this->declare_parameter<std::string>("common.namespace", "robot_0");
    ns = this->get_parameter("common.namespace").as_string();
    this->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
    lid_topic = this->get_parameter("common.lid_topic").as_string();
    this->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
    imu_topic = this->get_parameter("common.imu_topic").as_string();
    this->declare_parameter<std::string>("common.reloc_topic", "/slaver/robot_command");
    reloc_topic = this->get_parameter("common.reloc_topic").as_string();
    this->declare_parameter<std::string>("common.mode_topic", "/slaver/robot_status");
    mode_topic = this->get_parameter("common.mode_topic").as_string();
    this->declare_parameter<std::string>("pcd_path",
                                         "/home/luo1imasi/Documents/loc_env/ws_loc/install/"
                                         "robots_localization/share/robots_localization/PCD/map.pcd");
    pcd_path = this->get_parameter("pcd_path").as_string();
    this->declare_parameter<bool>("common.time_sync_en", false);
    time_sync_en = this->get_parameter("common.time_sync_en").as_bool();
    this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
    time_diff_lidar_to_imu = this->get_parameter("common.time_offset_lidar_to_imu").as_double();

    this->declare_parameter<int>("preprocess.lidar_type", AVIA);
    p_lidar->lidar_type = this->get_parameter("preprocess.lidar_type").as_int();
    this->declare_parameter<int>("preprocess.scan_line", 16);
    p_lidar->N_SCANS = this->get_parameter("preprocess.scan_line").as_int();
    this->declare_parameter<int>("preprocess.scan_rate", 10);
    p_lidar->SCAN_RATE = this->get_parameter("preprocess.scan_rate").as_int();
    this->declare_parameter<int>("preprocess.timestamp_unit", US);
    p_lidar->time_unit = this->get_parameter("preprocess.timestamp_unit").as_int();
    this->declare_parameter<double>("preprocess.blind", 0.01);
    p_lidar->blind = this->get_parameter("preprocess.blind").as_double();
    this->declare_parameter<int>("point_filter_num", 2);
    p_lidar->point_filter_num = this->get_parameter("point_filter_num").as_int();

    this->declare_parameter<double>("mapping.gyr_cov", 0.1);
    gyr_cov = this->get_parameter("mapping.gyr_cov").as_double();
    this->declare_parameter<double>("mapping.acc_cov", 0.1);
    acc_cov = this->get_parameter("mapping.acc_cov").as_double();
    this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
    b_gyr_cov = this->get_parameter("mapping.b_gyr_cov").as_double();
    this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
    b_acc_cov = this->get_parameter("mapping.b_acc_cov").as_double();
    this->declare_parameter<double>("mapping.fov_degree", 180.0);
    fov_deg = this->get_parameter("mapping.fov_degree").as_double();
    this->declare_parameter<double>("mapping.det_range", 300.0);  // Changed to double as param is double
    det_range = this->get_parameter("mapping.det_range").as_double();
    this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
    extrinsic_est_en = this->get_parameter("mapping.extrinsic_est_en").as_bool();
    this->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
    extrinT = this->get_parameter("mapping.extrinsic_T").as_double_array();
    this->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());
    extrinR = this->get_parameter("mapping.extrinsic_R").as_double_array();
    this->declare_parameter<bool>("mapping.mapping_en", false);
    mapping_en = this->get_parameter("mapping.mapping_en").as_bool();
    p_imu->mapping_en = mapping_en;

    this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
    pcd_save_en = this->get_parameter("pcd_save.pcd_save_en").as_bool();
    this->declare_parameter<int>("pcd_save.interval", -1);
    pcd_save_interval = this->get_parameter("pcd_save.interval").as_int();

    this->declare_parameter<bool>("publish.path_en", true);
    path_en = this->get_parameter("publish.path_en").as_bool();
    this->declare_parameter<std::vector<double>>("publish.imu2robot_T", std::vector<double>());
    imu2robotT = this->get_parameter("publish.imu2robot_T").as_double_array();
    this->declare_parameter<std::vector<double>>("publish.imu2robot_R", std::vector<double>());
    imu2robotR = this->get_parameter("publish.imu2robot_R").as_double_array();
    this->declare_parameter<bool>("publish.scan_publish_en", true);
    scan_pub_en = this->get_parameter("publish.scan_publish_en").as_bool();
    this->declare_parameter<bool>("publish.dense_publish_en", true);
    dense_pub_en = this->get_parameter("publish.dense_publish_en").as_bool();
    this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
    scan_body_pub_en = this->get_parameter("publish.scan_bodyframe_pub_en").as_bool();

    this->declare_parameter<int>("max_iteration", 4);
    NUM_MAX_ITERATIONS = this->get_parameter("max_iteration").as_int();
    this->declare_parameter<double>("filter_size_surf", 0.5);
    filter_size_surf_min = this->get_parameter("filter_size_surf").as_double();
    this->declare_parameter<double>("filter_size_map", 0.5);
    filter_size_map_min = this->get_parameter("filter_size_map").as_double();
    this->declare_parameter<double>("cube_side_length", 200.0);
    cube_len = this->get_parameter("cube_side_length").as_double();
    this->declare_parameter<bool>("runtime_pos_log_enable", false);
    runtime_pos_log = this->get_parameter("runtime_pos_log_enable").as_bool();

    this->declare_parameter<std::string>("init_method", "PPICP");
    p_imu->method = this->get_parameter("init_method").as_string();
    if (p_imu->method == "ICP") {
        this->declare_parameter<int>("ICP.max_iter", 10);
        ScanAligner::max_iter = this->get_parameter("ICP.max_iter").as_int();
    } else if (p_imu->method == "PPICP") {
        this->declare_parameter<double>("PPICP.plane_dist", 0.1f);
        ScanAligner::plane_dist = static_cast<float>(this->get_parameter("PPICP.plane_dist").as_double());
        this->declare_parameter<int>("PPICP.max_iter", 10);
        ScanAligner::max_iter = this->get_parameter("PPICP.max_iter").as_int();
    } else {
        RCLCPP_ERROR(this->get_logger(), "Not valid init method! Provided: %s", p_imu->method.c_str());
    }
    std::vector<double> tmp;
    this->declare_parameter<std::vector<double>>("prior.prior_T", std::vector<double>(3, 0.0f));
    tmp = this->get_parameter("prior.prior_T").as_double_array();
    std::transform(tmp.begin(), tmp.end(), priorT.begin(),
                   [](double val) { return static_cast<float>(val); });
    this->declare_parameter<std::vector<double>>("prior.prior_R", std::vector<double>(9, 0.0f));
    tmp = this->get_parameter("prior.prior_R").as_double_array();
    std::transform(tmp.begin(), tmp.end(), priorR.begin(),
                   [](double val) { return static_cast<float>(val); });
    this->declare_parameter<bool>("prior.estimateGrav", true);
    p_imu->estimateGrav = this->get_parameter("prior.estimateGrav").as_bool();

    this->declare_parameter<bool>("uwb.use_uwb", false);
    USE_UWB = this->get_parameter("uwb.use_uwb").as_bool();
    this->declare_parameter<std::string>("uwb.uwb_topic", "/linktrack4/nlink_linktrack_nodeframe2");
    uwb_topic = this->get_parameter("uwb.uwb_topic").as_string();
    this->declare_parameter<std::string>("uwb.mocap_topic", "/mavros/vision_pose/pose04");
    mocap_topic = this->get_parameter("uwb.mocap_topic").as_string();
    this->declare_parameter<bool>("uwb.log_mocap_traj", false);
    log_mocap_traj = this->get_parameter("uwb.log_mocap_traj").as_bool();
    this->declare_parameter<bool>("uwb.log_fusion_traj", false);
    log_fusion_traj = this->get_parameter("uwb.log_fusion_traj").as_bool();

    this->declare_parameter<bool>("uwb.esti_uwb_scale", false);
    esti_uwb_scale = this->get_parameter("uwb.esti_uwb_scale").as_bool();
    this->declare_parameter<bool>("uwb.esti_uwb_bias", true);
    esti_uwb_bias = this->get_parameter("uwb.esti_uwb_bias").as_bool();
    this->declare_parameter<bool>("uwb.esti_uwb_offset", false);
    esti_uwb_offset = this->get_parameter("uwb.esti_uwb_offset").as_bool();
    this->declare_parameter<bool>("uwb.esti_uwb_anchor", false);
    esti_uwb_anchor = this->get_parameter("uwb.esti_uwb_anchor").as_bool();
    this->declare_parameter<bool>("uwb.estimate_td", false);
    estimate_td = this->get_parameter("uwb.estimate_td").as_bool();
    this->declare_parameter<double>("uwb.td", 0.0);
    uwb_to_imu_td = this->get_parameter("uwb.td").as_double();
    this->declare_parameter<double>("uwb.td_std", 0.0);
    td_std = this->get_parameter("uwb.td_std").as_double();
    this->declare_parameter<double>("uwb.uwb_range_std", 0.2);
    uwb_range_std = this->get_parameter("uwb.uwb_range_std").as_double();
    this->declare_parameter<double>("uwb.uwb_chi2_threshold", 3.841);
    uwb_chi2_threshold = this->get_parameter("uwb.uwb_chi2_threshold").as_double();

    this->declare_parameter<std::vector<double>>("uwb.uwb2imuT", std::vector<double>{0.0, 0.0, 0.0});
    uwb2imuT = this->get_parameter("uwb.uwb2imuT").as_double_array();
    this->declare_parameter<double>("uwb.tag_move_threshold", 0.05);
    tag_move_threshold = this->get_parameter("uwb.tag_move_threshold").as_double();
    this->declare_parameter<int>("uwb.offline_calib_data_num", 10000);
    offline_calib_data_num = this->get_parameter("uwb.offline_calib_data_num").as_int();
    this->declare_parameter<std::vector<double>>("uwb.offline_calib_move_3d_threshold",
                                                 std::vector<double>(3, 0.0));
    offline_calib_move_3d_threshold =
        this->get_parameter("uwb.offline_calib_move_3d_threshold").as_double_array();
    this->declare_parameter<std::vector<double>>("uwb.offline_calib_maxmin_3d_threshold",
                                                 std::vector<double>(3, 0.0));
    offline_calib_maxmin_3d_threshold =
        this->get_parameter("uwb.offline_calib_maxmin_3d_threshold").as_double_array();
    this->declare_parameter<int>("uwb.init_bias_data_num", 200);
    init_bias_data_num = this->get_parameter("uwb.init_bias_data_num").as_int();
    this->declare_parameter<std::vector<double>>("uwb.online_calib_move_3d_threshold",
                                                 std::vector<double>(3, 0.0));
    online_calib_move_3d_threshold =
        this->get_parameter("uwb.online_calib_move_3d_threshold").as_double_array();
    this->declare_parameter<double>("uwb.near_measure_dist_threshold", 2.0);
    near_measure_dist_threshold = this->get_parameter("uwb.near_measure_dist_threshold").as_double();
    this->declare_parameter<int>("uwb.near_measure_num_threshold", 300);
    near_measure_num_threshold = this->get_parameter("uwb.near_measure_num_threshold").as_int();
    this->declare_parameter<bool>("uwb.constrain_bias", false);
    constrain_bias = this->get_parameter("uwb.constrain_bias").as_bool();
    this->declare_parameter<double>("uwb.bias_limit", 0.8);
    bias_limit = this->get_parameter("uwb.bias_limit").as_double();
    this->declare_parameter<int>("uwb.calib_data_group_num", 2);
    calib_data_group_num = this->get_parameter("uwb.calib_data_group_num").as_int();
    this->declare_parameter<double>("uwb.consistent_threshold", 0.5);
    consistent_threshold = this->get_parameter("uwb.consistent_threshold").as_double();
    this->declare_parameter<int>("uwb.ransac_sample_num", 100);
    ransac_sample_num = this->get_parameter("uwb.ransac_sample_num").as_int();
    this->declare_parameter<double>("uwb.ransac_average_error_threshold", 0.03);
    ransac_average_error_threshold = this->get_parameter("uwb.ransac_average_error_threshold").as_double();
    this->declare_parameter<bool>("uwb.use_calibrated_anchor", false);
    use_calibrated_anchor = this->get_parameter("uwb.use_calibrated_anchor").as_bool();
    this->declare_parameter<std::vector<double>>("uwb.calibrated_anchor1", std::vector<double>(3, 0.0));
    calibrated_anchor1 = this->get_parameter("uwb.calibrated_anchor1").as_double_array();
    this->declare_parameter<std::vector<double>>("uwb.calibrated_anchor2", std::vector<double>(3, 0.0));
    calibrated_anchor2 = this->get_parameter("uwb.calibrated_anchor2").as_double_array();
    this->declare_parameter<std::vector<double>>("uwb.calibrated_anchor3", std::vector<double>(3, 0.0));
    calibrated_anchor3 = this->get_parameter("uwb.calibrated_anchor3").as_double_array();
    this->declare_parameter<std::vector<double>>("uwb.calibrated_anchor4", std::vector<double>(3, 0.0));
    calibrated_anchor4 = this->get_parameter("uwb.calibrated_anchor4").as_double_array();

    this->declare_parameter<bool>("zupt.use_zupt", false);
    USE_ZUPT = this->get_parameter("zupt.use_zupt").as_bool();
    this->declare_parameter<double>("zupt.zupt_duration", 2.0);
    zupt_duration = this->get_parameter("zupt.zupt_duration").as_double();
    this->declare_parameter<int>("zupt.zupt_imu_data_num", 20);
    zupt_imu_data_num = this->get_parameter("zupt.zupt_imu_data_num").as_int();
    this->declare_parameter<double>("zupt.zupt_max_acc_std", 0.02);
    zupt_max_acc_std = this->get_parameter("zupt.zupt_max_acc_std").as_double();
    this->declare_parameter<double>("zupt.zupt_max_gyr_std", 0.001);
    zupt_max_gyr_std = this->get_parameter("zupt.zupt_max_gyr_std").as_double();
    this->declare_parameter<double>("zupt.zupt_max_gyr_median", 0.005);
    zupt_max_gyr_median = this->get_parameter("zupt.zupt_max_gyr_median").as_double();
    this->declare_parameter<double>("zupt.zupt_gyr_std", 0.02);
    zupt_gyr_std = this->get_parameter("zupt.zupt_gyr_std").as_double();
    this->declare_parameter<double>("zupt.zupt_vel_std", 0.02);
    zupt_vel_std = this->get_parameter("zupt.zupt_vel_std").as_double();
    this->declare_parameter<double>("zupt.zupt_acc_std", 0.02);
    zupt_acc_std = this->get_parameter("zupt.zupt_acc_std").as_double();
    this->declare_parameter<double>("zupt.zupt_chi2_threshold", 1.6);
    zupt_chi2_threshold = this->get_parameter("zupt.zupt_chi2_threshold").as_double();
}

void RobotsLocalizationNode::mainProcess() {
    if (!pose_inited && !mapping_en) {
        p_imu->set_init_pose(prior_T, prior_R);
        pose_inited = true;
    }

    if (sync_packages(Measures))  // 在Measure内，储存当前lidar数据及lidar扫描时间内对应的imu数据序列
    {
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

        // 重定位
        if (need_reloc && !mapping_en) {
            if (reloc_initT.norm() > 0.1) {
                if (!initT_flag) {
                    std::cout << "!!!!!!!!!start relocalization!!!!!!!!!" << std::endl;
                    std::cout << "reloc_initT: " << reloc_initT << std::endl;
                    initialized = false;
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
                if (p_imu->imu_need_init_) {
                    /// The very first lidar frame
                    p_imu->imu_init(Measures, kf, p_imu->init_iter_num);
                    return;
                }
                {
                    while (!initialized) {
                        initializing_pose = true;
                        initialized = p_imu->init_pose(Measures, kf, global_map, ikdtree, YAW_RANGE);
                        if (init_num > 15) {
                            initT_flag = false;
                            std::cout << "init failed, reset" << std::endl;
                            initializing_pose = false;
                            return;
                        }
                    }
                    initT_flag = false;
                    initializing_pose = false;
                    init_num = 0;
                }
                need_reloc = false;
                return;
            }
        }

        if (p_imu->imu_need_init_) {
            /// The very first lidar frame
            p_imu->imu_init(Measures, kf, p_imu->init_iter_num);
            return;
        }
        // 初始化位姿
        while (!initialized) {
            initializing_pose = true;
            initialized = p_imu->init_pose(Measures, kf, global_map, ikdtree, YAW_RANGE);
            init_num += 1;
            if (init_num > 15) {
                p_imu->reset();
                p_imu->set_init_pose(prior_T, prior_R);
                p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
                p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
                p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
                p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
                p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
                std::cout << "init failed, reset" << std::endl;
                initializing_pose = false;
                return;
            }
        }
        initializing_pose = false;
        init_num = 0;
        sig_buffer.notify_all();

        // 对IMU数据进行预处理，其中包含了前向传播、点云畸变处理
        // feats_undistort 为畸变纠正之后的点云,lidar系
        p_imu->process(Measures, kf, *feats_undistort);
        state_point = kf.get_x();  // 前向传播后body的状态预测值
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        if (feats_undistort->empty() || (feats_undistort == NULL)) {
            std::cout << "No point, skip this scan!" << std::endl;
            return;
        }
        if (mapping_en) {
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
        }

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);  // 降采样
        feats_down_size = feats_down_body->points.size();

        if (ikdtree.Root_Node == nullptr && mapping_en) {
            if (feats_down_size > 5) {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++) {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
            }
            return;
        }

        if (feats_down_size < 5) {
            std::cout << "No point, skip this scan!" << std::endl;
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
        SO3 rot_robot = state_point.rot * Robot_R_wrt_IMU;
        geoQuat.x = rot_robot.coeffs()[0];
        geoQuat.y = rot_robot.coeffs()[1];
        geoQuat.z = rot_robot.coeffs()[2];
        geoQuat.w = rot_robot.coeffs()[3];
        kf.change_x_imu();
        imu_only_ready = true;

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        if (mapping_en) {
            map_incremental();
        }

        /******* Publish points *******/
        if (path_en) publish_path(pubPath);
        if (scan_pub_en && first_pub && !mapping_en) {
            publish_frame_global(pubLaserCloudFull_global);
            first_pub = false;
        }
        if (scan_pub_en) {
            publish_frame_world_local(pubLaserCloudFull_world);
        }
        if (scan_pub_en && scan_body_pub_en) {
            publish_frame_body(pubLaserCloudFull_body);
        }
        if (scan_pub_en && mapping_en) {
            publish_frame_global(pubLaserCloudFull_global);
        }
    }
}

void RobotsLocalizationNode::mainProcessThread() {
    rclcpp::WallRate rate(500);
    while (rclcpp::ok()) {
        if (flg_exit) {
            break;
        }
        mainProcess();
        rate.sleep();
    }
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en && mapping_en) {
        string file_name = string("_scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + p_imu->timeStr + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << p_imu->timeStr + file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);

        PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;

        string file_name_ = string("_scans_ikdtree.pcd");
        string all_points_dir_(string(string(ROOT_DIR) + "PCD/") + p_imu->timeStr + file_name_);
        pcl::PCDWriter pcd_writer_;
        cout << "current scan saved to /PCD/" << p_imu->timeStr + file_name_ << endl;
        pcd_writer.writeBinary(all_points_dir_, *featsFromMap);
    }
    fout_pose.close();
    fout_mocap.close();
    fout_fusion.close();
    fout_uwb_calib.close();
    ikdtree.~KD_TREE();
    return;
}

void RobotsLocalizationNode::initializeUWB() {
    IMU_T_wrt_UWB << VEC_FROM_ARRAY(uwb2imuT);

    for (int i = 1; i <= UWB_ANCHOR_NUM; ++i) {
        UWBAnchor anchor(i);

        anchor_map[i] = anchor;
        anchor_id_state_index[i] = 26 + 5 * (i - 1);  // 每个uwb anchor状态变量在系统状态向量中的下标起始值
    }

    // 设置UWB标签与IMU之间的平移外参
    state_ikfom uwb_state = kf.get_x();
    uwb_state.offset_T_I_U = vect3(IMU_T_wrt_UWB);
    // 设置UWB与IMU时间系统偏移参数uwb_to_imu_td
    Eigen::VectorXd td(1);
    if (estimate_td)
        td << uwb_to_imu_td;
    else
        td << 0.0;
    uwb_state.td = vect1(td);
    std::cout << "init td: " << uwb_state.td[0] << std::endl;

    // 使用离线标定过的UWB锚点坐标
    if (use_calibrated_anchor) {
        Eigen::VectorXd calibrated_anchor(5);
        calibrated_anchor << calibrated_anchor1[0], calibrated_anchor1[1], calibrated_anchor1[2], 1.0, 0.0;
        uwb_state.anchor1 = vect5(calibrated_anchor);
        anchor_map[1].position = calibrated_anchor.head<3>();
        calibrated_anchor << calibrated_anchor2[0], calibrated_anchor2[1], calibrated_anchor2[2], 1.0, 0.0;
        uwb_state.anchor2 = vect5(calibrated_anchor);
        anchor_map[2].position = calibrated_anchor.head<3>();
        calibrated_anchor << calibrated_anchor3[0], calibrated_anchor3[1], calibrated_anchor3[2], 1.0, 0.0;
        uwb_state.anchor3 = vect5(calibrated_anchor);
        anchor_map[3].position = calibrated_anchor.head<3>();
        calibrated_anchor << calibrated_anchor4[0], calibrated_anchor4[1], calibrated_anchor4[2], 1.0, 0.0;
        uwb_state.anchor4 = vect5(calibrated_anchor);
        anchor_map[4].position = calibrated_anchor.head<3>();
    }
    move_3d_threshold << offline_calib_move_3d_threshold[0], offline_calib_move_3d_threshold[1],
        offline_calib_move_3d_threshold[2];
    maxmin_3d_threshold << offline_calib_maxmin_3d_threshold[0], offline_calib_maxmin_3d_threshold[1],
        offline_calib_maxmin_3d_threshold[2];
    kf.change_x(uwb_state);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotsLocalizationNode>();
    rclcpp::spin(node);
    return 0;
}
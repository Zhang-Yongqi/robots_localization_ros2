#include "lidar_processor.h"

LidarProcessor::LidarProcessor()
{
}

LidarProcessor::~LidarProcessor()
{
}

void LidarProcessor::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    avia_handler(msg);
    *pcl_out = pl_surf;
}

void LidarProcessor::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    switch (time_unit)
    {
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

    switch (lidar_type)
    {
    case OUST64:
        oust64_handler(msg);
        break;
    case VELO16:
        velodyne_handler(msg);
        break;
    case UNIL2:
        unilidar_handler(msg);
        break;
    case RS:
        rs_handler(msg);
        break;
    default:
        printf("Error LiDAR Type");
        break;
    }
    *pcl_out = pl_surf;
}

void LidarProcessor::rs_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<rslidar_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE; // scan angular velocity
    std::vector<bool> is_first(N_SCANS, true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);  // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0); // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].timestamp > 0) // todo check pl_orig.points[plsize - 1].time
    {
        given_offset_time = true;
        // std::cout << "given_offset_time = true " << std::endl;
    }
    else
    {
        given_offset_time = false;
        double yaw_first =
            atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578; // 记录第一个点(index 0)的yaw， to degree
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring; // 第一个点(index 0)的layer序号
        for (uint i = plsize - 1; i > 0; i--)     // 倒序遍历，找到与第一个点相同layer的最后一个点
        {
            if (pl_orig.points[i].ring == layer_first)
            {
                yaw_end =
                    atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578; // 与第一个点相同layer的最后一个点的yaw
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature =
            (pl_orig.points[i].timestamp - pl_orig.points[0].timestamp) * time_unit_scale; // curvature unit: ms
        // std::cout << "added_pt.curvature:" << added_pt.curvature << std::endl;

        if (!given_offset_time)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])
            {
                // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
            {
                pl_surf.points.push_back(added_pt);
            }
        }
    }
}

void LidarProcessor::unilidar_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<unilidar_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0)
        return;

    pl_surf.reserve(plsize);

    // std::cout << "plsize = " << plsize << ", given_offset_time = " << given_offset_time << std::endl;
    int countElimnated = 0;
    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;

        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;

        added_pt.intensity = pl_orig.points[i].intensity;

        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;

        if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
        {
            pl_surf.points.push_back(added_pt);
        }
        else
        {
            countElimnated++;
        }
    }

    // std::cout << "pl_surf.size() = " << pl_surf.size() << ", countElimnated = " << countElimnated <<
    // std::endl;
}

void LidarProcessor::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    int plsize = msg->point_num;
    if (plsize == 0)
        return;
    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);
    pl_full.resize(plsize);

    for (int i = 0; i < N_SCANS; i++)
    {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
    }
    uint valid_num = 0;

    for (uint i = 1; i < plsize; i++)
    {
        if ((msg->points[i].line < N_SCANS) &&
            ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            valid_num++;
            if (valid_num % point_filter_num == 0)
            {
                pl_full[i].x = msg->points[i].x;
                pl_full[i].y = msg->points[i].y;
                pl_full[i].z = msg->points[i].z;
                pl_full[i].intensity = msg->points[i].reflectivity;
                pl_full[i].curvature = msg->points[i].offset_time / float(1000000);
                // use curvature as time of each laser points, curvature unit: ms

                if (((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
                     (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7)) &&
                    (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z >
                     (blind * blind)))
                {
                    pl_surf.push_back(pl_full[i]);
                }
            }
        }
    }
}

void LidarProcessor::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    // 清除点云缓存
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    if (plsize == 0)
        return;
    pl_corn.reserve(plsize); // 角点
    pl_surf.reserve(plsize); // 面特征点
    pl_full.resize(plsize);  // 全部点

    double time_stamp = msg->header.stamp.toSec();
    // 默认不进行特征提取
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
        if (i % point_filter_num != 0)
            continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < (blind * blind))
            continue;

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
        // use curvature as time of each laser points, curvature unit: ms, w.r.t first lidar point

        pl_surf.points.push_back(added_pt);
    }
}

void LidarProcessor::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0)
        return;
    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);
    pl_full.resize(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE; // scan angular velocity
    std::vector<bool> is_first(N_SCANS, true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);  // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0); // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
        given_offset_time = true;
    }
    else
    {
        given_offset_time = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--)
        {
            if (pl_orig.points[i].ring == layer_first)
            {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++)
    {
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

        if (!given_offset_time)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])
            {
                // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
            {
                pl_surf.points.push_back(added_pt);
            }
        }
    }
}
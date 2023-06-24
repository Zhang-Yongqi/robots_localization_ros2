#include "ros/ros.h"
#include "Eigen/Dense"
#include "pcl_ros/io/pcd_io.h"
#include "pcl/common/transforms.h"

using POINT = pcl::PointXYZINormal;
using CLOUD = pcl::PointCloud<POINT>;
using CLOUD_PTR = CLOUD::Ptr;
using CLOUD_CONST_PTR = CLOUD::ConstPtr;

Eigen::Quaternionf fromAxisAngle(double angle, double x, double y, double z)
{
    // 将旋转角度转换为弧度
    double radian = angle * M_PI / 180.0;

    // 计算四元数
    Eigen::Quaternionf q;
    double s = std::sin(radian / 2);
    q.w() = std::cos(radian / 2);
    q.x() = x * s;
    q.y() = y * s;
    q.z() = z * s;

    return q;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "transform_map");
    ros::NodeHandle n("~");
    Eigen::Quaternionf rot(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0);
    Eigen::Vector3f trans1(0, 0, 0);
    std::string source_path;
    std::string output_path;

    double angle = -87.7422;
    double kx = 0.0, ky = 0.0, kz = 1.0; // k轴的坐标

    // ros::param::get("rot_w", rot.w());
    // ros::param::get("rot_x", rot.x());
    // ros::param::get("rot_y", rot.y());
    // ros::param::get("rot_z", rot.z());

    // ros::param::get("trans_x", trans1.x());
    // ros::param::get("trans_y", trans1.y());
    // ros::param::get("trans_z", trans1.z());

    ros::param::get("source_pcd", source_path);
    ros::param::get("output_pcd", output_path);
    ros::param::get("yaw_angle", angle);
    std::cout << "yaw" << angle << std::endl;
    rot = fromAxisAngle(angle, kx, ky, kz);

    CLOUD_PTR sourceCloudPtr = pcl::make_shared<CLOUD>();
    if (pcl::io::loadPCDFile(source_path, *sourceCloudPtr) == -1)
    {
        PCL_ERROR("Invalid source cloud:  %s", source_path.c_str());
        return -1;
    }
    // std::cout << "size: " << sourceCloudPtr->points.size() << std::endl;
    // Eigen::Matrix4f transform1 = Eigen::Matrix4f::Identity();
    // transform1.block<3, 1>(0, 3) = trans3;
    // Eigen::Matrix4f transform2 = Eigen::Matrix4f::Identity();
    // transform2.block<3, 3>(0, 0) = rot.matrix();
    // transform2.block<3, 1>(0, 3) = trans2;
    // Eigen::Matrix4f transform3 = Eigen::Matrix4f::Identity();
    // transform3.block<3, 1>(0, 3) = trans1;
    // auto transform = transform3 * transform2 * transform1;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rot.toRotationMatrix();
    transform.block<3, 1>(0, 3) = trans1;
    std::cout << "load done. start transform." << std::endl;
    auto start = std::chrono::steady_clock::now();
    pcl::transformPointCloud(*sourceCloudPtr, *sourceCloudPtr, transform);
    auto end = (std::chrono::steady_clock::now() - start).count() * 1e-9;
    std::cout << "transform done, time use: " << end << " s" << std::endl;
    std::cout << "output_path:   " << output_path << std::endl;
    pcl::io::savePCDFile(output_path, *sourceCloudPtr, true);
    std::cout << "save done." << std::endl;
    return EXIT_SUCCESS;
}

// position:
//       x: 2.4998083047452533
//       y: 5.014454450329978
//       z: 1.3479343670408372e-07
//     orientation:
//       x: -1.3157531104535684e-07
//       y: 2.1173643492727936e-07
//       z: -0.00017616388236661818
//       w: 0.9999999844831121

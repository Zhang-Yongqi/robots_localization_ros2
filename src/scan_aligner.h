#pragma once

#include <common_lib.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

#include <fstream>
#include <sophus/so3.hpp>

#include "ikd-Tree/ikd_Tree.h"

using namespace common_lib;
using namespace ikd_Tree;

class ScanAligner {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ScanAligner();

    ~ScanAligner();

    static int max_iter;
    static float plane_dist;
    static bool covInit;

    static std::pair<float, float> init_ndt_method(PointCloudXYZI::Ptr scan, M4F &predict_pose);
    static std::pair<float, float> init_icp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                                                   M4F &predict_pose);
    static std::pair<float, float> init_ppicp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                                                     M4F &predict_pose);
    static std::pair<float, float> init_gicp_method(M4F &predict_pose, PointCloudXYZI::Ptr source_,
                                                    PointCloudXYZI::Ptr target_);
    static bool calculate_covariances(
        const typename pcl::PointCloud<PointType>::ConstPtr &cloud,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &covariances,
        pcl::search::Search<PointType> &kdtree);

    static bool step_optimize(
        Eigen::Isometry3f &x0, Eigen::Isometry3f &delta, float &lm_lambda_,
        pcl::search::Search<PointType> &search_target_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
        PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_, float &error, float &valid_proportion);

    static bool is_converged(const Eigen::Isometry3f &delta);

    static std::pair<float, float> linearize(
        const Eigen::Isometry3f &trans, Eigen::Matrix<float, 6, 6> *H, Eigen::Matrix<float, 6, 1> *b,
        std::vector<int> &correspondences_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
        pcl::search::Search<PointType> &search_target_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
        PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_);

    static float update_correspondences(
        const Eigen::Isometry3f &trans, std::vector<int> &correspondences_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
        pcl::search::Search<PointType> &search_target_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
        PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_);

    static float compute_error(
        const Eigen::Isometry3f &trans, PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_,
        std::vector<int> &correspondences_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_);

    static Eigen::Matrix3f skew(const Eigen::Vector3f &x);
    static Eigen::Matrix3f skewd(const Eigen::Vector3f &x);
    static Eigen::Quaternionf so3_exp(const Eigen::Vector3f &omega);
    static Eigen::Isometry3f se3_exp(const Eigen::Matrix<float, 6, 1> &a);
};
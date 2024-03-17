#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#include "scan_aligner.h"
using namespace common_lib;
using namespace ikd_Tree;

ScanAligner::ScanAligner() {}

ScanAligner::~ScanAligner() {}

int ScanAligner::max_iter;
float ScanAligner::plane_dist;
bool ScanAligner::covInit = false;

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ScanAligner::source_covs_;
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ScanAligner::target_covs_;
pcl::search::KdTree<PointType> ScanAligner::search_source_;
pcl::search::KdTree<PointType> ScanAligner::search_target_;

std::pair<float, float> ScanAligner::init_ndt_method(PointCloudXYZI::Ptr scan, M4F &predict_pose) {
    return std::make_pair(0, 0);
}

std::pair<float, float> ScanAligner::init_icp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                                                     M4F &predict_pose) {
    PointCloudXYZI::Ptr trans_cloud(new PointCloudXYZI());
    int knn_num = 1;
    M3F rotation_matrix;
    V3F translation;
    rotation_matrix = predict_pose.block<3, 3>(0, 0);
    translation = predict_pose.block<3, 1>(0, 3);
    float whole_dist = 0.0;
    float p_valid_proportion = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
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
        for (size_t i = 0; i < point_num; i++) {
            auto ori_point = scan->points[i];
            // pcl::isFinite()用来检查点云中是否有NaN值
            if (!pcl::isFinite(ori_point)) continue;
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
                Eigen::Vector3f(trans_point.x, trans_point.y, trans_point.z) - closest_point;
            whole_dist += (trans_point.x - closest_point[0]) * (trans_point.x - closest_point[0]),
                (trans_point.y - closest_point[1]) * (trans_point.y - closest_point[1]),
                (trans_point.z - closest_point[2]) * (trans_point.z - closest_point[2]);

            Eigen::Matrix<float, 3, 6> J(Eigen::Matrix<float, 3, 6>::Zero());
            J.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
            J.block<3, 3>(0, 3) =
                -rotation_matrix * Sophus::SO3f::hat(Eigen::Vector3f(ori_point.x, ori_point.y, ori_point.z));
#ifdef MP_EN
#pragma omp critical
#endif
            {
                whole_dist += (trans_point.x - closest_point[0]) * (trans_point.x - closest_point[0]),
                    (trans_point.y - closest_point[1]) * (trans_point.y - closest_point[1]),
                    (trans_point.z - closest_point[2]) * (trans_point.z - closest_point[2]);
                H += J.transpose() * J;
                b += -J.transpose() * error_dist;
            }
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
    float p_num = (float)trans_cloud->size();
    float p_valid = 0.0;
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
        if (p_dist < 0.05) {  // 根据地图分辨率
            p_valid += 1.0;
        }
    }
    p_valid_proportion = p_valid / p_num;

    std::cout << "whole dist: " << whole_dist << " ";
    std::cout << "p_valid_proportion: " << p_valid_proportion << std::endl;
    return std::make_pair(whole_dist, p_valid_proportion);
}

std::pair<float, float> ScanAligner::init_ppicp_method(KD_TREE<PointType> &kdtree, PointCloudXYZI::Ptr scan,
                                                       M4F &predict_pose) {
    PointCloudXYZI::Ptr trans_cloud(new PointCloudXYZI());
    M3F rotation_matrix;
    V3F translation;
    rotation_matrix = predict_pose.block<3, 3>(0, 0);
    translation = predict_pose.block<3, 1>(0, 3);
    float whole_dist = 0.0;
    float p_valid_proportion = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
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
        for (size_t i = 0; i < point_num; i++) {
            auto ori_point = scan->points[i];
            if (!pcl::isFinite(ori_point)) continue;
            auto trans_point = trans_cloud->points[i];
            std::vector<float> dist;
            std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
            kdtree.Nearest_Search(trans_point, NUM_MATCH_POINTS, points_near, dist);

            VF(4)
            abcd;
            esti_plane(abcd, points_near, plane_dist);
            float error_dist =
                abcd[0] * trans_point.x + abcd[1] * trans_point.y + abcd[2] * trans_point.z + abcd[3];

            Eigen::Matrix<float, 1, 6> J(Eigen::Matrix<float, 1, 6>::Zero());
            J.block<1, 3>(0, 0) << abcd[0], abcd[1], abcd[2];
            Eigen::Matrix<float, 3, 3> tmp =
                -rotation_matrix * Sophus::SO3f::hat(Eigen::Vector3f(ori_point.x, ori_point.y, ori_point.z));
            J.block<1, 3>(0, 3) << abcd[0] * tmp(0, 0) + abcd[1] * tmp(1, 0) + abcd[2] * tmp(2, 0),
                abcd[0] * tmp(0, 1) + abcd[1] * tmp(1, 1) + abcd[2] * tmp(2, 1),
                abcd[0] * tmp(0, 2) + abcd[1] * tmp(1, 2) + abcd[2] * tmp(2, 2);
#ifdef MP_EN
#pragma omp critical
#endif
            {
                if (error_dist < 0.0) {
                    whole_dist += -error_dist;
                } else {
                    whole_dist += error_dist;
                }
                H += J.transpose() * J;
                b += -J.transpose() * error_dist;
            }
        }

        Eigen::Matrix<float, 6, 1> delta_x = H.ldlt().solve(b);
        translation += delta_x.block<3, 1>(0, 0);
        rotation_matrix *= Sophus::SO3f::exp(delta_x.block<3, 1>(3, 0)).matrix();
        predict_pose.block<3, 3>(0, 0) = rotation_matrix;
        predict_pose.block<3, 1>(0, 3) = translation;

        if (iter == max_iter - 1) {
            float p_num = (float)trans_cloud->size();
            float p_valid = 0.0;
            for (size_t i = 0; i < trans_cloud->size(); i++) {
                auto ori_point = scan->points[i];
                if (!pcl::isFinite(ori_point)) continue;
                auto trans_point = trans_cloud->points[i];
                std::vector<float> dist;
                std::vector<PointType, Eigen::aligned_allocator<PointType>> points_near;
                kdtree.Nearest_Search(trans_point, 1, points_near, dist);
                Eigen::Vector3f closest_point =
                    Eigen::Vector3f(points_near[0].x, points_near[0].y, points_near[0].z);
                float p_dist = (trans_point.x - closest_point[0]) * (trans_point.x - closest_point[0]) +
                               (trans_point.y - closest_point[1]) * (trans_point.y - closest_point[1]) +
                               (trans_point.z - closest_point[2]) * (trans_point.z - closest_point[2]);
                if (p_dist < 0.05) {  // 根据地图分辨率
                    p_valid += 1.0;
                }
            }
            p_valid_proportion = p_valid / p_num;
        }
    }

    std::cout << "whole dist: " << whole_dist << " ";
    std::cout << "p_valid_proportion: " << p_valid_proportion << std::endl;
    return std::make_pair(whole_dist, p_valid_proportion);
}

std::pair<float, float> ScanAligner::init_gicp_method(M4F &predict_pose, PointCloudXYZI::Ptr source_,
                                                      PointCloudXYZI::Ptr target_) {
    if (!covInit) {
        calculate_covariances(source_, source_covs_, search_source_);
        calculate_covariances(target_, target_covs_, search_target_);
        covInit = true;
    }
    Eigen::Isometry3f x0 = Eigen::Isometry3f(predict_pose);

    float lm_lambda_ = -1.0;
    bool converged_ = false;
    float error, valid_proportion;
    for (int i = 0; i < max_iter && !converged_; i++) {
        Eigen::Isometry3f delta;
        if (!step_optimize(x0, delta, lm_lambda_, search_target_, source_covs_, target_covs_, source_,
                           target_, error, valid_proportion)) {
            std::cerr << "lm not converged!!" << std::endl;
            break;
        }
        converged_ = is_converged(delta);
    }

    predict_pose = x0.matrix();

    std::cout << "whole dist: " << error << " ";
    std::cout << "p_valid_proportion: " << valid_proportion << std::endl;
    return std::pair<float, float>(error, valid_proportion);
}

bool ScanAligner::calculate_covariances(
    const typename pcl::PointCloud<PointType>::ConstPtr &cloud,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &covariances,
    pcl::search::Search<PointType> &kdtree) {
    kdtree.setInputCloud(cloud);
    covariances.resize(cloud->size());

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
    for (int i = 0; i < cloud->size(); i++) {
        std::vector<int> k_indices;
        std::vector<float> k_sq_distances;

        if (!pcl::isFinite(cloud->at(i))) {
            continue;
        }
        kdtree.nearestKSearch(cloud->at(i), 16, k_indices, k_sq_distances);

        Eigen::Matrix<float, 4, -1> neighbors(4, 16);
        for (int j = 0; j < k_indices.size(); j++) {
            neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<float>();
        }

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4f cov = neighbors * neighbors.transpose() / 16;

        // if (regularization_method_ == RegularizationMethod::NONE) {
        //     covariances[i] = cov;
        // } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        //     float lambda = 1e-3;
        //     Eigen::Matrix3f C = cov.block<3, 3>(0, 0).cast<float>() + lambda *
        //     Eigen::Matrix3f::Identity(); Eigen::Matrix3f C_inv = C.inverse(); covariances[i].setZero();
        //     covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
        // } else {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0),
                                              Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f values;

        // switch (regularization_method_) {
        // default:
        //     std::cerr << "here must not be reached" << std::endl;
        //     abort();
        // case RegularizationMethod::PLANE:
        values = Eigen::Vector3f(1, 1, 1e-3);
        // break;
        // case RegularizationMethod::MIN_EIG:
        //     values = svd.singularValues().array().max(1e-3);
        //     break;
        // case RegularizationMethod::NORMALIZED_MIN_EIG:
        //     values = svd.singularValues() / svd.singularValues().maxCoeff();
        //     values = values.array().max(1e-3);
        //     break;
        // }

        covariances[i].setZero();
        covariances[i].template block<3, 3>(0, 0) =
            svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        // }
    }

    return true;
}

bool ScanAligner::step_optimize(
    Eigen::Isometry3f &x0, Eigen::Isometry3f &delta, float &lm_lambda_,
    pcl::search::Search<PointType> &search_target_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
    PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_, float &error, float &valid_proportion) {
    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> b;
    std::vector<int> correspondences_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> mahalanobis_;
    std::pair<float, float> result = linearize(x0, &H, &b, correspondences_, mahalanobis_, search_target_,
                                               source_covs_, target_covs_, source_, target_);
    float y0 = result.first;
    valid_proportion = result.second;

    if (lm_lambda_ < 0.0) {
        lm_lambda_ = 1e-9 * H.diagonal().array().abs().maxCoeff();
    }
    float nu = 2.0;
    for (int i = 0; i < 10; i++) {
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> solver(H +
                                                       lm_lambda_ * Eigen::Matrix<float, 6, 6>::Identity());
        Eigen::Matrix<float, 6, 1> d = solver.solve(-b);

        delta = se3_exp(d);

        Eigen::Isometry3f xi = delta * x0;
        error =
            compute_error(xi, source_, target_, correspondences_, mahalanobis_, source_covs_, target_covs_);
        float rho = (y0 - error) / (d.dot(lm_lambda_ * d - b));

        if (rho < 0) {
            if (is_converged(delta)) {
                return true;
            }

            lm_lambda_ = nu * lm_lambda_;
            nu = 2 * nu;
            continue;
        }

        x0 = xi;
        lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
        return true;
    }

    return false;
}

bool ScanAligner::is_converged(const Eigen::Isometry3f &delta) {
    float rotation_epsilon_ = 2e-3;
    float transformation_epsilon_ = 5e-4;
    Eigen::Matrix3f R = delta.linear() - Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = delta.translation();

    Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
    Eigen::Vector3f t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

    return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

std::pair<float, float> ScanAligner::linearize(
    const Eigen::Isometry3f &trans, Eigen::Matrix<float, 6, 6> *H, Eigen::Matrix<float, 6, 1> *b,
    std::vector<int> &correspondences_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
    pcl::search::Search<PointType> &search_target_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
    PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_) {
    float p_valid_proportion = update_correspondences(trans, correspondences_, mahalanobis_, search_target_,
                                                      source_covs_, target_covs_, source_, target_);

    float sum_errors = 0.0;
    std::vector<Eigen::Matrix<float, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 6>>> Hs(
        MP_PROC_NUM);
    std::vector<Eigen::Matrix<float, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>> bs(
        MP_PROC_NUM);
    for (int i = 0; i < MP_PROC_NUM; i++) {
        Hs[i].setZero();
        bs[i].setZero();
    }

#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : sum_errors) schedule(guided, 8)
    for (int i = 0; i < source_->size(); i++) {
        int target_index = correspondences_[i];
        if (target_index < 0) {
            continue;
        }

        const Eigen::Vector4f mean_A = source_->at(i).getVector4fMap().template cast<float>();
        const auto &cov_A = source_covs_[i];

        const Eigen::Vector4f mean_B = target_->at(target_index).getVector4fMap().template cast<float>();
        const auto &cov_B = target_covs_[target_index];

        const Eigen::Vector4f transed_mean_A = trans * mean_A;
        const Eigen::Vector4f error = mean_B - transed_mean_A;

        sum_errors += error.transpose() * mahalanobis_[i] * error;

        if (H == nullptr || b == nullptr) {
            continue;
        }

        Eigen::Matrix<float, 4, 6> dtdx0 = Eigen::Matrix<float, 4, 6>::Zero();
        dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
        dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

        Eigen::Matrix<float, 4, 6> jlossexp = dtdx0;

        Eigen::Matrix<float, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
        Eigen::Matrix<float, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

        Hs[omp_get_thread_num()] += Hi;
        bs[omp_get_thread_num()] += bi;
    }

    if (H && b) {
        H->setZero();
        b->setZero();
        for (int i = 0; i < MP_PROC_NUM; i++) {
            (*H) += Hs[i];
            (*b) += bs[i];
        }
    }

    return std::pair<float, float>(sum_errors, p_valid_proportion);
}

float ScanAligner::update_correspondences(
    const Eigen::Isometry3f &trans, std::vector<int> &correspondences_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
    pcl::search::Search<PointType> &search_target_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_,
    PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_) {
    assert(source_covs_.size() == source_->size());
    assert(target_covs_.size() == source_->size());

    Eigen::Isometry3f trans_f = trans.cast<float>();
    correspondences_.resize(source_->size());
    std::vector<float> sq_distances_;
    sq_distances_.resize(source_->size());
    mahalanobis_.resize(source_->size());

    std::vector<int> k_indices(1);
    std::vector<float> k_sq_dists(1);
    // float corr_dist_threshold_ = 1.0;
    float corr_dist_threshold_ = std::numeric_limits<float>::max();
    float p_valid = 0;
    int p_num = source_->size();
#pragma omp parallel for num_threads(MP_PROC_NUM) firstprivate(k_indices, k_sq_dists) reduction(+ : p_valid) \
    schedule(guided, 8)
    for (int i = 0; i < source_->size(); i++) {
        PointType pt;

        pt.getVector4fMap() = trans_f * source_->at(i).getVector4fMap();
        if (!pcl::isFinite(pt)) {
            correspondences_[i] = -1;
            continue;
        }
        search_target_.nearestKSearch(pt, 1, k_indices, k_sq_dists);

        sq_distances_[i] = k_sq_dists[0];
        correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;

        if (correspondences_[i] < 0) {
            continue;
        }
        if (k_sq_dists[0] < 0.05) {  // 根据地图分辨率
            p_valid += 1.0;
        }
        const int target_index = correspondences_[i];
        const auto &cov_A = source_covs_[i];
        const auto &cov_B = target_covs_[target_index];

        Eigen::Matrix4f RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
        RCR(3, 3) = 1.0;

        mahalanobis_[i] = RCR.inverse();
        mahalanobis_[i](3, 3) = 0.0f;
    }
    float p_valid_proportion = p_valid / p_num;
    return p_valid_proportion;
}

float ScanAligner::compute_error(
    const Eigen::Isometry3f &trans, PointCloudXYZI::Ptr source_, PointCloudXYZI::Ptr target_,
    std::vector<int> &correspondences_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &mahalanobis_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &source_covs_,
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &target_covs_) {
    float sum_errors = 0.0;

#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : sum_errors) schedule(guided, 8)
    for (int i = 0; i < source_->size(); i++) {
        int target_index = correspondences_[i];
        if (target_index < 0) {
            continue;
        }

        const Eigen::Vector4f mean_A = source_->at(i).getVector4fMap().template cast<float>();
        const auto &cov_A = source_covs_[i];

        const Eigen::Vector4f mean_B = target_->at(target_index).getVector4fMap().template cast<float>();
        const auto &cov_B = target_covs_[target_index];

        const Eigen::Vector4f transed_mean_A = trans * mean_A;
        const Eigen::Vector4f error = mean_B - transed_mean_A;

        sum_errors += error.transpose() * mahalanobis_[i] * error;
    }

    return sum_errors;
}

Eigen::Matrix3f ScanAligner::skew(const Eigen::Vector3f &x) {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -x[2];
    skew(0, 2) = x[1];
    skew(1, 0) = x[2];
    skew(1, 2) = -x[0];
    skew(2, 0) = -x[1];
    skew(2, 1) = x[0];

    return skew;
}

Eigen::Matrix3f ScanAligner::skewd(const Eigen::Vector3f &x) {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -x[2];
    skew(0, 2) = x[1];
    skew(1, 0) = x[2];
    skew(1, 2) = -x[0];
    skew(2, 0) = -x[1];
    skew(2, 1) = x[0];

    return skew;
}

Eigen::Quaternionf ScanAligner::so3_exp(const Eigen::Vector3f &omega) {
    float theta_sq = omega.dot(omega);

    float theta;
    float imag_factor;
    float real_factor;
    if (theta_sq < 1e-10) {
        theta = 0;
        float theta_quad = theta_sq * theta_sq;
        imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
        real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
    } else {
        theta = std::sqrt(theta_sq);
        float half_theta = 0.5 * theta;
        imag_factor = std::sin(half_theta) / theta;
        real_factor = std::cos(half_theta);
    }

    return Eigen::Quaternionf(real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
                              imag_factor * omega.z());
}

Eigen::Isometry3f ScanAligner::se3_exp(const Eigen::Matrix<float, 6, 1> &a) {
    using std::cos;
    using std::sin;
    const Eigen::Vector3f omega = a.head<3>();

    float theta = std::sqrt(omega.dot(omega));
    const Eigen::Quaternionf so3 = so3_exp(omega);
    const Eigen::Matrix3f Omega = skewd(omega);
    const Eigen::Matrix3f Omega_sq = Omega * Omega;
    Eigen::Matrix3f V;

    if (theta < 1e-10) {
        V = so3.matrix();
        /// Note: That is an accurate expansion!
    } else {
        const float theta_sq = theta * theta;
        V = (Eigen::Matrix3f::Identity() + (1.0 - cos(theta)) / (theta_sq)*Omega +
             (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
    }

    Eigen::Isometry3f se3 = Eigen::Isometry3f::Identity();
    se3.linear() = so3.toRotationMatrix();
    se3.translation() = V * a.tail<3>();

    return se3;
}

#pragma clang diagnostic pop
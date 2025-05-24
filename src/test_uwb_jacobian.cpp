#include <iostream>
#include <vector>
#include <Eigen/Dense>

struct RobotState
{
  Eigen::Vector3d position;              // 位置 [x, y, z]
  Eigen::Matrix3d rotation;              // 旋转矩阵
  Eigen::Vector3d velocity;              // 速度 [vx, vy, vz]
  Eigen::Vector3d offset;                // UWB与IMU的偏移
  double time_delay;                     // 时延
  std::vector<Eigen::VectorXd> anchors;  // 基站状态 [x, y, z, scale, bias]
};

struct UWBMeasurement
{
  int anchor_idx;
  double distance;
};

double compute_dist_meas(const RobotState& state, const UWBMeasurement& measurement)
{
  const Eigen::VectorXd& anchor = state.anchors[measurement.anchor_idx];
  Eigen::Vector3d anchor_pos(anchor[0], anchor[1], anchor[2]);
  double scale = anchor[3];
  double bias = anchor[4];

  // 预测位置：position + rotation * offset + velocity * time_delay
  Eigen::Vector3d predicted_pos = state.position + state.rotation * state.offset + state.velocity * state.time_delay;
  Eigen::Vector3d diff = predicted_pos - anchor_pos;
  double predicted_distance = scale * diff.norm() + bias;

  return predicted_distance;
  //   return measurement.distance - predicted_distance;
}

// 解析雅克比，和融合代码保持一致
Eigen::MatrixXd compute_analytical_jacobian(const RobotState& state, const std::vector<UWBMeasurement>& measurements)
{
  int num_measurements = measurements.size();
  int num_anchors = state.anchors.size();
  int state_size = 3 + 3 + 3 + 3 + 1 + num_anchors * 5;  // pos, rot, vel, offset, td, anchors
  Eigen::MatrixXd jacobian(num_measurements, state_size);
  jacobian.setZero();

  for (int i = 0; i < num_measurements; ++i)
  {
    const UWBMeasurement& m = measurements[i];
    const Eigen::VectorXd& anchor = state.anchors[m.anchor_idx];
    Eigen::Vector3d anchor_pos(anchor[0], anchor[1], anchor[2]);
    double scale = anchor[3];

    Eigen::Vector3d predicted_pos = state.position + state.rotation * state.offset + state.velocity * state.time_delay;
    Eigen::Vector3d diff = predicted_pos - anchor_pos;
    double norm = diff.norm();
    Eigen::Vector3d direction = diff / norm;

    // 解析偏导数
    // 1. 对位置 (dh/dpos)
    jacobian.block<1, 3>(i, 0) = scale * direction.transpose();

    // 2. 对旋转 (dh/drot)，使用小角度近似
    // Eigen::Vector3d rot_deriv = scale * (state.rotation * state.offset);
    Eigen::Matrix3d skew;
    skew << 0, -state.offset[2], state.offset[1], state.offset[2], 0, -state.offset[0], -state.offset[1],
        state.offset[0], 0;
    jacobian.block<1, 3>(i, 3) = -scale * direction.transpose() * state.rotation * skew;

    // 3. 对速度 (dh/dvel)
    // jacobian.block<1, 3>(i, 6) = -scale * state.time_delay * direction.transpose();

    // 4. 对偏移 (dh/doffset)
    jacobian.block<1, 3>(i, 9) = scale * direction.transpose() * state.rotation;

    // 5. 对时延 (dh/dtd)
    jacobian(i, 12) = scale * state.velocity.dot(direction);

    // 6. 对基站参数 (dh/danchor)
    int anchor_start = 13 + m.anchor_idx * 5;
    jacobian.block<1, 3>(i, anchor_start) = -scale * direction.transpose();  // anchor position
    jacobian(i, anchor_start + 3) = norm;                                    // scale
    jacobian(i, anchor_start + 4) = 1.0;                                     // bias
  }
  return jacobian;
}

// 数值雅克比
Eigen::MatrixXd compute_numerical_jacobian(const RobotState& state, const std::vector<UWBMeasurement>& measurements,
                                           double delta)
{
  int num_measurements = measurements.size();
  int num_anchors = state.anchors.size();
  int state_size = 3 + 3 + 3 + 3 + 1 + num_anchors * 5;
  Eigen::MatrixXd jacobian(num_measurements, state_size);
  jacobian.setZero();

  for (int i = 0; i < num_measurements; ++i)
  {
    const UWBMeasurement& m = measurements[i];
    double dist_meas_base = compute_dist_meas(state, m);

    for (int j = 0; j < state_size; ++j)
    {
      RobotState perturbed_state = state;
      if (j < 3)
      {  // position
        perturbed_state.position(j) += delta;
      }
      else if (j < 6)
      {  // rotation
        Eigen::Vector3d rot_delta = Eigen::Vector3d::Zero();
        rot_delta(j - 3) = delta;
        perturbed_state.rotation *= Eigen::AngleAxisd(delta, rot_delta.normalized()).toRotationMatrix();
      }
      else if (j < 9)
      {  // velocity
         // perturbed_state.velocity(j - 6) += delta;
      }
      else if (j < 12)
      {  // offset
        perturbed_state.offset(j - 9) += delta;
      }
      else if (j == 12)
      {  // time_delay
        perturbed_state.time_delay += delta;
      }
      else
      {  // anchors
        int anchor_idx = (j - 13) / 5;
        int param_idx = (j - 13) % 5;
        perturbed_state.anchors[anchor_idx](param_idx) += delta;
      }

      double dist_meas_perturbed = compute_dist_meas(perturbed_state, m);
      jacobian(i, j) = (dist_meas_perturbed - dist_meas_base) / delta;
    }
  }
  return jacobian;
}

void test_jacobian_correctness()
{
  RobotState state;
  state.position.setZero();
  state.rotation.setIdentity();
  state.velocity.setZero();
  state.offset = Eigen::Vector3d(0.1, 0.2, 0.3);
  state.time_delay = 0.01;
  state.anchors.resize(1);
  state.anchors[0] = Eigen::VectorXd(5);
  state.anchors[0] << 1.0, 0.0, 0.0, 1.0, 0.0;  // [x, y, z, scale, bias]
                                                //   state.anchors[1] = Eigen::VectorXd(5);
                                                //   state.anchors[1] << 0.0, 1.0, 0.0, 1.0, 0.0;

  std::vector<UWBMeasurement> measurements = {
    { 0, 1.224 },  // 预期距离接近 sqrt(1^2 + 0.1^2 + 0.2^2 + 0.3^2)
    // { 1, 1.345 }   // 预期距离接近 sqrt(1^2 + 0.1^2 + 0.2^2 + 0.3^2)
  };

  Eigen::MatrixXd analytical_jacobian = compute_analytical_jacobian(state, measurements);
  double delta = 1e-6;
  Eigen::MatrixXd numerical_jacobian = compute_numerical_jacobian(state, measurements, delta);

  Eigen::MatrixXd diff = (analytical_jacobian - numerical_jacobian).cwiseAbs();
  double max_error = diff.maxCoeff();

  std::cout << "最大误差: " << max_error << std::endl;
  if (max_error < 1e-3)
  {
    std::cout << "雅克比测试通过！" << std::endl;
  }
  else
  {
    std::cout << "雅克比测试失败！" << std::endl;
    std::cout << "解析雅克比:\n" << analytical_jacobian << "\n数值雅克比:\n" << numerical_jacobian << std::endl;
  }
}

int main()
{
  test_jacobian_correctness();
  return 0;
}
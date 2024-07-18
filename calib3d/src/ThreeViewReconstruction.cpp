#include <calib3d/ThreeViewReconstruction.h>

#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>

#include <calib3d/calib_utils.h>

namespace calib3d {

ThreeViewReconstruction::ThreeViewReconstruction(const CameraSize& cam0_size,
                                                 const CameraSize& cam1_size,
                                                 const CameraSize& cam2_size,
                                                 double observation_noise)
    : observation_noise_(observation_noise), cam_sizes_({cam0_size, cam1_size, cam2_size}) {}

void ThreeViewReconstruction::reconstruct(const Observations& cam0_obs,
                                          const Observations& cam1_obs,
                                          const Observations& cam2_obs) {
  prepareCommonObservations(cam0_obs, cam1_obs, cam2_obs);
  performProjectiveReconstruction();
  performMetricRectification();
  recoverCameraCalibrations();
  prepareFinalReconstruction(cam0_obs, cam1_obs, cam2_obs);
}

void ThreeViewReconstruction::prepareCommonObservations(const Observations& cam0_obs,
                                                        const Observations& cam1_obs,
                                                        const Observations& cam2_obs) {
  const size_t max_required_capacity =
      std::min(cam0_obs.size(), std::min(cam1_obs.size(), cam2_obs.size()));

  image_pts_[0].resize(2, max_required_capacity);
  image_pts_[1].resize(2, max_required_capacity);
  image_pts_[2].resize(2, max_required_capacity);
  common_pt_ids_.reserve(max_required_capacity);

  for (const auto& [pt_id, cam0_pt] : cam0_obs) {
    const auto cam1_it = cam1_obs.find(pt_id);
    if (cam1_it == cam1_obs.end()) {
      continue;
    }

    const auto cam2_it = cam2_obs.find(pt_id);
    if (cam2_it == cam2_obs.end()) {
      continue;
    }

    const size_t idx = common_pt_ids_.size();
    image_pts_[0].col(idx) = cam0_pt - cam_sizes_[0].cast<double>();
    image_pts_[1].col(idx) = cam1_it->second - cam_sizes_[1].cast<double>() / 2.0;
    image_pts_[2].col(idx) = cam2_it->second - cam_sizes_[2].cast<double>() / 2.0;
    common_pt_ids_.push_back(pt_id);
  }
}

void ThreeViewReconstruction::performProjectiveReconstruction() {
  boost::math::chi_squared dist(2);
  const double percentile_95 = boost::math::quantile(dist, 0.95);

  const double ransac_thr = percentile_95 * observation_noise_;
  const double ransac_confidence = 0.99;
  const size_t ransac_max_iters = 1000;

  Eigen::Matrix3d F01 = findFundamentalMatrixRansac(
      image_pts_[0], image_pts_[1], ransac_thr, ransac_confidence, ransac_max_iters);

  P_[0].setZero();
  P_[0].leftCols<3>().setIdentity();
  P_[1] = getProjectionMatrixFromFundamentalMatrix(F01);

  world_pts_ = triangulatePoints(image_pts_[0], image_pts_[1], P_[0], P_[1]);

  P_[2] = findProjectionMatrixRansac(
      world_pts_, image_pts_[2], ransac_thr, ransac_confidence, ransac_max_iters);
}

Eigen::Matrix<double, 3, 4> ThreeViewReconstruction::getProjectionMatrixFromFundamentalMatrix(
    const Eigen::Matrix3d& F) {
  Eigen::JacobiSVD F_svd(F, Eigen::ComputeFullV);
  Eigen::Vector3d epipole = F_svd.matrixV().col(2);

  Eigen::Matrix<double, 3, 4> P = Eigen::Matrix<double, 3, 4>::Zero();
  P.leftCols<3>() = skewSymmetric(epipole) * F;
  P.col(3) = epipole;

  return P;
}

void ThreeViewReconstruction::performMetricRectification() {
  Eigen::Matrix4d ADQ = findAbsoluteDualQuadratic();
  findCameraMatrices(ADQ);
  Eigen::Matrix4d H = findRectificationHomography(ADQ);
  transformReconstruction(H);
}

Eigen::Matrix4d ThreeViewReconstruction::findAbsoluteDualQuadratic() const {
  using PMat = Eigen::Matrix<double, 3, 4>;

  auto DIAC_from_ADQ = [](const PMat& P, int row, int col) {
    return (P.row(row).transpose() * P.row(col)).eval();
  };

  auto DIAC_from_ADQ_flattened = [&](const PMat& P, int row, int col) {
    auto coeffs = DIAC_from_ADQ(P, row, col);
    // clang-format off
    return (Eigen::Vector<double, 10>() <<
        coeffs(0, 0),
        coeffs(0, 1) + coeffs(1, 0),
        coeffs(0, 2) + coeffs(2, 0),
        coeffs(0, 3) + coeffs(3, 0),
        coeffs(1, 1),
        coeffs(1, 2) + coeffs(2, 1),
        coeffs(1, 3) + coeffs(3, 1),
        coeffs(2, 2),
        coeffs(2, 3) + coeffs(3, 2),
        coeffs(3, 3)).finished();
    // clang-format on
  };

  auto ADQ_constraints_from_P = [&](const PMat& P) {
    Eigen::Matrix<double, 4, 10> constraints;
    // principal point at origin
    constraints.row(0) = DIAC_from_ADQ_flattened(P, 0, 2);
    constraints.row(1) = DIAC_from_ADQ_flattened(P, 1, 2);
    // zero skew
    constraints.row(2) = DIAC_from_ADQ_flattened(P, 0, 1);
    // aspect ratio == 1.
    constraints.row(3) = DIAC_from_ADQ_flattened(P, 0, 0) - DIAC_from_ADQ_flattened(P, 1, 1);

    return constraints;
  };

  Eigen::Matrix<double, 12, 10> ADQ_constraints;
  ADQ_constraints.middleRows<4>(0) = ADQ_constraints_from_P(P_[0]);
  ADQ_constraints.middleRows<4>(4) = ADQ_constraints_from_P(P_[1]);
  ADQ_constraints.middleRows<4>(8) = ADQ_constraints_from_P(P_[2]);

  Eigen::JacobiSVD ADQ_flattened_svd(ADQ_constraints, Eigen::ComputeFullV);
  Eigen::Vector<double, 10> ADQ_flattened = ADQ_flattened_svd.matrixV().col(9);

  // clang-format off
  Eigen::Matrix4d ADQ = (Eigen::Matrix4d() <<
    ADQ_flattened[0], ADQ_flattened[1], ADQ_flattened[2], ADQ_flattened[3],
    ADQ_flattened[1], ADQ_flattened[4], ADQ_flattened[5], ADQ_flattened[6],
    ADQ_flattened[2], ADQ_flattened[5], ADQ_flattened[7], ADQ_flattened[8],
    ADQ_flattened[3], ADQ_flattened[6], ADQ_flattened[8], ADQ_flattened[9]).finished();
  // clang-format on

  Eigen::JacobiSVD ADQ_svd(ADQ, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Vector4d ADQ_singvals = ADQ_svd.singularValues();
  ADQ_singvals(3) = 0.;
  ADQ = ADQ_svd.matrixU() * ADQ_singvals.asDiagonal() * ADQ_svd.matrixV().transpose();

  ADQ /= ADQ.reshaped().norm();

  return ADQ;
}

void ThreeViewReconstruction::findCameraMatrices(const Eigen::Matrix4d& ADQ) {
  for (size_t i = 0; i < 3; i++) {
    Eigen::Matrix3d DIAC = P_[i] * ADQ * P_[i].transpose();
    double focal_length = std::sqrt(DIAC(0, 0));
    K_[i] = Eigen::Vector3d(focal_length, focal_length, 1.0).asDiagonal();
  }
}

Eigen::Matrix4d ThreeViewReconstruction::findRectificationHomography(
    const Eigen::Matrix4d& ADQ) const {
  Eigen::Vector3d plane_at_infinity =
      Eigen::JacobiSVD(ADQ, Eigen::ComputeFullV).matrixV().col(3).hnormalized();

  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = K_[0];
  H.bottomRightCorner<1, 3>() = -plane_at_infinity.transpose() * K_[0];

  Eigen::Matrix3Xd potential_metric_world_pts =
      (H.inverse() * world_pts_.colwise().homogeneous()).colwise().hnormalized();

  int n_points_in_front_of_the_camera = (potential_metric_world_pts.row(2).array() > 0.).count();

  if (2 * n_points_in_front_of_the_camera < world_pts_.cols()) {
    H = H * Eigen::Vector4d(-1., -1., -1., 1.).asDiagonal();
  }

  Eigen::Matrix<double, 3, 4> metric_P1 = P_[1] * H;
  Eigen::Vector3d cam1_center =
      Eigen::JacobiSVD(metric_P1, Eigen::ComputeFullV).matrixV().col(3).hnormalized();
  double cam1_dist_from_origin = cam1_center.norm();
  H = H * Eigen::Vector4d(cam1_dist_from_origin, cam1_dist_from_origin, cam1_dist_from_origin, 1.)
              .asDiagonal();

  return H;
}

void ThreeViewReconstruction::transformReconstruction(const Eigen::Matrix4d& H) {
  for (size_t i = 0; i < 3; i++) {
    P_[i] = P_[i] * H;
    if (P_[i].leftCols<3>().determinant() < 0) {
      P_[i] *= -1.;
    }
  }

  world_pts_ = (H.inverse() * world_pts_.colwise().homogeneous()).colwise().hnormalized();
}

void ThreeViewReconstruction::recoverCameraCalibrations() {
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d cam_center =
        Eigen::JacobiSVD(P_[i], Eigen::ComputeFullV).matrixV().col(3).hnormalized();

    Eigen::Matrix<double, 3, 4> P_normalized = K_[i].inverse() * P_[i];

    Eigen::HouseholderQR<Eigen::Matrix<double, 3, 4>> qr(P_normalized);
    Eigen::Matrix3d qr_Q = qr.householderQ();
    Eigen::Matrix<double, 3, 4> qr_R = qr.matrixQR().triangularView<Eigen::Upper>();
    R_[i] = qr_Q * qr_R.diagonal().array().sign().matrix().asDiagonal();
    t_[i] = -R_[i] * cam_center;

    camera_calib_[i].intrinsics.f = K_[i](0, 0);
    camera_calib_[i].intrinsics.principal_point = cam_sizes_[i].cast<double>() / 2.;
    camera_calib_[i].extrinsics.world2cam_rot = R_[i];
    camera_calib_[i].extrinsics.world_in_cam_pos = t_[i];
  }
}

void ThreeViewReconstruction::prepareFinalReconstruction(const Observations& cam0_obs,
                                                         const Observations& cam1_obs,
                                                         const Observations& cam2_obs) {
  
}

} // namespace calib3d

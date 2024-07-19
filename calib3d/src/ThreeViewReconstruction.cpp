#include <calib3d/ThreeViewReconstruction.h>

#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <glog/logging.h>

#include <calib3d/calib_utils.h>

namespace calib3d {

ThreeViewReconstruction::ThreeViewReconstruction(const calib3d::ThreeViewReconstructionParams& params)
    : params_(params) {}

const std::map<CamId, CameraCalib>& ThreeViewReconstruction::getCameras() const { return cameras_; }

const std::map<PointId, PointData>& ThreeViewReconstruction::getPoints() const { return points_; }

void ThreeViewReconstruction::reconstruct(CamId cam0_id,
                                          const CameraSize& cam0_size,
                                          const Observations& cam0_obs,
                                          CamId cam1_id,
                                          const CameraSize& cam1_size,
                                          const Observations& cam1_obs,
                                          CamId cam2_id,
                                          const CameraSize& cam2_size,
                                          const Observations& cam2_obs) {
  LOG(INFO) << "Initializing 3-view reconstruction with cameras: " << cam0_id << ", " << cam1_id << ", " << cam2_id;

  insertCameraData(cam0_id, cam0_size, cam0_obs);
  insertCameraData(cam1_id, cam1_size, cam1_obs);
  insertCameraData(cam2_id, cam2_size, cam2_obs);

  performThreeViewReconstruction();
}

void ThreeViewReconstruction::insertCameraData(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs) {
  cameras_[cam_id].size = cam_size;
  for (const auto& [pt_id, obs] : cam_obs) {
    points_[pt_id].image_pts[cam_id] = obs;
  }
}

void ThreeViewReconstruction::performThreeViewReconstruction() {
  const auto [common_pt_ids, image_pts] = prepareCommonObservations();

  LOG(INFO) << "Performing 3-view reconstruction with " << common_pt_ids.size() << " points";

  auto [P, world_pts] = performProjectiveReconstruction(image_pts);
  auto K = performMetricRectification(P, world_pts);
  recoverCameraCalibrations(P, K);
  writeBackWorldPoints(common_pt_ids, world_pts);
  triangulateRemainingPoints();
}

std::pair<std::vector<PointId>, ThreeOf<Mat2X>> ThreeViewReconstruction::prepareCommonObservations() const {
  std::vector<PointId> common_pt_ids;
  ThreeOf<Eigen::Matrix2Xd> image_pts;

  const size_t max_required_capacity = points_.size();

  common_pt_ids.reserve(max_required_capacity);
  image_pts[0].resize(2, max_required_capacity);
  image_pts[1].resize(2, max_required_capacity);
  image_pts[2].resize(2, max_required_capacity);

  ThreeOf<CamId> cam_ids;
  ThreeOf<Vec2> principal_pt_offsets;
  auto cameras_it = cameras_.begin();
  for (size_t i = 0; i < 3; i++) {
    cam_ids[i] = cameras_it->first;
    principal_pt_offsets[i] = cameras_it->second.size.cast<double>() / 2.;
    ++cameras_it;
  }

  for (const auto& [pt_id, pt_data] : points_) {
    if (pt_data.image_pts.size() < 3) {
      continue;
    }

    const size_t idx = common_pt_ids.size();
    image_pts[0].col(idx) = pt_data.image_pts.at(cam_ids[0]) - principal_pt_offsets[0];
    image_pts[1].col(idx) = pt_data.image_pts.at(cam_ids[1]) - principal_pt_offsets[1];
    image_pts[2].col(idx) = pt_data.image_pts.at(cam_ids[2]) - principal_pt_offsets[2];
    common_pt_ids.push_back(pt_id);
  }

  image_pts[0].conservativeResize(Eigen::NoChange, common_pt_ids.size());
  image_pts[1].conservativeResize(Eigen::NoChange, common_pt_ids.size());
  image_pts[2].conservativeResize(Eigen::NoChange, common_pt_ids.size());

  return {common_pt_ids, image_pts};
}

std::pair<ThreeOf<Mat3x4>, Mat3X> ThreeViewReconstruction::performProjectiveReconstruction(
    const ThreeOf<Mat2X>& image_pts) const {
  boost::math::chi_squared dist(2);
  const double percentile_95 = boost::math::quantile(dist, 0.95);

  const double ransac_thr = percentile_95 * params_.observation_noise;
  const double ransac_confidence = 0.99;
  const size_t ransac_max_iters = 1000;

  Mat3 F01 = findFundamentalMatrixRansac(image_pts[0], image_pts[1], ransac_thr, ransac_confidence, ransac_max_iters);

  LOG(INFO) << "Fundamental matrix between first and second camera calculated:\n" << F01;

  ThreeOf<Mat3x4> P;
  P[0].setZero();
  P[0].leftCols<3>().setIdentity();
  P[1] = getProjectionMatrixFromFundamentalMatrix(F01);

  LOG(INFO) << "First camera projection matrix:\n" << P[0];
  LOG(INFO) << "Second camera projection matrix:\n" << P[1];

  Mat3X world_pts = triangulatePoints(image_pts[0], image_pts[1], P[0], P[1]);

  LOG(INFO) << "World points triangulated for projective reconstruction. First three points:\n"
            << world_pts.leftCols<3>().transpose();

  P[2] = findProjectionMatrixRansac(world_pts, image_pts[2], ransac_thr, ransac_confidence, ransac_max_iters);

  LOG(INFO) << "Third camera projection matrix:\n" << P[2];

  return {P, world_pts};
}

Mat3x4 ThreeViewReconstruction::getProjectionMatrixFromFundamentalMatrix(const Mat3& F) {
  Eigen::JacobiSVD FT_svd(F.transpose().eval(), Eigen::ComputeFullV);
  Vec3 epipole = FT_svd.matrixV().col(2);
  epipole /= epipole[2];

  Mat3x4 P = Mat3x4::Zero();
  P.leftCols<3>() = skewSymmetric(epipole) * F;
  P.col(3) = epipole;

  return normalizeMatrix(P);
}

ThreeOf<Mat3> ThreeViewReconstruction::performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts) {
  Mat4 ADQ = findAbsoluteDualQuadratic(P);
  ThreeOf<Mat3> K = findCameraMatrices(ADQ, P);
  Mat4 H = findRectifyingHomography(ADQ, K[0], P, world_pts);
  transformReconstruction(H, P, world_pts);

  return K;
}

Mat4 ThreeViewReconstruction::findAbsoluteDualQuadratic(const ThreeOf<Mat3x4>& P) {
  ThreeOf<Mat3x4> P_normed = {normalizeMatrix(P[0]), normalizeMatrix(P[1]), normalizeMatrix(P[2])};

  auto DIAC_from_ADQ = [](const Mat3x4& P, int row, int col) { return (P.row(row).transpose() * P.row(col)).eval(); };

  auto DIAC_from_ADQ_flattened = [&](const Mat3x4& P, int row, int col) {
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

  auto ADQ_constraints_from_P = [&](const Mat3x4& P) {
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
  ADQ_constraints.middleRows<4>(0) = ADQ_constraints_from_P(P_normed[0]);
  ADQ_constraints.middleRows<4>(4) = ADQ_constraints_from_P(P_normed[1]);
  ADQ_constraints.middleRows<4>(8) = ADQ_constraints_from_P(P_normed[2]);

  Eigen::JacobiSVD ADQ_flattened_svd(ADQ_constraints, Eigen::ComputeFullV);
  Eigen::Vector<double, 10> ADQ_flattened = ADQ_flattened_svd.matrixV().col(9);

  // clang-format off
  Eigen::Matrix4d ADQ = (Mat4() <<
    ADQ_flattened[0], ADQ_flattened[1], ADQ_flattened[2], ADQ_flattened[3],
    ADQ_flattened[1], ADQ_flattened[4], ADQ_flattened[5], ADQ_flattened[6],
    ADQ_flattened[2], ADQ_flattened[5], ADQ_flattened[7], ADQ_flattened[8],
    ADQ_flattened[3], ADQ_flattened[6], ADQ_flattened[8], ADQ_flattened[9]).finished();
  // clang-format on

  Eigen::JacobiSVD ADQ_svd(ADQ, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Vec4 ADQ_singvals = ADQ_svd.singularValues();
  ADQ_singvals(3) = 0.;
  ADQ = ADQ_svd.matrixU() * ADQ_singvals.asDiagonal() * ADQ_svd.matrixV().transpose();

  ADQ = normalizeMatrix(ADQ);

  LOG(INFO) << "Absolute dual quadratic (ADQ) calculated:\n" << ADQ;

  return ADQ;
}

ThreeOf<Mat3> ThreeViewReconstruction::findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P) {
  ThreeOf<Mat3> K;

  for (size_t i = 0; i < 3; i++) {
    Mat3 DIAC = P[i] * ADQ * P[i].transpose();
    DIAC /= DIAC(2, 2);

    LOG(INFO) << "Cam " << i << " DIAC:\n" << DIAC;

    double focal_length = std::sqrt(DIAC(0, 0));
    LOG(INFO) << "Focal length: " << focal_length;

    K[i] = Eigen::Vector3d(focal_length, focal_length, 1.0).asDiagonal();
    LOG(INFO) << "Camera matrix:\n" << K[i];
  }

  return K;
}

Mat4 ThreeViewReconstruction::findRectifyingHomography(const Mat4& ADQ,
                                                       const Mat3& K0,
                                                       const ThreeOf<Mat3x4>& P,
                                                       const Mat3X& world_pts) {
  Vec3 plane_at_infinity = Eigen::JacobiSVD(ADQ, Eigen::ComputeFullV).matrixV().col(3).hnormalized();

  LOG(INFO) << "Plane at infinity: " << plane_at_infinity.transpose();

  Mat4 H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = K0;
  H.bottomLeftCorner<1, 3>() = -plane_at_infinity.transpose() * K0;

  LOG(INFO) << "Rectifying H candidate:\n" << H;

  Mat3X potential_metric_world_pts = (H.inverse() * world_pts.colwise().homogeneous()).colwise().hnormalized();

  LOG(INFO) << "Probe Z axis of world_pts:\n" << potential_metric_world_pts.row(2);

  int n_points_in_front_of_the_camera = (potential_metric_world_pts.row(2).array() > 0.).count();
  LOG(INFO) << "Points in front of the camera: " << n_points_in_front_of_the_camera << " / " << world_pts.cols();

  if (2 * n_points_in_front_of_the_camera < world_pts.cols()) {
    LOG(INFO) << "Reflection detected. Multiplying the pointcloud by -1.";
    H = H * Vec4(-1., -1., -1., 1.).asDiagonal();
    LOG(INFO) << "New H candidate:\n" << H;
  }

  Mat3x4 metric_P1 = P[1] * H;
  LOG(INFO) << "Metric P1 for cam1 center calculations:\n" << metric_P1;

  Vec3 cam1_center = Eigen::JacobiSVD(metric_P1, Eigen::ComputeFullV).matrixV().col(3).hnormalized();
  LOG(INFO) << "Cam1 center: " << cam1_center.transpose();

  double cam1_dist_from_origin = cam1_center.norm();
  LOG(INFO) << "Cam1 distance from origin: " << cam1_dist_from_origin;

  H = H * Vec4(cam1_dist_from_origin, cam1_dist_from_origin, cam1_dist_from_origin, 1.).asDiagonal();
  LOG(INFO) << "Final rectifying H:\n" << H;

  return H;
}

void ThreeViewReconstruction::transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts) {
  for (size_t i = 0; i < 3; i++) {
    P[i] = P[i] * H;
    if (P[i].leftCols<3>().determinant() < 0) {
      P[i] *= -1.;
    }
    LOG(INFO) << "Rectified P " << i << ":\n" << P[i];
  }

  world_pts = (H.inverse() * world_pts.colwise().homogeneous()).colwise().hnormalized();
  LOG(INFO) << "Rectified world_pts:\n" << world_pts.leftCols<3>().transpose();
}

void ThreeViewReconstruction::recoverCameraCalibrations(const ThreeOf<Mat3x4>& P, const ThreeOf<Mat3>& K) {
  auto cameras_it = cameras_.begin();
  for (size_t i = 0; i < 3; i++) {
    auto& [cam_id, camera_calib] = *cameras_it;

    LOG(INFO) << "Recovering camera ID: " << cam_id;

    Vec3 cam_center = Eigen::JacobiSVD(P[i], Eigen::ComputeFullV).matrixV().col(3).hnormalized();

    LOG(INFO) << "Cam center: " << cam_center.transpose();

    Mat3x4 Kinv_P = normalizeMatrix(K[i].inverse() * P[i]);

    Eigen::HouseholderQR<Eigen::Matrix<double, 3, 4>> qr(Kinv_P);
    Eigen::Matrix3d qr_Q = qr.householderQ();
    Eigen::Matrix<double, 3, 4> qr_R = qr.matrixQR().triangularView<Eigen::Upper>();
    Mat3 Rmat = qr_Q * qr_R.diagonal().array().sign().matrix().asDiagonal();
    Vec3 tvec = -Rmat * cam_center;

    LOG(INFO) << "Rmat:\n" << Rmat;
    LOG(INFO) << "tvec: " << tvec.transpose();

    camera_calib.intrinsics.f = K[i](0, 0);
    camera_calib.intrinsics.principal_point = camera_calib.size.cast<double>() / 2.;
    camera_calib.extrinsics.world2cam_rot = Rmat;
    camera_calib.extrinsics.world_in_cam_pos = tvec;

    ++cameras_it;
  }
}

void ThreeViewReconstruction::writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts) {
  for (size_t i = 0; i < common_pt_ids.size(); i++) {
    PointId pt_id = common_pt_ids[i];
    points_[pt_id].world_pt = world_pts.col(i);
  }
}

void ThreeViewReconstruction::triangulateRemainingPoints() {
  std::map<CamId, Mat3x4> final_P;
  for (const auto& [cam_id, cam_calib] : cameras_) {
    final_P[cam_id] = cam_calib.P();
  }

  for (auto& [pt_id, pt_data] : points_) {
    if (pt_data.world_pt || pt_data.image_pts.size() < 2) {
      continue;
    }

    CamId cam1_id = pt_data.image_pts.begin()->first;
    CamId cam2_id = std::next(pt_data.image_pts.begin())->first;

    pt_data.world_pt = triangulatePoints(
        pt_data.image_pts.at(cam1_id), pt_data.image_pts.at(cam2_id), final_P.at(cam1_id), final_P.at(cam2_id));
  }
}

} // namespace calib3d

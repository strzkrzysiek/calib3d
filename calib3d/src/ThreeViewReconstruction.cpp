#include <calib3d/ThreeViewReconstruction.h>

#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <glog/logging.h>
#include <ranges>

#include <calib3d/CameraCalibRefinementProblem.h>
#include <calib3d/calib_utils.h>

namespace calib3d {

ThreeViewReconstruction::ThreeViewReconstruction(double observation_noise)
    : observation_noise_(observation_noise), percentile_95_(boost::math::quantile(boost::math::chi_squared(2), 0.95)),
      ransac_thr_(percentile_95_ * observation_noise_), ransac_confidence_(0.99), ransac_max_iters_(1000) {}

const std::map<CamId, CameraCalib>& ThreeViewReconstruction::getCameras() const {
  return cameras_;
}

const std::map<PointId, PointData>& ThreeViewReconstruction::getPoints() const {
  return points_;
}

void ThreeViewReconstruction::reconstruct(CamId cam0_id,
                                          const CameraSize& cam0_size,
                                          const Observations& cam0_obs,
                                          CamId cam1_id,
                                          const CameraSize& cam1_size,
                                          const Observations& cam1_obs,
                                          CamId cam2_id,
                                          const CameraSize& cam2_size,
                                          const Observations& cam2_obs) {
  CHECK(cameras_.empty());

  VLOG(1) << "Initializing 3-view reconstruction with cameras: " << cam0_id << ", " << cam1_id << ", " << cam2_id;

  insertCameraData(cam0_id, cam0_size, cam0_obs);
  insertCameraData(cam1_id, cam1_size, cam1_obs);
  insertCameraData(cam2_id, cam2_size, cam2_obs);

  performThreeViewReconstruction({cam0_id, cam1_id, cam2_id});
}

void ThreeViewReconstruction::insertCameraData(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs) {
  cameras_[cam_id].size = cam_size;
  for (const auto& [pt_id, obs] : cam_obs) {
    points_[pt_id].image_pts[cam_id] = obs;
  }
}

void ThreeViewReconstruction::performThreeViewReconstruction(const ThreeOf<CamId>& cam_ids) {
  const auto [common_pt_ids, image_pts] = prepareCommonObservations(cam_ids);

  VLOG(1) << "Performing 3-view reconstruction with " << common_pt_ids.size() << " points";

  auto [P, world_pts] = performProjectiveReconstruction(image_pts);
  auto K = performMetricRectification(P, world_pts);
  recoverCameraCalibrations(cam_ids, P, K);
  refineCameraCalibrations(cam_ids, P, world_pts, image_pts);
  fixRotationAndScale(cam_ids, world_pts);

  writeBackWorldPoints(common_pt_ids, world_pts);

  auto outlier_ids = identifyOutliers(cam_ids, world_pts, image_pts, common_pt_ids);
  retriangulateOutliers(outlier_ids);

  triangulateRemainingPoints();
}

std::pair<std::vector<PointId>, ThreeOf<Mat2X>> ThreeViewReconstruction::prepareCommonObservations(
    const ThreeOf<CamId>& cam_ids) const {
  std::vector<PointId> common_pt_ids;
  ThreeOf<Eigen::Matrix2Xd> image_pts;

  const size_t max_required_capacity = points_.size();

  common_pt_ids.reserve(max_required_capacity);
  image_pts[0].resize(2, max_required_capacity);
  image_pts[1].resize(2, max_required_capacity);
  image_pts[2].resize(2, max_required_capacity);

  ThreeOf<Vec2> principal_pt_offsets;
  for (size_t i = 0; i < 3; i++) {
    CamId cam_id = cam_ids[i];
    principal_pt_offsets[i] = cameras_.at(cam_id).size.cast<double>() / 2.;
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
  Mat3 F01 =
      findFundamentalMatrixRansac(image_pts[0], image_pts[1], ransac_thr_, ransac_confidence_, ransac_max_iters_);

  VLOG(2) << "Fundamental matrix between first and second camera calculated:\n" << F01;

  ThreeOf<Mat3x4> P;
  P[0].setZero();
  P[0].leftCols<3>().setIdentity();
  P[1] = getProjectionMatrixFromFundamentalMatrix(F01);

  VLOG(2) << "First camera projection matrix:\n" << P[0];
  VLOG(2) << "Second camera projection matrix:\n" << P[1];

  Mat3X world_pts = triangulatePoints(image_pts[0], image_pts[1], P[0], P[1]);

  P[2] = findProjectionMatrixRansac(world_pts, image_pts[2], ransac_thr_, ransac_confidence_, ransac_max_iters_);

  VLOG(2) << "Third camera projection matrix:\n" << P[2];

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
  Mat4 H = findRectifyingHomography(ADQ, K[0], world_pts);
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

  VLOG(2) << "Absolute dual quadratic (ADQ) calculated:\n" << ADQ;

  return ADQ;
}

ThreeOf<Mat3> ThreeViewReconstruction::findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P) {
  return {findCameraMatrix(ADQ, P[0]), findCameraMatrix(ADQ, P[1]), findCameraMatrix(ADQ, P[2])};
}

Mat3 ThreeViewReconstruction::findCameraMatrix(const Mat4& ADQ, const Mat3x4& P) {
  Mat3 DIAC = P * ADQ * P.transpose();
  DIAC /= DIAC(2, 2);

  VLOG(2) << "DIAC:\n" << DIAC;

  double focal_length = std::sqrt(DIAC(0, 0));
  VLOG(2) << "Focal length: " << focal_length;

  return Eigen::Vector3d(focal_length, focal_length, 1.0).asDiagonal();
}

Mat4 ThreeViewReconstruction::findRectifyingHomography(const Mat4& ADQ, const Mat3& K0, const Mat3X& world_pts) {
  Vec3 plane_at_infinity = Eigen::JacobiSVD(ADQ, Eigen::ComputeFullV).matrixV().col(3).hnormalized();

  VLOG(2) << "Plane at infinity: " << plane_at_infinity.transpose();

  Mat4 H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = K0;
  H.bottomLeftCorner<1, 3>() = -plane_at_infinity.transpose() * K0;

  VLOG(2) << "Rectifying H:\n" << H;

  Mat3X potential_metric_world_pts = (H.inverse() * world_pts.colwise().homogeneous()).colwise().hnormalized();
  int n_points_in_front_of_the_camera = (potential_metric_world_pts.row(2).array() > 0.).count();
  VLOG(2) << "Points in front of the camera: " << n_points_in_front_of_the_camera << " / " << world_pts.cols();

  if (2 * n_points_in_front_of_the_camera < world_pts.cols()) {
    VLOG(2) << "Reflection detected. Multiplying the pointcloud by -1.";
    H = H * Vec4(-1., -1., -1., 1.).asDiagonal();
    VLOG(2) << "New H:\n" << H;
  }

  return H;
}

void ThreeViewReconstruction::transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts) {
  for (size_t i = 0; i < 3; i++) {
    P[i] = P[i] * H;
    VLOG(2) << "Rectified P " << i << ":\n" << P[i];
  }

  world_pts = (H.inverse() * world_pts.colwise().homogeneous()).colwise().hnormalized();
}

void ThreeViewReconstruction::recoverCameraCalibrations(const ThreeOf<CamId>& cam_ids,
                                                        const ThreeOf<Mat3x4>& P,
                                                        const ThreeOf<Mat3>& K) {
  for (size_t i = 0; i < 3; i++) {
    CamId cam_id = cam_ids[i];

    VLOG(1) << "Recovering camera ID: " << cam_id;
    recoverCameraCalibration(P[i], K[i], cameras_.at(cam_id));
  }
}

void ThreeViewReconstruction::recoverCameraCalibration(const Mat3x4& P, const Mat3& K, CameraCalib& calib) {
  Vec3 cam_center = Eigen::JacobiSVD(P, Eigen::ComputeFullV).matrixV().col(3).hnormalized();
  VLOG(1) << "Cam center: " << cam_center.transpose();
  VLOG(1) << "Focal length: " << K(0, 0);

  Mat3x4 Kinv_P = normalizeMatrix(K.inverse() * P);
  Kinv_P *= Kinv_P.leftCols<3>().determinant();

  Eigen::HouseholderQR<Eigen::Matrix<double, 3, 4>> qr(Kinv_P);
  Eigen::Matrix3d qr_Q = qr.householderQ();
  Eigen::Matrix<double, 3, 4> qr_R = qr.matrixQR().triangularView<Eigen::Upper>();
  Mat3 Rmat = qr_Q * qr_R.diagonal().array().sign().matrix().asDiagonal();
  Vec3 tvec = -Rmat * cam_center;

  VLOG(1) << "Rotation matrix:\n" << Rmat;
  VLOG(1) << "Translation: " << tvec.transpose();

  calib.intrinsics.focal_length = K(0, 0);
  calib.intrinsics.principal_point = calib.size.cast<double>() / 2.;
  calib.world2cam.setRotationMatrix(Rmat);
  calib.world2cam.translation() = tvec;
}

void ThreeViewReconstruction::refineCameraCalibrations(const ThreeOf<CamId>& cam_ids,
                                                       const ThreeOf<Mat3x4>& P,
                                                       const Mat3X& world_pts,
                                                       const ThreeOf<Mat2X>& image_pts) {
  for (size_t i = 0; i < 3; i++) {
    CamId cam_id = cam_ids[i];

    VLOG(1) << "Refining camera ID: " << cam_id;
    refineCameraCalibration(P[i], world_pts, image_pts[i], cameras_.at(cam_id));
  }
}

void ThreeViewReconstruction::refineCameraCalibration(const Mat3x4& P,
                                                      const Mat3X& world_pts,
                                                      const Mat2X& image_pts,
                                                      CameraCalib& calib) const {
  CameraCalibRefinementProblem problem(calib, P, ransac_thr_);
  problem.addCorrespondences(world_pts, image_pts);
  problem.optimize();

  VLOG(1) << "Cam center: " << calib.world2cam.inverse().translation().transpose();
  VLOG(1) << "Focal length: " << calib.intrinsics.focal_length;
  VLOG(1) << "Rotation matrix:\n" << calib.world2cam.so3().matrix();
  VLOG(1) << "Translation: " << calib.world2cam.translation().transpose();
}

void ThreeViewReconstruction::fixRotationAndScale(const ThreeOf<CamId>& cam_ids, Mat3X& world_pts) {
  const auto& init_cam = cameras_.at(cam_ids[0]);
  SE3 world2init_cam = init_cam.world2cam;
  SE3 init_cam2world = world2init_cam.inverse();

  VLOG(1) << "Fixing orientation and scale";
  VLOG(1) << "Orientation update in degrees: " << init_cam2world.so3().logAndTheta().theta / M_PI * 180.;
  VLOG(1) << "Translation update: " << init_cam2world.translation().transpose();

  for (auto& [cam_id, cam_calib] : cameras_) {
    cam_calib.world2cam = cam_calib.world2cam * init_cam2world;
  }
  world_pts = world2init_cam.matrix3x4() * world_pts.colwise().homogeneous();

  const auto& second_cam = cameras_.at(cam_ids[1]);
  double second_cam_distance_from_origin = second_cam.world2cam.translation().norm();
  double scaling_factor = 1.0 / second_cam_distance_from_origin;
  VLOG(1) << "Scaling factor: " << scaling_factor;

  for (auto& [cam_id, cam_calib] : cameras_) {
    cam_calib.world2cam.translation() *= scaling_factor;
  }
  world_pts *= scaling_factor;
}

void ThreeViewReconstruction::writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts) {
  for (size_t i = 0; i < common_pt_ids.size(); i++) {
    PointId pt_id = common_pt_ids[i];
    points_[pt_id].world_pt = world_pts.col(i);
  }
}

std::set<PointId> ThreeViewReconstruction::identifyOutliers(const ThreeOf<CamId>& cam_ids,
                                                            const Mat3X& world_pts,
                                                            const ThreeOf<Mat2X>& image_pts,
                                                            const std::vector<PointId>& pt_ids) const {
  ThreeOf<std::set<PointId>> outlier_ids_per_camera = {identifyOutliers(cam_ids[0], world_pts, image_pts[0], pt_ids),
                                                       identifyOutliers(cam_ids[1], world_pts, image_pts[1], pt_ids),
                                                       identifyOutliers(cam_ids[2], world_pts, image_pts[2], pt_ids)};

  auto outlier_view = std::views::join(outlier_ids_per_camera);
  std::set<PointId> outlier_ids(outlier_view.begin(), outlier_view.end());

  return outlier_ids;
}

std::set<PointId> ThreeViewReconstruction::identifyOutliers(CamId cam_id,
                                                            const Mat3X& world_pts,
                                                            const Mat2X& image_pts,
                                                            const std::vector<PointId>& pt_ids) const {
  std::set<PointId> outlier_ids;
  const auto& calib = cameras_.at(cam_id);
  Mat3x4 P = calib.intrinsics.K().diagonal().asDiagonal() * calib.world2cam.matrix3x4();

  VecX err = ((P * world_pts.colwise().homogeneous()).colwise().hnormalized() - image_pts).colwise().norm();
  for (int i = 0; i < err.size(); i++) {
    if (err(i) > ransac_thr_) {
      outlier_ids.insert(pt_ids[i]);
    }
  }

  VLOG(1) << "Identified " << outlier_ids.size() << " outliers for camera " << cam_id;

  return outlier_ids;
}

void ThreeViewReconstruction::retriangulateOutliers(const std::set<PointId>& outlier_ids) {
  VLOG(1) << "Retriangulating " << outlier_ids.size() << " outliers";

  if (outlier_ids.empty()) {
    return;
  }

  std::map<CamId, size_t> cam_id_to_idx;
  Eigen::Matrix<double, 12, Eigen::Dynamic> P_flattened(12, cameras_.size());
  for (const auto& [cam_id, cam_calib] : cameras_) {
    size_t idx = cam_id_to_idx.size();
    P_flattened.col(idx) = cam_calib.P().reshaped();
    cam_id_to_idx[cam_id] = idx;
  }

  for (PointId outlier_id : outlier_ids) {
    auto& pt_data = points_.at(outlier_id);
    CHECK_GE(pt_data.image_pts.size(), 3);

    Mat2X image_pts(2, pt_data.image_pts.size());
    std::vector<size_t> P_indices;
    for (const auto& [cam_id, image_pt] : pt_data.image_pts) {
      image_pts.col(P_indices.size()) = image_pt;
      P_indices.push_back(cam_id_to_idx.at(cam_id));
    }

    pt_data.world_pt =
        triangulatePointRansac(image_pts, P_flattened(Eigen::all, P_indices), ransac_thr_, ransac_confidence_, 50);
    VLOG(2) << "New PT " << outlier_id << " pos: " << pt_data.world_pt.value().transpose();
  }
}

void ThreeViewReconstruction::triangulateRemainingPoints() {
  std::map<CamId, Mat3x4> P;
  for (const auto& [cam_id, cam_calib] : cameras_) {
    P[cam_id] = cam_calib.P();
  }

  for (auto& [pt_id, pt_data] : points_) {
    if (pt_data.world_pt || pt_data.image_pts.size() < 2) {
      continue;
    }

    CHECK_EQ(pt_data.image_pts.size(), 2);

    CamId cam1_id = pt_data.image_pts.begin()->first;
    CamId cam2_id = std::next(pt_data.image_pts.begin())->first;

    pt_data.world_pt =
        triangulatePoints(pt_data.image_pts.at(cam1_id), pt_data.image_pts.at(cam2_id), P.at(cam1_id), P.at(cam2_id));
  }
}

} // namespace calib3d

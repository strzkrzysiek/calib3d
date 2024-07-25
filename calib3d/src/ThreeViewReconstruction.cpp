// Copyright 2024 Krzysztof Wrobel

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
    : observation_noise_(observation_noise),
      // Calculate the 95th percentile of Chi2 distribution with 2 DoFs
      percentile_95_(boost::math::quantile(boost::math::chi_squared(2), 0.95)),
      // Calculate the outlier threshold for RANSAC algorithms as the 95th percentile of the observation noise
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
  // Get image points in each camera and their IDs of all the points that are visible in all three cameras
  // The image_pts variable holds shifted observations as if they were measured in cameras
  // with principle points at [0, 0]
  // In other words, half of the camera size is subtracted from the original observation coordinates.
  const auto [common_pt_ids, image_pts] = prepareCommonObservations(cam_ids);

  VLOG(1) << "Performing 3-view reconstruction with " << common_pt_ids.size() << " points";

  // Perform the projective reconstruction with common image points to get
  // projection matrices of each camera and triangulated world points
  auto [P, world_pts] = performProjectiveReconstruction(image_pts);

  // Perform metric rectification of the projection reconstruction.
  // Both, the projective matrices and world points are modified here.
  // The rectification method recovers also the camera matrices (K)
  auto K = performMetricRectification(P, world_pts);

  // Recover coarse extrinsic calibrations for each of the cameras using direct method
  recoverCameraCalibrations(cam_ids, P, K);

  // Refine the coars extrinsic and intrinsic calibration with iterative nonlinear least-squares
  refineCameraCalibrations(cam_ids, P, world_pts, image_pts);

  // Set the orientation, origin and scale of the reconstruction
  fixRotationAndScale(cam_ids, world_pts);

  // Write the world points stored in the world_pts to the points_ map
  writeBackWorldPoints(common_pt_ids, world_pts);

  // Identify the outliers that should be retriangulated
  auto outlier_ids = identifyOutliers(cam_ids, world_pts, image_pts, common_pt_ids);
  // And retriangulate them using a RANSAC method. This should correctly triangulate points for which
  // one of the observation is an outlier
  retriangulateOutliers(outlier_ids);

  // Triangulate points that are visible only in two cameras, because all the calculations above
  // were performed for points visible in all the three cameras.
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
  // Find the fundamental matrix between the first and the second camera
  Mat3 F01 =
      findFundamentalMatrixRansac(image_pts[0], image_pts[1], ransac_thr_, ransac_confidence_, ransac_max_iters_);

  VLOG(2) << "Fundamental matrix between first and second camera calculated:\n" << F01;

  // Find projection matrices in the canonical form.
  // This means that initially (for the projective reconstruction) we set the projection matrix
  // of the first camera to P[0] = [ I | 0 ]
  ThreeOf<Mat3x4> P;
  P[0].setZero();
  P[0].leftCols<3>().setIdentity();

  // Get the projection matrix of the second camera given the fundamental matrix and assuming
  // that the first camera is in the form defined above
  P[1] = getProjectionMatrixFromFundamentalMatrix(F01);

  VLOG(2) << "First camera projection matrix:\n" << P[0];
  VLOG(2) << "Second camera projection matrix:\n" << P[1];

  // Having two projection matrices we can triangulate some points
  Mat3X world_pts = triangulatePoints(image_pts[0], image_pts[1], P[0], P[1]);

  // Recover a projection matrix for the third camera by matching the triangulated 3D points with image observations
  P[2] = findProjectionMatrixRansac(world_pts, image_pts[2], ransac_thr_, ransac_confidence_, ransac_max_iters_);

  VLOG(2) << "Third camera projection matrix:\n" << P[2];

  // Return the projective reconstruction which is the projection matrices together with the triangulated points
  return {P, world_pts};
}

Mat3x4 ThreeViewReconstruction::getProjectionMatrixFromFundamentalMatrix(const Mat3& F) {
  // Compute the epipole in the second image
  // which is the left null vector of F (or right null vector of F.transpose());
  // See: Section 9.2.4 Properties of the fundamental matrix in Hartley and Zisserman (2003)
  Eigen::JacobiSVD FT_svd(F.transpose().eval(), Eigen::ComputeFullV);
  Vec3 epipole = FT_svd.matrixV().col(2);
  epipole /= epipole[2];

  // Calculate the projection matrix according to the Result 9.14 in Hartley and Zisserman (2003)
  Mat3x4 P = Mat3x4::Zero();
  P.leftCols<3>() = skewSymmetric(epipole) * F;
  P.col(3) = epipole;

  // Normalize the projection matrix for numerical stability
  return normalizeMatrix(P);
}

ThreeOf<Mat3> ThreeViewReconstruction::performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts) {
  // Perform the metric rectification by identifying the absolute dual quadric
  // in the reconstructed projective reconstruction
  Mat4 ADQ = findAbsoluteDualQuadric(P);

  // Having found the ADQ, calculate the camera matrices
  ThreeOf<Mat3> K = findCameraMatrices(ADQ, P);

  // Also, we can construct a rectifying matrix as defined in step (ii)
  // of Section 19.1 Introduction in Hartley and Zisserman (2003)
  // Given a projective reconstruction {P_i, X_j} where
  // P_i is a projection matrix for i-th camera and X_j is the j-th world point
  // The metric reconstruction can be obtained as {P_i * H, H.inverse() *  X_j }
  Mat4 H = findRectifyingHomography(ADQ, K[0], world_pts);

  // Transform the projection matrices and the world point by the calculated rectifying homography
  transformReconstruction(H, P, world_pts);

  // Return the camera matrices
  return K;
}

Mat4 ThreeViewReconstruction::findAbsoluteDualQuadric(const ThreeOf<Mat3x4>& P) {
  // This function implements the calculation of the absolute dual quadric
  // as defined in Section 3.7 The absolute dual quadric in Hartley and Zisserman (2003)
  // The equations below are derived with a help of Section 19.3 Calibration using the absolute dual quadric
  // And in particular:
  // Equation (19.6) for the relationship between ADQ and DIAC (dual image of absolute conic) and camera matrix K
  // Table 19.2 Auto-calibration constrains derived from the DIAC for the actual constrains to solve for ADQ
  // Example 19.5. Linear solution for variable focal length

  // First, be sure to have similarly normalized projection matrices - this is very important for numerical stability
  ThreeOf<Mat3x4> P_normed = {normalizeMatrix(P[0]), normalizeMatrix(P[1]), normalizeMatrix(P[2])};

  // Given the fact, that the DIAC is a projection of ADQ:
  // DIAC = P * ADQ * P.transpose()
  // DIAC is a 3x3 matrix.
  // Suppose we want to calculate one of the coefficients of DIAC, say DIAC(row, col)
  // The lambda function below returns a 4x4 matrix (constructed by transforming the equation above)
  // that multiplied element-wise with ADQ and summed gives the required coefficient of DIAC.
  // Or in other words, this matrix defines the coefficient of DIAC as a linear combination of ADQ.
  // DIAC(row, col) = DIAC_from_ADQ(P, row, col).cwiseProduct(ADQ).sum()
  auto DIAC_from_ADQ = [](const Mat3x4& P, int row, int col) { return (P.row(row).transpose() * P.row(col)).eval(); };

  // The ADQ is a symmetric matrix, which means that we may define it with 10 coefficients
  // We construct a similar function to the above that defines a single coefficient of a DIAC
  // as a linear combination of 10 distinct elements of ADQ put into a 10-element vector
  // DIAC(row, col) = DIAC_from_ADQ_flattened.dot(ADQ_flattened)
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

  // Now we have some instruments to define the constraints for ADQ_flattened given a single projection matrix P
  // The idea here is to transfer the constraints on the camera matrix to constraints on DIAC and
  // subsequently to constraints on ADQ using the two equations:
  // DIAC = K * K.transpose()
  // DIAC = P * ADQ * P.transpose()
  // This is defined per each camera as the camera matrices may vary
  auto ADQ_constraints_from_P = [&](const Mat3x4& P) {
    Eigen::Matrix<double, 4, 10> constraints;
    // The first two constraints require that the principal point is at origin:
    // K(0, 2) = 0 ==> DIAC(0, 2) = 0 ==> DIAC_from_ADQ_flattened(P, 0, 2) * ADQ_flattened = 0
    constraints.row(0) = DIAC_from_ADQ_flattened(P, 0, 2);
    // K(1, 2) = 0 ==> DIAC(1, 2) = 0 ==> DIAC_from_ADQ_flattened(P, 1, 2) * ADQ_flattened = 0
    constraints.row(1) = DIAC_from_ADQ_flattened(P, 1, 2);
    // Zero skew with principal point at origin:
    // K(0, 1) = 0 ==> DIAC(0, 1) == 0 ==> DIAC_from_ADQ_flattened(P, 0, 1) * ADQ_flattened = 0
    constraints.row(2) = DIAC_from_ADQ_flattened(P, 0, 1);
    // Aspect ratio is equal to one (or the focal length is equal on X and Y axes):
    // K(0, 0) - K(1, 1) = 0 ==> DIAC(0, 0) - DIAC(1, 1) = 0 ==>
    // ==> (DIAC_from_ADQ_flattened(P, 0, 0) - DIAC_from_ADQ_flattened(P, 1, 1)) * ADQ_flattened = 0
    constraints.row(3) = DIAC_from_ADQ_flattened(P, 0, 0) - DIAC_from_ADQ_flattened(P, 1, 1);

    return constraints;
  };

  // Build the ADQ constraints matrix with the constraints arising from all the three views
  Eigen::Matrix<double, 12, 10> ADQ_constraints;
  ADQ_constraints.middleRows<4>(0) = ADQ_constraints_from_P(P_normed[0]);
  ADQ_constraints.middleRows<4>(4) = ADQ_constraints_from_P(P_normed[1]);
  ADQ_constraints.middleRows<4>(8) = ADQ_constraints_from_P(P_normed[2]);

  // And solve the system of linear equations by finding the null vector ad ADQ_constraints
  Eigen::JacobiSVD ADQ_flattened_svd(ADQ_constraints, Eigen::ComputeFullV);
  Eigen::Vector<double, 10> ADQ_flattened = ADQ_flattened_svd.matrixV().col(9);

  // Recover the symmetric ADQ from the 10-element vector (ADQ_flattened)
  // clang-format off
  Eigen::Matrix4d ADQ = (Mat4() <<
    ADQ_flattened[0], ADQ_flattened[1], ADQ_flattened[2], ADQ_flattened[3],
    ADQ_flattened[1], ADQ_flattened[4], ADQ_flattened[5], ADQ_flattened[6],
    ADQ_flattened[2], ADQ_flattened[5], ADQ_flattened[7], ADQ_flattened[8],
    ADQ_flattened[3], ADQ_flattened[6], ADQ_flattened[8], ADQ_flattened[9]).finished();
  // clang-format on

  // ADQ should be a singular, rank 3 matrix. We impose it by setting the smallest singular value to zero
  // See: 19.3.5 Limitations of the absolute quadric approach to calibration in Hartley and Zisserman (2003)
  Eigen::JacobiSVD ADQ_svd(ADQ, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Vec4 ADQ_singvals = ADQ_svd.singularValues();
  ADQ_singvals(3) = 0.;
  ADQ = ADQ_svd.matrixU() * ADQ_singvals.asDiagonal() * ADQ_svd.matrixV().transpose();

  // Normalize for numerical stability
  ADQ = normalizeMatrix(ADQ);

  VLOG(2) << "Absolute dual quadric (ADQ) calculated:\n" << ADQ;

  return ADQ;
}

ThreeOf<Mat3> ThreeViewReconstruction::findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P) {
  return {findCameraMatrix(ADQ, P[0]), findCameraMatrix(ADQ, P[1]), findCameraMatrix(ADQ, P[2])};
}

Mat3 ThreeViewReconstruction::findCameraMatrix(const Mat4& ADQ, const Mat3x4& P) {
  // First we project the ADQ to get a dual image of absolute quardic
  // See: equation (19.6) in Hartley and Zisserman (2003)
  Mat3 DIAC = P * ADQ * P.transpose();
  // Normalize it to get DIAC(2, 2) == 1
  DIAC /= DIAC(2, 2);

  VLOG(2) << "DIAC:\n" << DIAC;

  // In theory, DIAC = K * K.transpose()
  // So to get the camera matrix K, we should calculate it with Cholesky decomposition.
  // However, this requires the DIAC to be semi-positive definite and because of numerical errors
  // it may not be always true (see: Section 19.3.5 Limitations of the absolute quadric approach to calibration)
  // But because of our constraints for K, we want to extract only the focal length, and we may reconstruct
  // the rest of the K matrix ourselves.
  // K(0, 0) = sqrt(DIAC(0, 0) and it doesn't require the Cholesky decomposition

  double focal_length = std::sqrt(DIAC(0, 0));
  VLOG(2) << "Focal length: " << focal_length;

  return Eigen::Vector3d(focal_length, focal_length, 1.0).asDiagonal();
}

Mat4 ThreeViewReconstruction::findRectifyingHomography(const Mat4& ADQ, const Mat3& K0, const Mat3X& world_pts) {
  // We construct the rectifying homography with help of Result 19.1 in Hartley and Zisserman (2003)
  // It may also be calculated by decomposing ADQ as in Result 19.4, but we choose the first method for numerical issues

  // Since the ADQ lies on the plane at infinity, we may calculate the plane at infinity as the null vector of ADQ
  Vec3 plane_at_infinity = Eigen::JacobiSVD(ADQ, Eigen::ComputeFullV).matrixV().col(3).hnormalized();

  VLOG(2) << "Plane at infinity: " << plane_at_infinity.transpose();

  // Construct the H matrix as in the equation (19.2)
  Mat4 H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = K0;
  H.bottomLeftCorner<1, 3>() = -plane_at_infinity.transpose() * K0;

  VLOG(2) << "Rectifying H:\n" << H;

  // Since the rectification is ambiguous up to an arbitrary similarity transform, it may happen
  // that the scaling factor is negative and our current reconstruction is reflected.
  // We check it by verifying if the Z-coordinate of the world points after the rectification is positive as the first
  // camera looks in the direction of Z axis and we want to have those points in front of that camera

  Mat3X potential_metric_world_pts = (H.inverse() * world_pts.colwise().homogeneous()).colwise().hnormalized();
  int n_points_in_front_of_the_camera = (potential_metric_world_pts.row(2).array() > 0.).count();
  VLOG(2) << "Points in front of the camera: " << n_points_in_front_of_the_camera << " / " << world_pts.cols();

  // Due to numerical instabilities, some points may land behind the camera, so check where is the majority
  if (2 * n_points_in_front_of_the_camera < world_pts.cols()) {
    VLOG(2) << "Reflection detected. Multiplying the pointcloud by -1.";
    H = H * Vec4(-1., -1., -1., 1.).asDiagonal();
    VLOG(2) << "New H:\n" << H;
  }

  return H;
}

void ThreeViewReconstruction::transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts) {
  // Transform the projection matrices: P_metric = P * H
  for (size_t i = 0; i < 3; i++) {
    P[i] = P[i] * H;
    VLOG(2) << "Rectified P " << i << ":\n" << P[i];
  }

  // Transform the world points: X_metric = H.inverse() * X
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
  // The camera center is the null vector of the projection matrix as the projection
  // transforms this point to (0, 0) in camera coordinates
  Vec3 cam_center = Eigen::JacobiSVD(P, Eigen::ComputeFullV).matrixV().col(3).hnormalized();
  VLOG(1) << "Cam center: " << cam_center.transpose();
  VLOG(1) << "Focal length: " << K(0, 0);

  // Remove the camera matrix from the projection matrix
  Mat3x4 Kinv_P = normalizeMatrix(K.inverse() * P);

  // The three left columns of the resulting matrix should be the rotation matrix.
  // Rotation matrix has its determinant equal to 1. So, scale Kinv_P accordingly.
  Kinv_P *= Kinv_P.leftCols<3>().determinant();

  // Because of numerical instabilities, Kinv_P.leftCols<3>() most probably is not an orthonormal matrix.
  // Use the QR decomposition to enforce it.
  Eigen::HouseholderQR<Eigen::Matrix<double, 3, 4>> qr(Kinv_P);
  Eigen::Matrix3d qr_Q = qr.householderQ();
  Eigen::Matrix<double, 3, 4> qr_R = qr.matrixQR().triangularView<Eigen::Upper>();
  Mat3 Rmat = qr_Q * qr_R.diagonal().array().sign().matrix().asDiagonal();
  Vec3 tvec = -Rmat * cam_center;

  VLOG(1) << "Rotation matrix:\n" << Rmat;
  VLOG(1) << "Translation: " << tvec.transpose();

  // Since we calculate the calibration from noisy observation, this is a very coarse estimation.
  // In particular, the projections with P are not equal to the projections with K [R | t].
  // In fact, P has 11 DoFs and K [R | t] has 7 (in our case)

  // Write back the coarse estimation to the internal structures
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
  // Refine the camera calibration with an iterative non-linear optimization
  // Only focal length and camera pose is optimized here.
  CameraCalibRefinementProblem problem(calib, P, ransac_thr_);
  problem.addCorrespondences(world_pts, image_pts);
  problem.optimize();

  VLOG(1) << "Cam center: " << calib.world2cam.inverse().translation().transpose();
  VLOG(1) << "Focal length: " << calib.intrinsics.focal_length;
  VLOG(1) << "Rotation matrix:\n" << calib.world2cam.so3().matrix();
  VLOG(1) << "Translation: " << calib.world2cam.translation().transpose();
}

void ThreeViewReconstruction::fixRotationAndScale(const ThreeOf<CamId>& cam_ids, Mat3X& world_pts) {
  // Rotate and translate the scene, so that the first camera is at origin.
  // Actually, it is meant to be at origin already, but the procedure of refining camera calibration
  // might have moved it slightly
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

  // Scale the scene, so that the first camera is at unit distance from the origin
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
  // Identify the outliers for all cameras
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
  // Construct the projection matrix
  // Here, we still use 'moved' image points as if the principal point is at [0, 0]
  // so we have to take only the diagonal of the K matrix.
  Mat3x4 P = calib.intrinsics.K().diagonal().asDiagonal() * calib.world2cam.matrix3x4();

  // Calculate the reprojection error for all the points
  VecX err = ((P * world_pts.colwise().homogeneous()).colwise().hnormalized() - image_pts).colwise().norm();
  for (int i = 0; i < err.size(); i++) {
    if (err(i) > ransac_thr_) {
      // If the reprojection error is bigger than the threshold, it is considered as an outlier
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

  // Prepare a map of flattened projection matrices that are required by the RANSAC algorithm
  std::map<CamId, size_t> cam_id_to_idx;
  Eigen::Matrix<double, 12, Eigen::Dynamic> P_flattened(12, cameras_.size());
  for (const auto& [cam_id, cam_calib] : cameras_) {
    size_t idx = cam_id_to_idx.size();
    P_flattened.col(idx) = cam_calib.P().reshaped();
    cam_id_to_idx[cam_id] = idx;
  }

  // Retriangulate the outliers
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
  // Triangulate all the points that have just received their second image correspondence

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

    // There is no need to use RANSAC here as we have only two observations
    pt_data.world_pt =
        triangulatePoints(pt_data.image_pts.at(cam1_id), pt_data.image_pts.at(cam2_id), P.at(cam1_id), P.at(cam2_id));
  }
}

} // namespace calib3d

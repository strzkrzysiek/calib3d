// Copyright 2024 Krzysztof Wrobel

#include <calib3d/BAProblem.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace calib3d {

// Regular reprojection error with automatic jacobians
// The image points are constant while the world points and camera calibrations are variable
struct ReprojectionError {
  explicit ReprojectionError(const Vec2& image_pt) : image_pt(image_pt) {}

  template <class T>
  bool operator()(const T* const raw_world2cam,
                  const T* const raw_principal_point,
                  const T* const raw_focal_length,
                  const T* const raw_world_pt,
                  T* raw_residual) const {
    Eigen::Map<const SE3T<T>> world2cam(raw_world2cam);
    Eigen::Map<const Vec2T<T>> principal_point(raw_principal_point);
    const T& focal_length = *raw_focal_length;
    Eigen::Map<const Vec3T<T>> world_pt(raw_world_pt);
    Eigen::Map<Vec2T<T>> residual(raw_residual);

    Vec2T<T> reprojected_pt = (world2cam * world_pt).hnormalized() * focal_length + principal_point;
    residual = reprojected_pt - image_pt.cast<T>();

    return true;
  }

  static ceres::CostFunction* create(const Vec2& image_pt) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, SE3::num_parameters, 2, 1, 3>(
        new ReprojectionError(image_pt));
  }

  const Vec2& image_pt;
};

struct BAProblem::Impl {
  // Special parameterization of Sophus::SE3 which is internally a quaternion followed by a 3D translation vector
  using SE3Manifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
  // Even more special parameterization for the second camera whose center is supposed to be at unit distance from
  // origin. This means that is it should be at a unit sphere centered at origin, and we may enforce it
  // by setting a SphereManifold for the translation vector of the first camera.
  // This is required so that the scene doesn't collapse during the optimization process.
  using SE3WithFixedNormManifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::SphereManifold<3>>;

  explicit Impl(double observation_noise)
      : problem_options_(createProblemOptions()), solver_options_(createSolverOptions()), problem_(problem_options_),
        cauchy_loss_(observation_noise) {}

  void addCamera(CameraCalib& calib, CameraType type = CameraType::CAM_N) {
    // Initially, don't optimize the principal points.
    // Ideally, don't optimize it at all if you don't have ideal correspondences
    problem_.AddParameterBlock(calib.intrinsics.principal_point.data(), 2);
    problem_.SetParameterBlockConstant(calib.intrinsics.principal_point.data());

    // Add a parameter block for camera extrinsic parameters
    problem_.AddParameterBlock(calib.world2cam.data(), SE3::num_parameters);

    switch (type) {
    case CameraType::CAM_0:
      // If the first camera is added
      // Double-check that it is aligned with origin
      CHECK(calib.world2cam.so3().unit_quaternion().isApprox(Eigen::Quaterniond::Identity()));
      CHECK(calib.world2cam.translation().isZero());

      // And set its pose to identity to eliminate any small errors
      calib.world2cam.so3().setQuaternion(Eigen::Quaterniond::Identity());
      calib.world2cam.translation().setZero();

      // Set this block constant, as we don't want to move the origin and left the scene drift away.
      problem_.SetParameterBlockConstant(calib.world2cam.data());
      break;

    case CameraType::CAM_1:
      // If the second camera is added
      // Double-check that its distance from the origin is unit
      CHECK_NEAR(calib.world2cam.translation().norm(), 1.0, 1e-5);

      // And set the distance exactly to unit if it is close to it
      calib.world2cam.translation().normalize();

      // Set the special parameterization for the first camera to keep it at unit distance from origin
      problem_.SetManifold(calib.world2cam.data(), &se3_with_fixed_norm_manifold_);
      break;

    case CameraType::CAM_N:
      // If any other camera is added
      // Just set the correct parameterization for SE3 type
      problem_.SetManifold(calib.world2cam.data(), &se3_manifold_);
      break;
    }

    cameras_.push_back(&calib);
  }

  void addObservation(CameraCalib& calib, Vec3& world_pt, const Vec2& image_pt) {
    // Create a cost function with Cauchy loss function which should effectively discriminate the influence of outliers
    auto cost_function = ReprojectionError::create(image_pt);
    problem_.AddResidualBlock(cost_function,
                              &cauchy_loss_,
                              calib.world2cam.data(),
                              calib.intrinsics.principal_point.data(),
                              &calib.intrinsics.focal_length,
                              world_pt.data());

    if (VLOG_IS_ON(2)) {
      Vec2 residual;
      std::vector<const double*> parameters = {calib.world2cam.data(),
                                               calib.intrinsics.principal_point.data(),
                                               &calib.intrinsics.focal_length,
                                               world_pt.data()};
      cost_function->Evaluate(parameters.data(), residual.data(), nullptr);
      VLOG(2) << "addObservation ( 3D: " << world_pt.transpose() << " / 2D: " << image_pt.transpose() << " )";
      VLOG(2) << "init residual: " << residual.transpose();
    }
  }

  void optimize() {
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, &problem_, &summary);

    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BA Problem solver report:\n" << summary.FullReport();
    } else {
      VLOG(1) << "BA Problem solver report:\n" << summary.BriefReport();
    }
  }

  void setPrincipalPointVariable() {
    for (CameraCalib* calib : cameras_) {
      problem_.SetParameterBlockVariable(calib->intrinsics.principal_point.data());
    }
  }

  void setPrincipalPointConstant() {
    for (CameraCalib* calib : cameras_) {
      problem_.SetParameterBlockConstant(calib->intrinsics.principal_point.data());
    }
  }

  [[nodiscard]] static ceres::Problem::Options createProblemOptions() {
    ceres::Problem::Options options;
    options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    return options;
  }

  [[nodiscard]] static ceres::Solver::Options createSolverOptions() {
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.logging_type = VLOG_IS_ON(2) ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;

    return options;
  }

  const ceres::Problem::Options problem_options_;
  const ceres::Solver::Options solver_options_;
  ceres::Problem problem_;

  SE3Manifold se3_manifold_;
  SE3WithFixedNormManifold se3_with_fixed_norm_manifold_;
  ceres::CauchyLoss cauchy_loss_;

  std::vector<CameraCalib*> cameras_;
};

BAProblem::BAProblem(double observation_noise) : impl_(new Impl(observation_noise)) {}

BAProblem::~BAProblem() = default;

void BAProblem::addCamera(CameraCalib& calib, CameraType type) {
  impl_->addCamera(calib, type);
}

void BAProblem::addObservation(CameraCalib& calib, Vec3& world_pt, const Vec2& image_pt) {
  impl_->addObservation(calib, world_pt, image_pt);
}

void BAProblem::optimize() {
  impl_->optimize();
}

void BAProblem::setPrincipalPointVariable() {
  impl_->setPrincipalPointVariable();
}

void BAProblem::setPrincipalPointConstant() {
  impl_->setPrincipalPointConstant();
}

} // namespace calib3d

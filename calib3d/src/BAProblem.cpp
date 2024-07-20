#include <calib3d/BAProblem.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace calib3d {

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
  using SE3Manifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
  using SE3WithFixedNormManifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::SphereManifold<3>>;

  explicit Impl(double observation_noise)
      : problem_options_(createProblemOptions()), solver_options_(createSolverOptions()), problem_(problem_options_),
        cauchy_loss_(observation_noise) {}

  void addCamera(CameraCalib& calib, CameraType type = CameraType::CAM_N) {
    problem_.AddParameterBlock(calib.world2cam.data(), SE3::num_parameters);

    switch (type) {
    case CameraType::CAM_0:
      CHECK(calib.world2cam.so3().unit_quaternion().isApprox(Eigen::Quaterniond::Identity()));
      CHECK(calib.world2cam.translation().isZero());

      calib.world2cam.so3().setQuaternion(Eigen::Quaterniond::Identity());
      calib.world2cam.translation().setZero();

      problem_.SetParameterBlockConstant(calib.world2cam.data());
      break;

    case CameraType::CAM_1:
      CHECK_NEAR(calib.world2cam.translation().norm(), 1.0, 1e-5);

      calib.world2cam.translation().normalize();

      problem_.SetManifold(calib.world2cam.data(), &se3_with_fixed_norm_manifold_);
      break;

    case CameraType::CAM_N:
      problem_.SetManifold(calib.world2cam.data(), &se3_manifold_);
      break;
    }
  }

  void addObservation(CameraCalib& calib, Vec3& world_pt, const Vec2& image_pt) {
    auto cost_function = ReprojectionError::create(image_pt);
    problem_.AddResidualBlock(cost_function,
                              &cauchy_loss_,
                              calib.world2cam.data(),
                              calib.intrinsics.principal_point.data(),
                              &calib.intrinsics.focal_length,
                              world_pt.data());

    {
      Vec2 residual;
      std::vector<const double*> parameters = {calib.world2cam.data(),
                                               calib.intrinsics.principal_point.data(),
                                               &calib.intrinsics.focal_length,
                                               world_pt.data()};
      cost_function->Evaluate(parameters.data(), residual.data(), nullptr);
      LOG(INFO) << "addObservation ( 3D: " << world_pt.transpose() << " / 2D: " << image_pt.transpose() << " )";
      LOG(INFO) << "init residual: " << residual.transpose();
    }
  }

  void optimize() {
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, &problem_, &summary);

    LOG(INFO) << "Solver report:\n" << summary.FullReport();
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
    options.linear_solver_type = ceres::DENSE_QR;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;

    return options;
  }

  const ceres::Problem::Options problem_options_;
  const ceres::Solver::Options solver_options_;
  ceres::Problem problem_;

  SE3Manifold se3_manifold_;
  SE3WithFixedNormManifold se3_with_fixed_norm_manifold_;
  ceres::CauchyLoss cauchy_loss_;
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

} // namespace calib3d

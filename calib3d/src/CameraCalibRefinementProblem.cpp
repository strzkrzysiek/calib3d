#include <calib3d/CameraCalibRefinementProblem.h>

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace calib3d {

struct ReprojectionError {
  ReprojectionError(const Mat3X& world_pts, const Mat2X& image_pts, int idx)
      : world_pts(world_pts), image_pts(image_pts), idx(idx) {}

  template <class T>
  bool operator()(const T* const raw_world2cam, const T* const raw_focal_length, T* raw_residual) const {
    Eigen::Map<const SE3T<T>> world2cam(raw_world2cam);
    const T& focal_length = *raw_focal_length;
    Eigen::Map<Vec2T<T>> residual(raw_residual);

    Vec2T<T> reprojected_pt = (world2cam * world_pts.col(idx).cast<T>()).hnormalized() * focal_length;
    residual = reprojected_pt - image_pts.col(idx).cast<T>();

    return true;
  }

  static ceres::CostFunction* create(const Mat3X& world_pts, const Mat2X& image_pts, int idx) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, SE3::num_parameters, 1>(
        new ReprojectionError(world_pts, image_pts, idx));
  }

  const Mat3X& world_pts;
  const Mat2X& image_pts;
  int idx;
};

struct CameraCalibRefinementProblem::Impl {
  using SE3Manifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;

  Impl(CameraCalib& calib, const Mat3x4& initial_P, double outlier_thr)
      : calib_(calib), initial_P_(initial_P), outlier_thr_(outlier_thr), problem_options_(createProblemOptions()),
        solver_options_(createSolverOptions()), problem_(problem_options_) {
    problem_.AddParameterBlock(calib_.world2cam.data(), SE3::num_parameters, &se3_manifold_);
  }

  void addCorrespondences(const Mat3X& world_pts, const Mat2X& image_pts) {
    CHECK_EQ(world_pts.cols(), image_pts.cols());

    auto reprojected_pts = (initial_P_ * world_pts.colwise().homogeneous()).colwise().hnormalized();
    VecX distance = (reprojected_pts - image_pts).colwise().squaredNorm();
    for (int i = 0; i < world_pts.cols(); i++) {
      if (distance(i) > outlier_thr_) {
        continue;
      }

      auto cost_function = ReprojectionError::create(world_pts, image_pts, i);
      problem_.AddResidualBlock(cost_function, nullptr, calib_.world2cam.data(), &calib_.intrinsics.focal_length);
    }
  }

  void optimize() {
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, &problem_, &summary);

    LOG(INFO) << "Optimized calib";
    LOG(INFO) << "F: " << calib_.intrinsics.focal_length;
    LOG(INFO) << "tvec: " << calib_.world2cam.translation().transpose();
    LOG(INFO) << "Rmat:\n" << calib_.world2cam.so3().matrix();

    LOG(INFO) << "Solver report:\n" << summary.BriefReport();
  }

  [[nodiscard]] static ceres::Problem::Options createProblemOptions() {
    ceres::Problem::Options options;
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

  CameraCalib& calib_;
  const Mat3x4& initial_P_;
  const double outlier_thr_;

  const ceres::Problem::Options problem_options_;
  const ceres::Solver::Options solver_options_;
  ceres::Problem problem_;
  SE3Manifold se3_manifold_;
};

CameraCalibRefinementProblem::CameraCalibRefinementProblem(CameraCalib& calib,
                                                           const Mat3x4& initial_P,
                                                           double outlier_thr)
    : impl_(new Impl(calib, initial_P, outlier_thr)) {}

CameraCalibRefinementProblem::~CameraCalibRefinementProblem() = default;

void CameraCalibRefinementProblem::addCorrespondences(const calib3d::Mat3X& world_pts,
                                                      const calib3d::Mat2X& image_pts) {
  impl_->addCorrespondences(world_pts, image_pts);
}

void CameraCalibRefinementProblem::optimize() {
  impl_->optimize();
}

} // namespace calib3d

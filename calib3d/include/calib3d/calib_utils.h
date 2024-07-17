#pragma once

#include <Eigen/Core>
#include <random>

namespace calib3d {

template <class Derived>
inline Eigen::Matrix3d skewSymmetric(const Eigen::DenseBase<Derived>& v);

template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> sampsonDistanceFromFundamentalMatrix(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> symmetricEpipolarDistance(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

template <class Derived>
void normalizePoints(
    const Eigen::DenseBase<Derived>& points,
    Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::ColsAtCompileTime>& X_normed,
    Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::RowsAtCompileTime + 1>& T);

template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrix(const Eigen::DenseBase<DerivedSrc>& src_points,
                                      const Eigen::DenseBase<DerivedDst>& dst_points);

template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrixRansac(const Eigen::DenseBase<DerivedSrc>& src_points,
                                            const Eigen::DenseBase<DerivedDst>& dst_points,
                                            double ransac_thr,
                                            double confidence,
                                            size_t max_iters,
                                            size_t seed = std::random_device()());

} // namespace calib3d

#include <calib3d/calib_utils.hpp>

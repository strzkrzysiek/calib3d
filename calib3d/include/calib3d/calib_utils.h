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

template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrix(const Eigen::DenseBase<Derived3D>& world_points,
                                                 const Eigen::DenseBase<Derived2D>& image_points);

template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrixRansac(
    const Eigen::DenseBase<Derived3D>& world_points,
    const Eigen::DenseBase<Derived2D>& image_points,
    double ransac_thr,
    double confidence,
    size_t max_iters,
    size_t seed = std::random_device()());

template <class Derived1, class Derived2>
Eigen::Matrix<double, 3, Derived1::ColsAtCompileTime> triangulatePoints(
    const Eigen::DenseBase<Derived1>& image_points1,
    const Eigen::DenseBase<Derived2>& image_points2,
    const Eigen::Matrix<double, 3, 4>& P1,
    const Eigen::Matrix<double, 3, 4>& P2);

} // namespace calib3d

#include <calib3d/calib_utils.hpp>

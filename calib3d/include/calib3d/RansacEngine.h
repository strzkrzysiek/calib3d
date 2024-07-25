// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

namespace calib3d {

// RANSAC engine for robust model fitting
// See: Section 4.7.1 RANSAC in Hartley and Zisserman (2003)
// It implements an adaptive method for estimation of iteration number and outlier ratio
//
// The RansacEngine to be used must be instantiated with a 'Spec' type that implements the following:
// 1. ModelMatrix Type:
//    Spec must define a type alias `ModelMatrix` that represents the type of the model matrix to be fitted.
//    using ModelMatrix = Mat3; // Example type alias
// 2. MinSampleSize Constant:
//    Spec must define a constant `MinSampleSize` that indicates the minimum number of samples required to fit the
//    model. static constexpr size_t MinSampleSize = 2;
// 3. Fit Model Function:
//    Spec must provide a static member function `fitModel` that fits the model given the input points.
//    The function should have the following signature:
//    template <class Derived1, class Derived2>
//    static ModelMatrix fitModel(const Eigen::DenseBase<Derived1>& pts1, const Eigen::DenseBase<Derived2>& pts2);
// 4. Distance Function:
//    Spec must provide a static member function `distance` that computes the squared(!) distance (error) between
//    the observed points and the model.
//    The function should have the following signature:
//    template <class Derived1, class Derived2>
//    static Eigen::Vector<double, Derived1::ColsAtCompileTime> distance(const Eigen::DenseBase<Derived1>& pts1,
//                                                                       const Eigen::DenseBase<Derived2>& pts2,
//                                                                       const ModelMatrix& model);
// 5. Dim1 and Dim2 Constants:
//    Spec must define constants `Dim1` and `Dim2` that describe the dimensionality of the input points.
//    static constexpr size_t Dim1 = 3;
//    static constexpr size_t Dim2 = 2;
template <class Spec>
class RansacEngine {
public:
  // Fits a model using RANSAC given two sets of points
  // It is optional to provide a seed value for deterministic results
  template <class Derived1, class Derived2>
  static typename Spec::ModelMatrix fit(const Eigen::DenseBase<Derived1>& points1,
                                        const Eigen::DenseBase<Derived2>& points2,
                                        double ransac_thr,
                                        double confidence,
                                        size_t max_iters,
                                        size_t seed = std::random_device()());

private:
  // Selects a random sample (of size Spec::MinSampleSize) of points
  template <class Derived1, class Derived2, class RNG>
  static auto getRandomSample(const Eigen::DenseBase<Derived1>& points1,
                              const Eigen::DenseBase<Derived2>& points2,
                              RNG& rnd);

  // Converts a binary mask to a vector of indices
  static std::vector<size_t> fromBinMask(const Eigen::Array<bool, Eigen::Dynamic, 1>& mask);
};

} // namespace calib3d

#include <calib3d/RansacEngine.hpp>

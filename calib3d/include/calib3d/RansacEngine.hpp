// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <array>
#include <glog/logging.h>
#include <ranges>
#include <set>

namespace calib3d {

template <class Spec>
template <class Derived1, class Derived2>
typename Spec::ModelMatrix RansacEngine<Spec>::fit(const Eigen::DenseBase<Derived1>& points1,
                                                   const Eigen::DenseBase<Derived2>& points2,
                                                   double ransac_thr,
                                                   double confidence,
                                                   size_t max_iters,
                                                   size_t seed) {
  static_assert(Derived1::RowsAtCompileTime == Spec::Dim1);
  static_assert(Derived2::RowsAtCompileTime == Spec::Dim2);

  CHECK_EQ(points1.cols(), points2.cols());

  const size_t n_points = points1.cols();
  CHECK_GE(n_points, Spec::MinSampleSize);

  std::mt19937 rnd_gen(seed);

  // Square the threshold as the distance function should return a squared metric
  const double squared_thr = ransac_thr * ransac_thr;

  // Keep track of the best consensus set
  std::vector<size_t> consensus_set_ids;
  double consensus_set_variance = 0.;

  for (size_t i = 0; i < max_iters; i++) {
    // Draw a random subset (of size Spec::MinSampleSize) from the set of provided points
    auto [sample1, sample2] = getRandomSample(points1, points2, rnd_gen);
    // Fit the model to the sample subset
    auto model_candidate = Spec::template fitModel(sample1, sample2);

    // Calculate the distance metric for the calculated model canidate for all of the points
    auto squared_distances = Spec::template distance(points1, points2, model_candidate);
    // Identify the inlier set
    auto inlier_ids = fromBinMask((squared_distances.array() < squared_thr).eval());
    const size_t n_inliers = inlier_ids.size();

    // If the inlier set size is at least as big as the current consensus set
    if (n_inliers >= consensus_set_ids.size() && n_inliers > 0) {
      double inlier_variance = squared_distances(inlier_ids).sum() / n_inliers;

      // If it is bigger or the same is the same but the new variance is smaller
      if (n_inliers > consensus_set_ids.size() || inlier_variance < consensus_set_variance) {
        // New consensus set is found
        consensus_set_ids = inlier_ids;
        consensus_set_variance = inlier_variance;

        VLOG(3) << "Iteration: " << i;
        VLOG(3) << "New consensus set: " << n_inliers << " (var: " << inlier_variance << ")";

        // Implement an daptive algorithm for determining the number of RANSAC samples
        // See: Algorithm 4.5 in Hartley and Zisserman (2003)
        double inlier_ratio = static_cast<double>(n_inliers) / n_points;
        VLOG(3) << "Inlier ratio: " << inlier_ratio;

        // Equation (4.18) in Hartley and Zisserman (2003)
        auto new_max_iters = static_cast<size_t>(
            std::ceil(std::log(1. - confidence) / std::log(1. - std::pow(inlier_ratio, Spec::MinSampleSize))));
        if (new_max_iters < max_iters) {
          max_iters = new_max_iters;
          VLOG(3) << "New max iters: " << max_iters;
        }
      }
    }
  }

  // If the provided set contains a significant number of outliers and/or
  // is so noisy so that most of the samples are considered as outliers
  // then it is possible that the largest consensus set contains fewer elements than
  // the minimal sample size.
  // It is clear then that the estimation of the inlier set failed.
  // Fall back to using all the input points as inliers.
  if (consensus_set_ids.size() < Spec::MinSampleSize) {
    LOG(WARNING) << "Consensus set (" << consensus_set_ids.size()
                 << ") is smaller than the minimal required sample size";
    LOG(WARNING) << "Consider increasing the max_iters argument.";
    LOG(WARNING) << "Fitting with all provided data.";

    return Spec::template fitModel(points1, points2);
  }

  VLOG(2) << "Final estimation using " << consensus_set_ids.size() << " inliers";

  return Spec::template fitModel(points1(Eigen::all, consensus_set_ids), points2(Eigen::all, consensus_set_ids));
}

template <class Spec>
template <class Derived1, class Derived2, class RNG>
auto RansacEngine<Spec>::getRandomSample(const Eigen::DenseBase<Derived1>& points1,
                                         const Eigen::DenseBase<Derived2>& points2,
                                         RNG& rnd) {
  const size_t n_points = points1.cols();
  std::uniform_int_distribution<size_t> idx_dist(0, n_points - 1);

  std::set<size_t> index_set;
  while (index_set.size() < Spec::MinSampleSize) {
    index_set.insert(idx_dist(rnd));
  }
  std::array<size_t, Spec::MinSampleSize> index_array;
  std::copy(index_set.begin(), index_set.end(), index_array.begin());

  return std::make_pair(points1(Eigen::all, index_array), points2(Eigen::all, index_array));
}

template <class Spec>
std::vector<size_t> RansacEngine<Spec>::fromBinMask(const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
  auto indices = std::views::iota(0, mask.size()) | std::views::filter([&mask](const auto& i) { return mask(i); });
  std::vector<size_t> result;
  std::ranges::copy(indices, std::back_inserter(result));
  return result;
}

} // namespace calib3d

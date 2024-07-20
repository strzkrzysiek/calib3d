#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

namespace calib3d {

template <class Spec>
class RansacEngine {
public:
  template <class Derived1, class Derived2>
  static typename Spec::ModelMatrix fit(const Eigen::DenseBase<Derived1>& points1,
                                        const Eigen::DenseBase<Derived2>& points2,
                                        double ransac_thr,
                                        double confidence,
                                        size_t max_iters,
                                        size_t seed = std::random_device()());

private:
  template <class Derived1, class Derived2, class RNG>
  static auto getRandomSample(const Eigen::DenseBase<Derived1>& points1,
                              const Eigen::DenseBase<Derived2>& points2,
                              RNG& rnd);

  static std::vector<size_t> fromBinMask(const Eigen::Array<bool, Eigen::Dynamic, 1>& mask);
};

} // namespace calib3d

#include <calib3d/RansacEngine.hpp>

#pragma once

#include <Eigen/Core>
#include <map>

#include <calib3d/types.h>

namespace calib3d {

struct CamObservations {};

class Calib3D {
public:
  Calib3D();

  void initializeReconstruction(const std::map<size_t, CamObservations>& init_observations);

  void addCamera(size_t cam_id, const CamObservations& observations);

private:
};

} // namespace calib3d

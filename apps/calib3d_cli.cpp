// Copyright 2024 Krzysztof Wrobel

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <set>

#include <calib3d/Dataset.h>
#include <calib3d/NViewReconstruction.h>

namespace po = boost::program_options;

int runApp(const std::string& inpath,
           const std::string& outpath,
           double observation_noise,
           bool optimize_principal_point) {
  calib3d::Dataset dataset;
  if (!dataset.loadFromJson(inpath)) {
    LOG(ERROR) << "Failed to load the input file";
    return -1;
  }

  if (dataset.cameras.size() < 3) {
    LOG(ERROR) << "At least 3 cameras are required. Got " << dataset.cameras.size();
    return -1;
  }

  calib3d::NViewReconstruction reconstruction(observation_noise);

  auto cam_it = dataset.cameras.begin();
  calib3d::ThreeOf<calib3d::CamId> init_cam_ids = {(cam_it++)->first, (cam_it++)->first, (cam_it++)->first};

  reconstruction.initReconstruction(init_cam_ids[0],
                                    dataset.cameras.at(init_cam_ids[0]).calib.size,
                                    dataset.cameras.at(init_cam_ids[0]).observations,
                                    init_cam_ids[1],
                                    dataset.cameras.at(init_cam_ids[1]).calib.size,
                                    dataset.cameras.at(init_cam_ids[1]).observations,
                                    init_cam_ids[2],
                                    dataset.cameras.at(init_cam_ids[2]).calib.size,
                                    dataset.cameras.at(init_cam_ids[2]).observations);

  for (; cam_it != dataset.cameras.end(); ++cam_it) {
    reconstruction.addNewCamera(cam_it->first, cam_it->second.calib.size, cam_it->second.observations);
  }

  if (optimize_principal_point) {
    reconstruction.optimizeWithPrincipalPoint();
  }

  for (auto& [cam_id, cam_bundle] : dataset.cameras) {
    cam_bundle.calib = reconstruction.getCameras().at(cam_id);
  }

  for (auto& [pt_id, world_pt] : dataset.world_points) {
    world_pt = reconstruction.getPoints().at(pt_id).world_pt.value();
  }

  if (!dataset.dumpToJson(outpath)) {
    LOG(ERROR) << "Failed to dump data to the output file";

    return -1;
  }

  return 0;
}

int main(int argc, char* argv[]) {
  try {
    FLAGS_logtostderr = true;
    FLAGS_v = 1;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "Calib3D";

    // PROGRAM ARGUMENTS ///////////////////////////////////////////////////////////

    std::string inpath;
    std::string outpath;
    double observation_noise;

    po::options_description po_desc("Calib3D - usage");
    // clang-format off
    po_desc.add_options()
        ("help", "Produce help message")
        ("input,i", po::value<std::string>(&inpath)->required(), "Input file path")
        ("output,o", po::value<std::string>(&outpath)->required(), "Output file path")
        ("noise,n", po::value<double>(&observation_noise)->default_value(2.0), "Standard deviation of observation error")
        ("optprincipal,p", "Optimize also the principal point");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, po_desc), vm);

    if (argc <= 1 || vm.count("help")) {
      LOG(INFO) << "\n" << po_desc;
      return 0;
    }

    try {
      po::notify(vm);
    } catch (const po::error& e) {
      LOG(ERROR) << "Program options error: " << e.what() << '\n' << po_desc;
      return -1;
    }

    return runApp(inpath, outpath, observation_noise, vm.count("optprincipal"));

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error: " << e.what();
    return -1;
  } catch (...) {
    LOG(ERROR) << "Unknown error";
    return -1;
  }

  return 0;
}
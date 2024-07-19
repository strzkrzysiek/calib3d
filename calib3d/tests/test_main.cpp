#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 0;

  int result = RUN_ALL_TESTS();

  google::ShutdownGoogleLogging();

  return result;
}

#include "gtest/gtest.h"
#include <string>
#include <boost/filesystem.hpp>

namespace j {
class ImageSegmenter;

class ImageSegmenterTests : public testing::Test {
protected:
  std::string test_file_;
  unsigned char threshold_ = 180;
  unsigned int min_segment_area_ = 50;
  unsigned char crop_margin_ = 4;
  boost::filesystem::path tempfile_;

  virtual void SetUp() override;
};
}
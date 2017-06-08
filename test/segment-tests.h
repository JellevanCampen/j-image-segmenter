#include "gtest/gtest.h"
#include <boost/filesystem.hpp>
#include "../source/segment.h"

using namespace j;

class SegmentTests : public testing::Test {
protected:
  Segment segment_;
  std::vector<cv::Point> contour_exp_;
  cv::Rect bounding_rectangle_exp_;
  Segment::Tag tag_exp_;
  unsigned int area_exp_;
  boost::filesystem::path tempfile_;

  virtual void SetUp() override;
};
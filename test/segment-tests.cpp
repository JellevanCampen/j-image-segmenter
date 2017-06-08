#include "segment-tests.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

void SegmentTests::SetUp()
{
  // Create a sample segment
  contour_exp_.push_back(cv::Point(1, 2));
  contour_exp_.push_back(cv::Point(4, 2));
  contour_exp_.push_back(cv::Point(4, 4));
  contour_exp_.push_back(cv::Point(1, 4));
  bounding_rectangle_exp_ = cv::Rect(1, 2, 4, 3);
  tag_exp_ = Segment::Tag::CORRECT;
  area_exp_ = 6;
  segment_ = Segment(contour_exp_, tag_exp_);

  boost::filesystem::path tempfile_abs = boost::filesystem::unique_path("%%%%_%%%%_%%%%_%%%%.progress");
  tempfile_ = boost::filesystem::temp_directory_path() / tempfile_abs;
}

TEST_F(SegmentTests, DefaultConstructor) {
  Segment s;
  EXPECT_EQ(s.contour_.size(), 0);
  EXPECT_EQ(s.bounding_rectangle_, cv::Rect());
  EXPECT_EQ(s.tag_, Segment::Tag::UNDEFINED);
  EXPECT_EQ(s.area_, 0);
}

TEST_F(SegmentTests, ContourConstructor) {
  EXPECT_EQ(segment_.contour_, contour_exp_);
  EXPECT_EQ(segment_.bounding_rectangle_, bounding_rectangle_exp_);
  EXPECT_EQ(segment_.tag_, tag_exp_);
  EXPECT_EQ(segment_.area_, area_exp_);
}

TEST_F(SegmentTests, Serialize) {
  boost::filesystem::ofstream ofs(tempfile_);
  boost::archive::binary_oarchive oa(ofs);
  oa << segment_;
  ofs.close();

  Segment segment_read;
  boost::filesystem::ifstream ifs(tempfile_);
  boost::archive::binary_iarchive ia(ifs);
  ia >> segment_read;
  ifs.close();

  EXPECT_EQ(segment_read.contour_, segment_.contour_);
  EXPECT_EQ(segment_read.bounding_rectangle_, segment_.bounding_rectangle_);
  EXPECT_EQ(segment_read.tag_, segment_.tag_);
  EXPECT_EQ(segment_read.area_, segment_.area_);
}
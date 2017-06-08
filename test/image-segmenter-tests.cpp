#include "image-segmenter-tests.h"
#include "../source/image_segmenter.h"

namespace j {
void ImageSegmenterTests::SetUp() {
  test_file_ = "../../input/sample.jpg";
  boost::filesystem::path tempfile_abs = boost::filesystem::unique_path("%%%%_%%%%_%%%%_%%%%.progress");
  tempfile_ = boost::filesystem::temp_directory_path() / tempfile_abs;
}

TEST_F(ImageSegmenterTests, PerformSegmentation) {
  cv::Mat image = cv::imread(test_file_, cv::IMREAD_GRAYSCALE);
  cv::threshold(image, image, threshold_, 255, cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat image_inverted = 255 - image;
  cv::findContours(image_inverted, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  auto it = contours.begin();
  while (it != contours.end()) {
    unsigned int area = (unsigned int)cv::contourArea(*it);
    if (area < min_segment_area_) { it = contours.erase(it); }
    else { it++; }
  }
  std::vector<Segment> segments = ImageSegmenter::PerformSegmentation(image, min_segment_area_);
  EXPECT_EQ(segments.size(), contours.size());
}
}
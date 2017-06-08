#pragma once
#ifndef IMAGESEGMENTER_SEGMENT_H
#define IMAGESEGMENTER_SEGMENT_H

#include "opencv2/opencv.hpp"
#include <vector>
#include "serialize_template_specializations.h"

namespace j {
class Segment {
public:
  enum class Tag {
    UNDEFINED,
    NOISE,
    PARTIAL,
    MERGED,
    CORRECT
  };
  std::vector<cv::Point> contour_;
  cv::Rect bounding_rectangle_;
  Tag tag_{ Tag::UNDEFINED };
  uint area_{ 0 };

  Segment() { }

  Segment(std::vector<cv::Point> contour, Tag tag = Tag::UNDEFINED) : contour_(contour), tag_(tag) {
    bounding_rectangle_ = cv::boundingRect(contour_);
    area_ = (unsigned int)cv::contourArea(contour_);
  }

  // Serialization via Boost
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & contour_;
    ar & bounding_rectangle_;
    ar & tag_;
    ar & area_;
  }
};
}

#endif
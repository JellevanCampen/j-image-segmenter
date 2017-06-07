#pragma once
#ifndef IMAGESEGMENTER_UTILITY_H
#define IMAGESEGMENTER_UTILITY_H

#include "opencv2/opencv.hpp"
#include "segment.h"

namespace j {

// Get the mininum of two values
template<typename T>
static T min(const T& a, const T& b) {
  return (a < b) ? a : b;
}

// Get the maximum of two values
template<typename T>
static T max(const T& a, const T& b) {
  return (a > b) ? a : b;
}

// Clip a variable value between a lower and upper limit
template<typename T>
static T clip(const T& a, const T& lower, const T& upper) {
  return min(upper, max(lower, a));
}

// Get the bounding rectangle of a set of segments
static cv::Rect GetBoundingRect(const std::vector<Segment>& segments) {
  if (segments.size() == 0) { return cv::Rect(); }
  int x1_min = segments[0].bounding_rectangle_.x;
  int x2_max = segments[0].bounding_rectangle_.x + segments[0].bounding_rectangle_.width;
  int y1_min = segments[0].bounding_rectangle_.y;
  int y2_max = segments[0].bounding_rectangle_.y + segments[0].bounding_rectangle_.height;

  for (int i = 1; i < segments.size(); i++) {
    x1_min = min(x1_min, segments[i].bounding_rectangle_.x);
    x2_max = max(x2_max, segments[i].bounding_rectangle_.x + segments[i].bounding_rectangle_.width);
    y1_min = min(y1_min, segments[i].bounding_rectangle_.y);
    y2_max = max(y2_max, segments[i].bounding_rectangle_.y + segments[i].bounding_rectangle_.height);
  }
  return cv::Rect(x1_min, y1_min, x2_max - x1_min, y2_max - y1_min);
}
}

#endif
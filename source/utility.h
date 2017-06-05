#include "opencv2/opencv.hpp"

using namespace cv;

// Clip a variable value between a lower and upper limit
template<typename T>
static float clip(T i, T lower, T upper) {
  return (i < upper ? i : upper) > lower ? (i < upper ? i : upper) : lower;
}

// Get the bounding rectangle of a set of rectangles
static Rect GetBoundingRect(const std::vector<Rect>& rectangles) {
  uint x1_min = rectangles.at(0).x;
  uint x2_max = rectangles.at(0).x + rectangles.at(0).width;
  uint y1_min = rectangles.at(0).y;
  uint y2_max = rectangles.at(0).y + rectangles.at(0).height;
  for (int i = 1; i < rectangles.size(); i++) {
    x1_min = (x1_min < rectangles.at(i).x) ? x1_min : rectangles.at(i).x;
    x2_max = (x2_max > rectangles.at(i).x + rectangles.at(i).width) ? x2_max : rectangles.at(i).x + rectangles.at(i).width;
    y1_min = (y1_min < rectangles.at(i).y) ? y1_min : rectangles.at(i).y;
    y2_max = (y2_max > rectangles.at(i).y + rectangles.at(i).height) ? y2_max : rectangles.at(i).y + rectangles.at(i).height;
  }
  return Rect(x1_min, y1_min, x2_max - x1_min, y2_max - y1_min);
}
#pragma once
#ifndef IMAGESEGMENTER_SERIALIZETEMPLATESPECIALIZATIONS_H
#define IMAGESEGMENTER_SERIALIZETEMPLATESPECIALIZATIONS_H

#include "opencv2/opencv.hpp"

namespace boost {
namespace serialization {
  template<class Archive>
  void serialize(Archive & ar, cv::Point & p, const unsigned int version) {
    ar & p.x;
    ar & p.y;
  }
  template<class Archive>
  void serialize(Archive & ar, cv::Rect & r, const unsigned int version) {
    ar & r.x; 
    ar & r.y;
    ar & r.width;
    ar & r.height;
  }
}
}

#endif
#pragma once
#ifndef IMAGESEGMENTER_IMAGESEGMENTER_H
#define IMAGESEGMENTER_IMAGESEGMENTER_H

#include "opencv2/opencv.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include "segment.h"

namespace j {
// Implements a procedure for segmenting images into individual segments. 
// Todo: example
class ImageSegmenter {
  // Steps in the image segmentation procedure
  enum class Step {
    THRESHOLDING,
    SEGMENT_DETECTION,
    SEGMENT_TAGGING,
    SEGMENT_MERGING,
    SEGMENT_EXPORTING
  };

private:
  // Data
  Step current_step_ = Step::THRESHOLDING;
  cv::Mat image_3c_;
  cv::Mat image_1c_;
  cv::Mat threshold_mask_image_1c_;
  std::vector<Segment> segments_todo_;
  std::vector<Segment> segments_correct_;
  std::vector<Segment> segments_merged_;
  std::vector<std::vector<Segment>> segments_partial_sets_;

  // Settings
  std::string image_file_;
  unsigned char threshold_;
  unsigned int min_segment_area_;
  unsigned char outline_thickness_;
  float surroundings_size_;
  std::string output_directory_;
  unsigned int crop_margin_;

  // Serialization via Boost for saving and loading progress
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version);

public:
  // Construct a new image segmenter
  ImageSegmenter(
    const std::string& image_file,
    unsigned char threshold = 192,
    unsigned int min_segment_area = 20,
    unsigned char outline_thickness = 4,
    float surroundings_size = 10.f,
    std::string output_directory = "output",
    unsigned char crop_margin = 1)
    :image_file_(image_file)
    , threshold_(threshold)
    , min_segment_area_(min_segment_area)
    , outline_thickness_(outline_thickness)
    , surroundings_size_(surroundings_size)
    , output_directory_(output_directory)
    , crop_margin_(crop_margin) {
    image_3c_ = cv::imread(image_file_, cv::IMREAD_COLOR);
    cv::cvtColor(image_3c_, image_1c_, CV_BGR2GRAY);
  }

  // Loads and plays back progress from file
  void LoadProgress(const std::string& progress_file);

  // Different steps in the procedure
  void RunThresholdingStep(bool interactive = false);
  void RunSegmentDetectionStep(bool interactive = false);
  void RunSegmentTaggingStep();
  void RunPartialSegmentMergingStep();
  void RunSegmentExportingStep();

  // Processing subroutines
  std::vector<Segment> PerformSegmentation(const cv::Mat& threshold_mask_image_1c, unsigned int min_segment_area) const;
  cv::Mat GenerateSegmentationPreview(const cv::Mat& image_background, const std::vector<Segment>& segments, int line_thickness) const;
  void GenerateSegmentPreviews(const cv::Mat& image, const Segment& segment, const cv::Scalar& color, float surroundings_size, cv::Mat* out_preview, cv::Mat* out_preview_contour) const;
  void GenerateSegmentPreviews(const cv::Mat& image, const std::vector<Segment>& segments, const std::vector<cv::Scalar>& colors, float surroundings_size, cv::Mat* out_preview, cv::Mat* out_preview_contour) const;
  void MoveSegmentsByTag(std::vector<Segment>& segments_src, Segment::Tag tag, std::vector<Segment>& segments_dst) const;
  void RemoveSegmentsByTag(std::vector<Segment>& segments_src, Segment::Tag tag) const;
  void SaveSegmentsToFile(const cv::Mat& image, const Segment& segment, const std::string& filename, int margin) const;
  void SaveSegmentsToFile(const cv::Mat& image, const std::vector<Segment>& segments, const std::string& filename, int margin) const;
  void SaveMultipleSegmentsToFiles(const cv::Mat& image, const std::vector<Segment>& segments, const std::string& directory, const std::string& prefix, int margin) const;
  void SaveMultipleSegmentsToFiles(const cv::Mat& image, const std::vector<std::vector<Segment>>& segments, const std::string& directory, const std::string& prefix, int margin) const;
};
}

#endif
#include "image_segmenter.h"
#include <sstream>
#include "utility.h"

namespace j {
  template<class Archive>
  void j::ImageSegmenter::serialize(Archive & ar, const unsigned int version) {
    ar & current_step_;
    ar & segments_todo_;
    ar & segments_correct_;
    ar & segments_merged_;
    ar & input_image_file_;
    ar & threshold_;
    ar & min_segment_area_;
    ar & outline_thickness_;
    ar & surroundings_size_;
    ar & output_directory_;
    ar & crop_margin_;
  }

  void ImageSegmenter::LoadProgress(const std::string & progress_file) {
    // Todo: load progress from a file
  }

  void ImageSegmenter::RunThresholdingStep(bool interactive) {
    current_step_ = Step::THRESHOLDING;
    if (interactive) {
      std::cout << "Step 1. Thresholding" << std::endl
                << "================" << std::endl
                << ">> Separating background and foreground segments." << std::endl;
      cv::threshold(image_1c_, threshold_mask_image_1c_, threshold_, 255, cv::THRESH_BINARY);
      cv::namedWindow("ImageSegmenter (Step 1. Thresholding)", cv::WINDOW_NORMAL);
      cv::imshow("ImageSegmenter (Step 1. Thresholding)", threshold_mask_image_1c_);
      std::cout << ">> Press [SPACE] to continue" << std::endl;
      while (cv::waitKey(0) != ' ');
      cv::destroyWindow("ImageSegmenter (Step 1. Thresholding)");
    } else {
      std::cout << "Performing thresholding ..." << std::endl;
      cv::threshold(image_1c_, threshold_mask_image_1c_, threshold_, 255, cv::THRESH_BINARY);
    }
  }

  void ImageSegmenter::RunSegmentDetectionStep(bool interactive) {
    current_step_ = Step::SEGMENT_DETECTION;
    if (interactive) {
      std::cout << "Step 2. Segment detection" << std::endl
                << "================" << std::endl
                << ">> Detecting all individual segments after thresholding." << std::endl;
      segments_todo_ = PerformSegmentation(threshold_mask_image_1c_, min_segment_area_);
      cv::namedWindow("ImageSegmenter (Step 2. Segment detection)", cv::WINDOW_NORMAL);
      cv::Mat segmentation_preview = GenerateSegmentationPreview(image_3c_, segments_todo_, outline_thickness_);
      cv::imshow("ImageSegmenter (Step 2. Segment detection)", segmentation_preview);
      std::cout << ">> Press [SPACE] to confirm" << std::endl;
      while (cv::waitKey(0) != ' ');
      cv::destroyWindow("ImageSegmenter (Step 2. Segment detection)");
    } else {
      std::cout << "Performing segment detection ..." << std::endl;
      segments_todo_ = PerformSegmentation(threshold_mask_image_1c_, min_segment_area_);
    }
  }

  void ImageSegmenter::RunSegmentTaggingStep() {
    current_step_ = Step::SEGMENT_TAGGING;
    std::cout << "Step 3. Segment tagging" << std::endl
              << "================" << std::endl
              << ">> Tagging segments, use the following keys:" << std::endl
              << "   [N] Noise segment (will be discarded)" << std::endl
              << "   [P] Partial segment (will be combinable with other partial segments)" << std::endl
              << "   [M] Merged segment (will be stored separately so it can be split)" << std::endl
              << "   [C] Correct segment (will be stored as is)" << std::endl
              << std::endl
              << "   [Z] Undo (move back in the tagging sequence)" << std::endl
              << "================" << std::endl;
    cv::namedWindow("ImageSegmenter (Step 3. Segment tagging)", cv::WINDOW_NORMAL);
      
    auto it = segments_todo_.begin();
    while (it != segments_todo_.end()) {
      std::cout << ">> Tagging segment [" << (it - segments_todo_.begin() + 1) << "/" << segments_todo_.size() << "]: ";
      cv::Mat preview, preview_contour;
      GenerateSegmentPreviews(image_3c_, *it, cv::Scalar(255, 0, 0), surroundings_size_, &preview, &preview_contour);

      bool show_contour = true;
      int last_key = -1;
      while (last_key != 'n' && last_key != 'p' && last_key != 'm' && last_key != 'c' && last_key != 'z') {
        show_contour = !show_contour;
        imshow("ImageSegmenter (Step 3. Segment tagging)", show_contour ? preview_contour : preview);
        last_key = cv::waitKeyEx(250);
      }
      switch (last_key) {
      case 'n':
        std::cout << "NOISE" << std::endl;
        it->tag_ = Segment::Tag::NOISE;
        it++;
        break;
      case 'p':
        std::cout << "PARTIAL" << std::endl;
        it->tag_ = Segment::Tag::PARTIAL;
        it++; 
        break;
      case 'm':
        std::cout << "MERGED" << std::endl;
        it->tag_ = Segment::Tag::MERGED;
        it++;
        break;
      case 'c':
        std::cout << "CORRECT" << std::endl;
        it->tag_ = Segment::Tag::CORRECT;
        it++;
        break;
      case 'z':
        if (it != segments_todo_.begin()) {
          std::cout << "... undoing previous tag" << std::endl;
          it--;
        }
        break;
      }
    }
    cv::destroyWindow("ImageSegmenter (Step 3. Segment tagging)");
    MoveSegmentsByTag(segments_todo_, Segment::Tag::CORRECT, segments_correct_);
    MoveSegmentsByTag(segments_todo_, Segment::Tag::MERGED, segments_merged_);
    RemoveSegmentsByTag(segments_todo_, Segment::Tag::NOISE);
  }

  void ImageSegmenter::RunPartialSegmentMergingStep() {
    current_step_ = Step::SEGMENT_MERGING;
    std::cout << "Step 4. Partial segment merging" << std::endl
              << "================" << std::endl
              << ">> Merging partial segments. The partial segments in [BLUE] are looking for"
              << "partial segments to merge with. The partial segment in [GREEN] proposes to"
              << "to merge. Use the following keys:" << std::endl
              << "   [A] Accept segment (green will be merged with blue)" << std::endl
              << "   [R] Reject segment (green will not be merged with blue)" << std::endl
              << "   [C] Complete merging (blue is complete and will be saved)" << std::endl
              << "================" << std::endl;
    cv::namedWindow("ImageSegmenter (Step 4. Partial segment merging)", cv::WINDOW_NORMAL);

    while (!segments_todo_.empty()) {
      auto it_todo = segments_todo_.begin();
      // Start a new partial set
      segments_partial_sets_.push_back(std::vector<Segment>());
      auto it_set = segments_partial_sets_.end();
      it_set--;
      it_set->push_back(*it_todo);
      it_todo = segments_todo_.erase(it_todo);

      while (true) { // Cycle through merge candidates
        if (segments_todo_.empty()) {
          std::cout << "   PARTIAL SET COMPLETED (no partial segments left)" << std::endl;
          break;
        }
        std::cout << "   Proposing partial segment [" << (it_todo - segments_todo_.begin() + 1) << "/" << segments_todo_.size() << "]: ";
        std::vector<Segment> preview_segments(*it_set);
        preview_segments.push_back(*it_todo);
        std::vector<cv::Scalar> preview_colours;
        for (int i = 0; i < it_set->size(); i++) { preview_colours.push_back(cv::Scalar(255, 0, 0)); }
        preview_colours.push_back(cv::Scalar(0, 255, 0));
        cv::Mat preview, preview_contours;
        GenerateSegmentPreviews(image_3c_, preview_segments, preview_colours, surroundings_size_, &preview, &preview_contours);

        bool show_contour = true;
        int last_key = -1;
        while (last_key != 'a' && last_key != 'r' && last_key != 'c') {
          show_contour = !show_contour;
          imshow("ImageSegmenter (Step 4. Partial segment merging)", show_contour ? preview_contours : preview);
          last_key = cv::waitKeyEx(250);
        }
        switch (last_key) {
        case 'a':
          std::cout << "   ACCEPTED" << std::endl;
          it_set->push_back(*it_todo);
          it_todo = segments_todo_.erase(it_todo);
          break;
        case 'r':
          std::cout << "   REJECTED" << std::endl;
          it_todo++;
          break;
        case 'c':
          std::cout << "   REJECTED" << std::endl << "   PARTIAL SET COMPLETED" << std::endl;
          goto complete_set;
        }
        if (it_todo == segments_todo_.end() && !segments_todo_.empty()) { it_todo = segments_todo_.begin(); }
      }
    complete_set:
      continue;
    }
  }

  void ImageSegmenter::RunSegmentExportingStep() {
    current_step_ = Step::SEGMENT_EXPORTING;
    std::cout << "Step 5. Segment exporting" << std::endl
              << "================" << std::endl
              << ">> Isolating segments and exporting to files." << std::endl
              << "================" << std::endl;
    std::cout << "   Exporting [Correct] segments" << std::endl;
    SaveMultipleSegmentsToFiles(image_3c_, segments_correct_, output_directory_ + "/correct", "c_", crop_margin_);
    std::cout << "   Exporting [Merged] segments" << std::endl;
    SaveMultipleSegmentsToFiles(image_3c_, segments_merged_, output_directory_ + "/merged", "m_", crop_margin_);
    std::cout << "   Exporting [Partial] segment sets" << std::endl;
    SaveMultipleSegmentsToFiles(image_3c_, segments_partial_sets_, output_directory_ + "/partial_sets", "p_", crop_margin_);
  }

  std::vector<Segment> ImageSegmenter::PerformSegmentation(const cv::Mat& threshold_mask_image_1c, unsigned int min_segment_area) const {
    cv::Mat image_thresholded_inverted = 255 - threshold_mask_image_1c;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image_thresholded_inverted, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create segments for all detected contours
    std::vector<Segment> segments;
    for (const auto& c : contours) {
      Segment s;
      s.area_ = cv::contourArea(c);
      if (s.area_ < min_segment_area) { continue; }
      s.contour_ = c;
      s.bounding_rectangle_ = cv::boundingRect(c);
      s.tag_ = Segment::Tag::UNDEFINED;
      segments.push_back(s);
    }

    // Sort contours and their metadata by area
    std::sort(segments.begin(), segments.end(), 
      [](const Segment& a, const Segment& b) -> bool { return a.area_ > b.area_; });
    return segments;
  }

  cv::Mat ImageSegmenter::GenerateSegmentationPreview(const cv::Mat & image_background, const std::vector<Segment>& segments, int line_thickness) const {
    cv::Mat result;
    image_background.copyTo(result);
    for (const auto& s : segments) {
      std::vector<std::vector<cv::Point>> contour;
      contour.push_back(s.contour_);
      cv::drawContours(result, contour, 0, cv::Scalar(rand() % 255, rand() % 255, rand() % 255), line_thickness);
    }
    return result;
  }

  void ImageSegmenter::GenerateSegmentPreviews(const cv::Mat & image, const Segment & segment, const cv::Scalar & color, float surroundings_size, cv::Mat * out_preview, cv::Mat * out_preview_contour) const {
    int size = max(segment.bounding_rectangle_.width, segment.bounding_rectangle_.height);
    int size_surroundings = size * surroundings_size;
    int x_center = segment.bounding_rectangle_.x + segment.bounding_rectangle_.width / 2;
    int y_center = segment.bounding_rectangle_.y + segment.bounding_rectangle_.height / 2;
    int x1 = clip(x_center - size_surroundings, 0, image.cols);
    int x2 = clip(x_center + size_surroundings, 0, image.cols);
    int y1 = clip(y_center - size_surroundings, 0, image.rows);
    int y2 = clip(y_center + size_surroundings, 0, image.rows);

    cv::Mat preview(y2 - y1, x2 - x1, CV_8UC3);
    image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(preview);
    *out_preview = preview.clone();

    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(segment.contour_);
    drawContours(preview, contours, 0, color, -1, cv::LINE_8, cv::noArray(), 1, cv::Point(-x1, -y1));
    *out_preview_contour = preview.clone();
  }

  void ImageSegmenter::GenerateSegmentPreviews(const cv::Mat & image, const std::vector<Segment>& segments, const std::vector<cv::Scalar>& colors, float surroundings_size, cv::Mat * out_preview, cv::Mat * out_preview_contour) const {
    cv::Rect combined_bounding_rectangle = GetBoundingRect(segments);
    int size = max(combined_bounding_rectangle.width, combined_bounding_rectangle.height);
    int size_surroundings = size * surroundings_size;
    int x_center = combined_bounding_rectangle.x + combined_bounding_rectangle.width / 2;
    int y_center = combined_bounding_rectangle.y + combined_bounding_rectangle.height / 2;
    uint x1 = clip(x_center - size_surroundings, 0, image.cols);
    uint x2 = clip(x_center + size_surroundings, 0, image.cols);
    uint y1 = clip(y_center - size_surroundings, 0, image.rows);
    uint y2 = clip(y_center + size_surroundings, 0, image.rows);

    cv::Mat preview(y2 - y1, x2 - x1, CV_8UC3);
    image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(preview);
    *out_preview = preview.clone();

    for (int i = 0; i < segments.size(); i++) {
      std::vector<std::vector<cv::Point>> contours;
      contours.push_back(segments[i].contour_);
      cv::drawContours(preview, contours, 0, colors[i], -1, cv::LINE_8, cv::noArray(), 1, cv::Point(-x1, -y1));
    }
    *out_preview_contour = preview.clone();
  }

  void ImageSegmenter::MoveSegmentsByTag(std::vector<Segment>& segments_src, Segment::Tag tag, std::vector<Segment>& segments_dst) const {
    auto it = segments_src.begin();
    while (it != segments_src.end()) {
      if (it->tag_ == tag) {
        segments_dst.push_back(*it);
        it = segments_src.erase(it);
      } else { it++; }
    }
  }

  void ImageSegmenter::RemoveSegmentsByTag(std::vector<Segment>& segments_src, Segment::Tag tag) const {
    auto it = segments_src.begin();
    while (it != segments_src.end()) {
      if (it->tag_ == tag) { it = segments_src.erase(it); 
      } else { it++; }
    }
  }

  void ImageSegmenter::SaveSegmentsToFile(const cv::Mat & image, const Segment & segment, const std::string & filename, int margin) const {
    uint x1 = clip(segment.bounding_rectangle_.x - margin, 0, image.cols);
    uint x2 = clip(segment.bounding_rectangle_.x + segment.bounding_rectangle_.width + margin, 0, image.cols);
    uint y1 = clip(segment.bounding_rectangle_.y - margin, 0, image.rows);
    uint y2 = clip(segment.bounding_rectangle_.y + segment.bounding_rectangle_.height + margin, 0, image.rows);

    cv::Mat output(y2 - y1, x2 - x1, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat mask = cv::Mat::zeros(y2 - y1, x2 - x1, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(segment.contour_);
    drawContours(mask, contours, 0, cv::Scalar(255), -1, cv::LINE_8, cv::noArray(), 1, cv::Point(-x1, -y1));
    image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(output, mask);
    cv::imwrite(filename + ".jpg", output);
  }

  void ImageSegmenter::SaveSegmentsToFile(const cv::Mat & image, const std::vector<Segment>& segments, const std::string & filename, int margin) const {
    cv::Rect combined_bounding_rectangle = GetBoundingRect(segments);
    uint x1 = clip(combined_bounding_rectangle.x - margin, 0, image.cols);
    uint x2 = clip(combined_bounding_rectangle.x + combined_bounding_rectangle.width + margin, 0, image.cols);
    uint y1 = clip(combined_bounding_rectangle.y - margin, 0, image.rows);
    uint y2 = clip(combined_bounding_rectangle.y + combined_bounding_rectangle.height + margin, 0, image.rows);

    cv::Mat output(y2 - y1, x2 - x1, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat mask = cv::Mat::zeros(y2 - y1, x2 - x1, CV_8U);
    for (const auto& s : segments) { 
      std::vector<std::vector<cv::Point>> contour;
      contour.push_back(s.contour_);
      cv::drawContours(mask, contour, 0, cv::Scalar(255), -1, cv::LINE_8, cv::noArray(), 1, cv::Point(-x1, -y1)); 
    }
    image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(output, mask);
    imwrite(filename + ".jpg", output);
  }

  void ImageSegmenter::SaveMultipleSegmentsToFiles(const cv::Mat & image, const std::vector<Segment>& segments, const std::string & directory, const std::string & prefix, int margin) const {
    unsigned int i = 0;
    for (const auto& s : segments) {
      std::stringstream ss_filename;
      ss_filename << directory << "/" << prefix << std::setfill('0') << std::setw(8) << i;
      SaveSegmentsToFile(image, s, ss_filename.str(), margin);
      i++;
    }
  }

  void ImageSegmenter::SaveMultipleSegmentsToFiles(const cv::Mat & image, const std::vector<std::vector<Segment>>& segments, const std::string & directory, const std::string & prefix, int margin) const {
    unsigned int i = 0;
    for (const auto& s : segments) {
      std::stringstream ss_filename;
      ss_filename << directory << "/" << prefix << std::setfill('0') << std::setw(8) << i;
      SaveSegmentsToFile(image, s, ss_filename.str(), margin);
      i++;
    }
  }
}
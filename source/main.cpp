#include "opencv2/opencv.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "sort_permutation.h"
#include "utility.h"

using namespace cv;

// Step that is in progress
enum class Step {
  THRESHOLDING,
  SEGMENT_DETECTION,
  SEGMENT_TAGGING,
  SEGMENT_MERGING,
  SEGMENT_EXPORTING
};

// Tags that can be assigned to a segment
enum class Tag {
  NOISE, 
  PARTIAL, 
  MERGED, 
  CORRECT
};

// Data being processed
struct Data {
  Step step;

  Mat input_image_3c;
  Mat input_image_1c;
  int threshold;
  Mat threshold_mask_image;
  std::vector<std::vector<Point>> contours;
  std::vector<uint> areas;
  std::vector<Rect> bounding_rectangles;
  std::vector<Tag> tags;

  std::vector<std::vector<Point>> contours_correct;
  std::vector<Rect> bounding_rectangles_correct;
  std::vector<std::vector<Point>> contours_merged;
  std::vector<Rect> bounding_rectangles_merged;
  std::vector<std::vector<std::vector<Point>>> contours_partial_sets;
  std::vector<std::vector<Rect>> bounding_rectangles_partial_sets;
};

// Settings used for processing
struct Settings {
  std::string input_image_file;
  std::string progress_file;
  uint min_segment_area;
  uint outline_thickness;
  float surroundings_size;
  std::string output_directory;
  uint crop_margin;
};

namespace boost {
  namespace serialization {
    template<class Archive>
    void serialize(Archive& ar, Data& data, const unsigned int version) {
      ar & data.step;
      ar & data.threshold;
      ar & data.tags;
    }
    template<class Archive>
    void serialize(Archive& ar, Settings& settings, const unsigned int version) {
      ar & settings.input_image_file;
      ar & settings.progress_file;
      ar & settings.min_segment_area;
      ar & settings.outline_thickness;
      ar & settings.surroundings_size;
      ar & settings.output_directory;
      ar & settings.crop_margin;
    }
  }
}

// Parse the command line arguments
static bool ParseCommandLineArguments(int argc, char* argv[], Settings* out_settings) {
  const String clp_keys =
    "{help h ? usage | | show help on the command line arguments}"
    "{@image | | image containing characters to be segmented}"
    "{progress-file | | progress file to resume a previous Image Segmentation session }"
    "{outline-thickness | 4 | thickness of the outline used to highlight segments}"
    "{min-area | 20 | min area of a detected character (to remove noise speckles)}"
    "{surroundings-size | 10.0 | relative size of surroundings to show on preview}"
    "{output-dir | output | directory where to store output}"
    "{crop-margin | 2 | margin to add when cropping segments}"
    ;

  // Show help if requested
  CommandLineParser clp(argc, argv, clp_keys);
  if (clp.has("help")) {
    clp.printMessage();
    return false;
  }

  // Parse arguments
  out_settings->input_image_file = clp.get<String>("@image");
  out_settings->progress_file = clp.get<String>("progress-file");
  out_settings->outline_thickness = clp.get<uint>("outline-thickness");
  out_settings->min_segment_area = clp.get<uint>("min-area");
  out_settings->surroundings_size = clp.get<float>("surroundings-size");
  out_settings->output_directory = clp.get<String>("output-dir");
  out_settings->crop_margin = clp.get<uint>("crop-margin");

  // Show errors if any occurred
  if (!clp.check()) {
    clp.printErrors();
    return false;
  }
  if (out_settings->input_image_file.compare("") == 0) {
    std::cout << "ERROR: no input image specified. Use 'CharacterSegmenter -help' for info." << std::endl;
  }

  return true;
}

// Perform thresholding to separate characters from background
static void PerformThresholding(const Mat& image, int t, Mat* out_threshold_mask) {
  threshold(image, *out_threshold_mask, t, 255, THRESH_BINARY);
  imshow("CharacterSegmenter (Step 1. Thresholding)", *out_threshold_mask);
}

// Callback for adjusting the threshold slider
static void CallbackThreshold(int t, void* userdata) {
  Data* data = (Data*)(userdata);
  PerformThresholding(data->input_image_1c, t, &(data->threshold_mask_image));
}

// Perform segmentation to find individual characters
static void PerformSegmentation(const Mat& image_thresholded, uint min_area, std::vector<std::vector<Point>>* out_contours, std::vector<uint>* out_areas, std::vector<Rect>* out_bounding_rectangles) {
  Mat image_thresholded_inverted = 255 - image_thresholded;
  findContours(image_thresholded_inverted, *out_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);  

  // Calculate metadata for all contours
  for (int i = 0; i < out_contours->size(); i++) {
    out_areas->push_back(contourArea(out_contours->at(i)));
    out_bounding_rectangles->push_back(boundingRect(out_contours->at(i)));
  }

  // Sort contours and their metadata by area
  auto p = sort_permutation(*out_areas, [](uint const& a, uint const& b) { return a > b; });
  apply_permutation_in_place(*out_contours, p);
  apply_permutation_in_place(*out_areas, p);
  apply_permutation_in_place(*out_bounding_rectangles, p);

  // Remove contours of zero area
  while (out_areas->size() > 0 && out_areas->back() < min_area) {
    out_contours->pop_back();
    out_areas->pop_back();
    out_bounding_rectangles->pop_back();
  }
}

// Draw the contours of all segments found in the image
static void DrawSegmentationContours(const Mat& image, const std::vector<std::vector<Point>>& contours, int line_thickness) {
  Mat result;
  image.copyTo(result);
  for (int i = 0; i < contours.size(); i++) {
    drawContours(result, contours, i, Scalar(rand() % 255, rand() % 255, rand() % 255), line_thickness);
  }
  imshow("CharacterSegmenter (Step 2. Character detection)", result);
}

// Callback for adjusting the line thickness of contours
static void CallbackLinethickness(int line_thickness, void* userdata) {
  Data* data = (Data*)(userdata);
  DrawSegmentationContours(data->input_image_3c, data->contours, line_thickness);
}

// Generate a preview of a detected contour in its surroundings
static void GeneratePreview(const Mat& image, std::vector<Point>& contour, const Rect& bounding_rectangle, const Scalar& color, float surroundings_size, Mat* out_preview, Mat* out_preview_contour) {
  int size = bounding_rectangle.width > bounding_rectangle.height ? bounding_rectangle.width : bounding_rectangle.height;
  int size_surroundings = size * surroundings_size;
  int x_center = bounding_rectangle.x + bounding_rectangle.width / 2;
  int y_center = bounding_rectangle.y + bounding_rectangle.height / 2;

  uint x1 = clip(x_center - size_surroundings, 0, image.cols);
  uint x2 = clip(x_center + size_surroundings, 0, image.cols);
  uint y1 = clip(y_center - size_surroundings, 0, image.rows);
  uint y2 = clip(y_center + size_surroundings, 0, image.rows);
  
  Mat preview(y2 - y1, x2 - x1, CV_8UC3);
  image(Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(preview);
  *out_preview = preview.clone();

  std::vector<std::vector<Point>> contours;
  contours.push_back(contour);
  drawContours(preview, contours, 0, color, -1, LINE_8, noArray(), 1, Point(-x1, -y1));
  *out_preview_contour = preview.clone();
}

// Generate a preview of multiple contours in their surroundings
static void GenerateMultiPreview(const Mat& image, std::vector<std::vector<Point>>& contours, const std::vector<Rect>& bounding_rectangles, const std::vector<Scalar>& colors, float surroundings_size, Mat* out_preview, Mat* out_preview_contour) {
  Rect combined_bounding_rectangle = GetBoundingRect(bounding_rectangles);
  int size = combined_bounding_rectangle.width > combined_bounding_rectangle.height ? combined_bounding_rectangle.width : combined_bounding_rectangle.height;
  int size_surroundings = size * surroundings_size;
  int x_center = combined_bounding_rectangle.x + combined_bounding_rectangle.width / 2;
  int y_center = combined_bounding_rectangle.y + combined_bounding_rectangle.height / 2;

  uint x1 = clip(x_center - size_surroundings, 0, image.cols);
  uint x2 = clip(x_center + size_surroundings, 0, image.cols);
  uint y1 = clip(y_center - size_surroundings, 0, image.rows);
  uint y2 = clip(y_center + size_surroundings, 0, image.rows);

  Mat preview(y2 - y1, x2 - x1, CV_8UC3);
  image(Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(preview);
  *out_preview = preview.clone();

  for (int i = 0; i < contours.size(); i++) {
    drawContours(preview, contours, i, colors[i], -1, LINE_8, noArray(), 1, Point(-x1, -y1));
  }
  *out_preview_contour = preview.clone();
}

// Move segments of a specific tag to a different vector
static void MoveSegmentsByTag(std::vector<std::vector<Point>>* in_contours, std::vector<Tag>* in_tags, std::vector<Rect>* in_bounding_rectangles, Tag tag, std::vector<std::vector<Point>>* out_contours, std::vector<Rect>* out_bounding_rectangles) {
  int i = 0;
  while (true) {
    if (in_tags->at(i) == tag) {
      out_contours->push_back(in_contours->at(i));
      out_bounding_rectangles->push_back(in_bounding_rectangles->at(i));
      in_contours->erase(in_contours->begin() + i);
      in_tags->erase(in_tags->begin() + i);
      in_bounding_rectangles->erase(in_bounding_rectangles->begin() + i);
    } else {
      i += 1;
    }
    if (i >= in_tags->size()) { break; }
  }
}

// Remove segments of a specific tag
static void RemoveSegmentsByTag(std::vector<std::vector<Point>>* in_contours, std::vector<Tag>* in_tags, std::vector<Rect>* in_bounding_rectangles, Tag tag) {
  int i = 0;
  while (true) {
    if (in_tags->at(i) == tag) {
      in_contours->erase(in_contours->begin() + i);
      in_tags->erase(in_tags->begin() + i);
      in_bounding_rectangles->erase(in_bounding_rectangles->begin() + i);
    }
    else {
      i += 1;
    }
    if (i >= in_tags->size()) { break; }
  }
}

// Saves a segment to an image file
static void SaveSegment(const Mat& image, const std::vector<Point>& contour, const Rect& bounding_rectangle, const String& path, const String& name, int margin) {
  uint x1 = clip(bounding_rectangle.x - margin, 0, image.cols);
  uint x2 = clip(bounding_rectangle.x + bounding_rectangle.width + margin, 0, image.cols);
  uint y1 = clip(bounding_rectangle.y - margin, 0, image.rows);
  uint y2 = clip(bounding_rectangle.y + bounding_rectangle.height + margin, 0, image.rows);

  Mat output(y2 - y1, x2 - x1, CV_8UC3, Scalar(255, 255, 255));
  Mat mask = Mat::zeros(y2 - y1, x2 - x1, CV_8U);
  std::vector<std::vector<Point>> contours;
  contours.push_back(contour);
  drawContours(mask, contours, 0, Scalar(255), -1, LINE_8, noArray(), 1, Point(-x1 , -y1));
  image(Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(output, mask);
  imwrite(path + "/" + name + ".jpg", output);
}

// Saves multiple merged segments to an image file
static void SaveMultiSegment(const Mat& image, const std::vector<std::vector<Point>>& contours, const std::vector<Rect>& bounding_rectangles, const String& path, const String& name, int margin) {
  Rect combined_bounding_rectangle = GetBoundingRect(bounding_rectangles);
  uint x1 = clip(combined_bounding_rectangle.x - margin, 0, image.cols);
  uint x2 = clip(combined_bounding_rectangle.x + combined_bounding_rectangle.width + margin, 0, image.cols);
  uint y1 = clip(combined_bounding_rectangle.y - margin, 0, image.rows);
  uint y2 = clip(combined_bounding_rectangle.y + combined_bounding_rectangle.height + margin, 0, image.rows);

  Mat output(y2 - y1, x2 - x1, CV_8UC3, Scalar(255, 255, 255));
  Mat mask = Mat::zeros(y2 - y1, x2 - x1, CV_8U);
  for (int i = 0; i < contours.size(); i++) { drawContours(mask, contours, i, Scalar(255), -1, LINE_8, noArray(), 1, Point(-x1, -y1)); }
  image(Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(output, mask);
  imwrite(path + "/" + name + ".jpg", output);
}

// Save progress to a file
static void SaveProgress(const String& filename, const Data* data, const Settings* settings) {
  std::ofstream ofs(filename);
  boost::archive::text_oarchive oa(ofs);
  oa << *data;
  oa << *settings;
}

// Load progress from a file
static void LoadProgress(const String& filename, Data* data, Settings* settings) {
  std::ifstream ifs(filename);
  boost::archive::text_iarchive ia(ifs);
  ia >> *data;
  ia >> *settings;
}

// Run the thresholding stage
static void RunThresholdingStage(Data* data, Settings* settings) {
  data->step = Step::THRESHOLDING;
  std::cout << "Step 1. Thresholding" << std::endl;
  std::cout << "================" << std::endl;
  std::cout << ">> Use the slider to define the threshold for separating the characters from" << std::endl
            << "   the background. Characters should appear complete in black without noise." << std::endl
            << "   Adjust the threshold to reduce the noise in the background (white) as much" << std::endl
            << "   as possible, without removing parts of characters." << std::endl;

  namedWindow("CharacterSegmenter (Step 1. Thresholding)", WINDOW_NORMAL);
  data->threshold = 192;
  createTrackbar("Threshold", "CharacterSegmenter (Step 1. Thresholding)", &(data->threshold), 255, CallbackThreshold, data);
  CallbackThreshold(data->threshold, data);
  std::cout << ">> Press [SPACE] to confirm your threshold" << std::endl;
  while (waitKey(0) != ' ');
  destroyWindow("CharacterSegmenter (Step 1. Thresholding)");
}

// Run segment detection stage
static void RunSegmentDetectionStage(Data* data, Settings* settings) {
  data->step = Step::SEGMENT_DETECTION;
  std::cout << "Step 2. Character detection" << std::endl;
  std::cout << "================" << std::endl;
  std::cout << ">> Detecting all individual characters after thresholding. This can" << std::endl
            << "   take a few seconds. Use the line thickness trackbar to adjust the" << std::endl
            << "   size of the detection lines (for visual clarity)." << std::endl;
  PerformSegmentation(data->threshold_mask_image, settings->min_segment_area, &data->contours, &data->areas, &data->bounding_rectangles);

  namedWindow("CharacterSegmenter (Step 2. Character detection)", WINDOW_NORMAL);
  int line_thickness = 4;
  createTrackbar("Line thickness", "CharacterSegmenter (Step 2. Character detection)", &line_thickness, 16, CallbackLinethickness, data);
  CallbackLinethickness(settings->outline_thickness, data);
  std::cout << ">> Press [SPACE] to confirm" << std::endl;
  while (waitKey(0) != ' ');
  destroyWindow("CharacterSegmenter (Step 2. Character detection)");
}

// Run segment tagging stage
static void RunSegmentTaggingStage(Data* data, Settings* settings) {
  data->step = Step::SEGMENT_TAGGING;
  std::cout << "Step 3. Segment tagging" << std::endl;
  std::cout << "================" << std::endl;
  std::cout << ">> Tagging segments, use the following keys:" << std::endl
            << "   [N] Noise segment (will be discarded)" << std::endl
            << "   [P] Partial segment (will be combinable with other partial segments)" << std::endl
            << "   [M] Merged segment (will be stored separately so it can be split)" << std::endl
            << "   [C] Correct segment (will be stored as is)" << std::endl
            << std::endl
            << "   [Z] Undo (move back in the tagging sequence)" << std::endl
            << "================" << std::endl;

  namedWindow("CharacterSegmenter (Step 3. Segment tagging)", WINDOW_NORMAL);
  bool show_contour = true;
  Mat preview;
  Mat preview_contour;
  for (int i = 0; i < data->contours.size(); i++) {
    std::cout << ">> Tagging segment [" << i << "/" << data->contours.size() << "]: ";
    GeneratePreview(data->input_image_3c, data->contours[i], data->bounding_rectangles[i], Scalar(255, 0, 0), settings->surroundings_size, &preview, &preview_contour);
    imshow("CharacterSegmenter (Step 3. Segment tagging)", preview);
    int last_key = -1;
    while (last_key != 'n' && last_key != 'p' && last_key != 'm' && last_key != 'c' && last_key != 'z') {
      show_contour = !show_contour;
      imshow("CharacterSegmenter (Step 3. Segment tagging)", show_contour ? preview_contour : preview);
      last_key = waitKeyEx(250);
    }

    // Process the pressed key
    if (last_key == 'n') {
      std::cout << "NOISE" << std::endl;
      data->tags.push_back(Tag::NOISE);
      continue;
    }
    if (last_key == 'p') {
      std::cout << "PARTIAL" << std::endl;
      data->tags.push_back(Tag::PARTIAL);
      continue;
    }
    if (last_key == 'm') {
      std::cout << "MERGED" << std::endl;
      data->tags.push_back(Tag::MERGED);
      continue;
    }
    if (last_key == 'c') {
      std::cout << "CORRECT" << std::endl;
      data->tags.push_back(Tag::CORRECT);
      continue;
    }
    if (last_key == 'z' && i > 0) {
      std::cout << "... undoing previous tag" << std::endl;
      i -= 2;
      data->tags.pop_back();
      continue;
    }
  }
  destroyWindow("CharacterSegmenter (Step 3. Segment tagging)");
  MoveSegmentsByTag(&data->contours, &data->tags, &data->bounding_rectangles, Tag::CORRECT, &data->contours_correct, &data->bounding_rectangles_correct);
  MoveSegmentsByTag(&data->contours, &data->tags, &data->bounding_rectangles, Tag::MERGED, &data->contours_merged, &data->bounding_rectangles_merged);
  RemoveSegmentsByTag(&data->contours, &data->tags, &data->bounding_rectangles, Tag::NOISE);
  SaveProgress("../progress.cs", data, settings);
}

// Run partial segment merging stage
static void RunPartialSegmentMergingStage(Data* data, Settings* settings) {
  data->step = Step::SEGMENT_MERGING;
  std::cout << "Step 4. Partial segment merging" << std::endl;
  std::cout << "================" << std::endl;
  std::cout << ">> Merging partial segments. The partial segments in [BLUE] are looking for"
            << "partial segments to merge with. The partial segment in [GREEN] proposes to"
            << "to merge. Use the following keys:" << std::endl
            << "   [A] Accept segment (green will be merged with blue)" << std::endl
            << "   [R] Reject segment (green will not be merged with blue)" << std::endl
            << "   [C] Complete merging (blue is complete and will be saved)" << std::endl
            // << "   [S] Skip segment (will be stored as is)" << std::endl
            // << std::endl
            // << "   [Z] Undo (move back in the tagging sequence)" << std::endl
            << "================" << std::endl;

  namedWindow("CharacterSegmenter (Step 4. Partial segment merging)", WINDOW_NORMAL);
  bool show_contour = true;
  Mat preview;
  Mat preview_contours;

  while (data->contours.size() > 0) { // Start new partial set
    std::cout << ">> Starting new partial set." << std::endl;
    std::vector<std::vector<Point>> current_partial_set_contours;
    std::vector<Rect> current_partial_set_bounding_rectangles;
    current_partial_set_contours.push_back(data->contours.front());
    current_partial_set_bounding_rectangles.push_back(data->bounding_rectangles.front());
    data->contours.erase(data->contours.begin());
    data->bounding_rectangles.erase(data->bounding_rectangles.begin());

    std::vector<std::vector<Point>> preview_data_contours(current_partial_set_contours);
    std::vector<Rect> preview_data_bounding_rectangles(current_partial_set_bounding_rectangles);

    int i_proposed = 0;
    while (true) { // Cycle through proposed merge candidates
      if (data->contours.size() == 0) {
        std::cout << "   PARTIAL SET COMPLETED (no partial segments left)" << std::endl;
        break;
      }

      std::cout << "   Proposing partial segment [" << i_proposed << "/" << data->contours.size() << "]: ";
      preview_data_contours.push_back(data->contours.at(i_proposed));
      preview_data_bounding_rectangles.push_back(data->bounding_rectangles.at(i_proposed));
      std::vector<Scalar> preview_data_colours;
      for (int i = 0; i < current_partial_set_contours.size(); i++) { preview_data_colours.push_back(Scalar(255, 0, 0)); }
      preview_data_colours.push_back(Scalar(0, 255, 0));
      GenerateMultiPreview(data->input_image_3c, preview_data_contours, preview_data_bounding_rectangles, preview_data_colours, settings->surroundings_size, &preview, &preview_contours);
      imshow("CharacterSegmenter (Step 4. Partial segment merging)", preview);
      int last_key = -1;
      while (last_key != 'a' && last_key != 'r' && last_key != 'c') {
        show_contour = !show_contour;
        imshow("CharacterSegmenter (Step 4. Partial segment merging)", show_contour ? preview_contours : preview);
        last_key = waitKeyEx(250);
      }     

      // Process the pressed key
      if (last_key == 'a') {
        std::cout << "   ACCEPTED" << std::endl;
        current_partial_set_contours.push_back(data->contours.at(i_proposed));
        current_partial_set_bounding_rectangles.push_back(data->bounding_rectangles.at(i_proposed));
        data->contours.erase(data->contours.begin() + i_proposed);
        data->bounding_rectangles.erase(data->bounding_rectangles.begin() + i_proposed);
        i_proposed = (i_proposed >= data->contours.size()) ? 0 : i_proposed;
        continue;
      }
      if (last_key == 'r') {
        std::cout << "   REJECTED" << std::endl;
        preview_data_contours.pop_back();
        preview_data_bounding_rectangles.pop_back();
        i_proposed = (i_proposed + 1 >= data->contours.size()) ? 0 : i_proposed + 1;
        continue;
      }
      if (last_key == 'c') {
        std::cout << "   REJECTED" << std::endl << "   PARTIAL SET COMPLETED" << std::endl;
        break;
      }
    }
    data->contours_partial_sets.push_back(current_partial_set_contours);
    data->bounding_rectangles_partial_sets.push_back(current_partial_set_bounding_rectangles);
  }
}

// Run segment exporting stage
static void RunSegmentExportingStage(Data* data, Settings* settings) {
  data->step = Step::SEGMENT_EXPORTING;
  std::cout << "Step 5. Segment exporting" << std::endl;
  std::cout << "================" << std::endl;
  std::cout << ">> Isolating segments and exporting to files." << std::endl
            << "================" << std::endl;

  std::stringstream ss_dir;
  std::cout << "   Exporting [Correct segments]: ... ";
  ss_dir.clear();
  ss_dir << settings->output_directory << "/correct";
  for (int i = 0; i < data->contours_correct.size(); i++) {
    std::stringstream ss_name;
    ss_name << "c" << std::setfill('0') << std::setw(8) << i;
    SaveSegment(data->input_image_3c, data->contours_correct[i], data->bounding_rectangles_correct[i], ss_dir.str(), ss_name.str(), settings->crop_margin);
  }
  std::cout << "DONE" << std::endl;

  std::cout << "   Exporting [Merged segments]: ... ";
  ss_dir.clear();
  ss_dir << settings->output_directory << "/merged";
  for (int i = 0; i < data->contours_merged.size(); i++) {
    std::stringstream ss_name;
    ss_name << "m" << std::setfill('0') << std::setw(8) << i;
    SaveSegment(data->input_image_3c, data->contours_merged[i], data->bounding_rectangles_merged[i], ss_dir.str(), ss_name.str(), settings->crop_margin);
  }
  std::cout << "DONE" << std::endl;

  std::cout << "   Exporting [Merged partial segments]: ... ";
  ss_dir.clear();
  ss_dir << settings->output_directory << "/correct";
  for (int i = 0; i < data->contours_partial_sets.size(); i++) {
    std::stringstream ss_name;
    ss_name << "p" << std::setfill('0') << std::setw(8) << i;
    SaveMultiSegment(data->input_image_3c, data->contours_partial_sets[i], data->bounding_rectangles_partial_sets[i], ss_dir.str(), ss_name.str(), settings->crop_margin);
  }
  std::cout << "DONE" << std::endl;
}

// Run character segmentation procedure
int main(int argc, char* argv[]) {

  Data data;
  Settings settings;

  // Parse command line arguments
  if (!ParseCommandLineArguments(argc, argv, &settings)) { return 1; }

  // Load image
  data.input_image_3c = imread(settings.input_image_file, IMREAD_COLOR);
  cvtColor(data.input_image_3c, data.input_image_1c, CV_BGR2GRAY);

  // Run the procedure

  LoadProgress("../progress.cs", &data, &settings);

  RunThresholdingStage(&data, &settings);
  RunSegmentDetectionStage(&data, &settings);
  RunSegmentTaggingStage(&data, &settings);
  RunPartialSegmentMergingStage(&data, &settings);
  RunSegmentExportingStage(&data, &settings);

  return 0;
}
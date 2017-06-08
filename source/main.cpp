#include "opencv2/opencv.hpp"
#include "image_segmenter.h"

// Run character segmentation procedure
int main(int argc, char* argv[]) {

  // Parse command line arguments
  const cv::String clp_keys =
    "{help h ? usage | | show help on the command line arguments}"
    "{@image | | image containing characters to be segmented}"
    "{progress-file | | progress file to resume a previous Image Segmentation session }"
    "{threshold | 192 | luminosity threshold for background/foreground separation }"
    "{min-area | 20 | min area of a detected character (to remove noise speckles)}"
    "{outline-thickness | 4 | thickness of the outline used to highlight segments}"
    "{surroundings-size | 10.0 | relative size of surroundings to show on preview}"
    "{output-dir | output | directory where to store output}"
    "{crop-margin | 2 | margin to add when cropping segments}"
    ;

  cv::CommandLineParser clp(argc, argv, clp_keys);
  if (clp.has("help")) {
    clp.printMessage();
    return -1;
  }

  cv::String image_file = clp.get<cv::String>("@image");
  cv::String progress_file = clp.get<cv::String>("progress-file");
  unsigned char threshold = (unsigned char)clp.get<unsigned int>("threshold");
  unsigned char outline_thickness = (unsigned char)clp.get<unsigned int>("outline-thickness");
  unsigned int min_segment_area = clp.get<unsigned int>("min-area");
  float surroundings_size = clp.get<float>("surroundings-size");
  cv::String output_directory = clp.get<cv::String>("output-dir");
  unsigned int crop_margin = clp.get<unsigned int>("crop-margin");

  if (!clp.check()) {
    clp.printErrors();
    return -1;
  }
  if (image_file.compare("") == 0) {
    std::cout << "ERROR: no input image specified. Use 'ImageSegmenter -help' for info." << std::endl;
    return -1;
  }

  // Perform image segmentation
  j::ImageSegmenter is(image_file, threshold, min_segment_area, outline_thickness, surroundings_size, output_directory, crop_margin);
  if (progress_file.compare("") != 0) { is.LoadProgress(progress_file); }

  is.RunThresholdingStep(true);
  is.RunSegmentDetectionStep(true);
  is.RunSegmentTaggingStep();
  is.RunPartialSegmentMergingStep();
  is.RunSegmentExportingStep();

  return 0;
}
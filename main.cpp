#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "src/CardRecognizer.h"
#include "src/HarrisCornerDetector.h"
#include "src/SIFTDetector.h"

#define TESTING_IMAGE_PATH "D:/2. Area/facultate/card-recognision/cards_photos/set4"

namespace fs = std::filesystem;

int test_image(
    grg::HarrisCornerDetector harris, 
    grg::SIFTDetector siftDetector,
    grg::CardRecognizer recognizer, 
    std::string input
) {
  /** Open image */
  const cv::Mat image = cv::imread(input, cv::IMREAD_COLOR);

  if (image.empty()) {
    std::cout << "Image not found!" << std::endl;
    return -1;
  }

  /** Compute corners with Harris */
  const cv::Mat result = harris.detect(image);
  std::string windowName = "Harris Corners" + input;
  cv::imshow(windowName, result);

  /** Compute descriptors with SIFT */
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keyPoints;
  siftDetector.detectAndCompute(image, keyPoints, descriptors);
  const cv::Mat outputImage = siftDetector.drawKeyPointsOnImage(image, keyPoints);

  windowName = "SIFT key points" + input;
  cv::imshow(windowName, outputImage);

  /** Determine card type */
  cv::Mat result2 = recognizer.recognize(image);
  windowName = "Rezultat" + input;
  cv::imshow(windowName, result2);
}

int main() {
  const grg::HarrisCornerDetector harris(0.04f, 130.0, 3, 1.0);
  const grg::SIFTDetector siftDetector(0, 3, 0.24, 10, 1.6);
  const grg::CardRecognizer recognizer(0.04f, 100.0, 3, 2.0, 0, 3, 0.04, 10, 1.6);

  for (const auto &entry : fs::directory_iterator(TESTING_IMAGE_PATH)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      std::string input_path = entry.path().string();

      if (filename.find('9') != std::string::npos) {
        test_image(harris, siftDetector, recognizer, input_path);
      }
    }
  }

  cv::waitKey(0);

  return 0;
}

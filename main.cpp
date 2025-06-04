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
  /*cv::Mat result2 = recognizer.recognize(image);*/
  /*windowName = "Rezultat" + input;*/
  /*cv::imshow(windowName, result2);*/
}

int main(void) {
  const grg::HarrisCornerDetector harris(0.04f, 130.0, 3, 1.0);
  const cv::Mat image = cv::imread("D:\\2. Area\\facultate\\card-recognision\\cards_photos\\set4\\hearts_4.png", cv::IMREAD_COLOR); 

  const cv::Mat result = harris.detect(image);
  std::string windowName = "Harris Corners";
  cv::imshow(windowName, result);

  int roiW = image.cols / 5;    // e.g. left 25% of width
  int roiH = image.rows / 4;    // top 20% of height
  cv::Rect roi(0, 0, roiW, roiH);
  cv::Mat tl = result(roi);

  cv::Mat gray, tresh;
  cv::cvtColor(tl, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, tresh, harris.getThreshold(), 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(tresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  cv::Mat contourVis;
  cv::cvtColor(tresh, contourVis, cv::COLOR_GRAY2BGR);
  cv::drawContours(contourVis, contours, -1, cv::Scalar(0, 255, 0), 2);
  cv::imshow("countours", contourVis);

  std::vector<cv::Rect> rects;
  for (auto &c : contours) {
    cv::Rect bb = rects[0];
    for (size_t i = 1; i < rects.size(); ++i)
      bb |= rects[i];

    bb.x += roi.x;
    bb.y += roi.y;
    /*cv::rectangle(image, bb, cv::Scalar(0,255,0), 2);*/
  }

  // 8. Show results
  cv::imshow("Detected Top-Left Symbol", image);

  cv::waitKey(0);
}

int main2() {
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

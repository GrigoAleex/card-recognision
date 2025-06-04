//
// Created by grig on 03/04/2025.
//

#include "CardRecognizer.h"

namespace grg {

CardRecognizer::CardRecognizer(float harrisK, double harrisThreshold,
                               int harrisBlockSize, double harrisSigma,
                               int siftFeatures, int siftOctaveLayers,
                               double siftContrastThreshold,
                               double siftEdgeThreshold, double siftSigma)
    : harrisDetector(harrisK, harrisThreshold, harrisBlockSize, harrisSigma),
      siftDetector(siftFeatures, siftOctaveLayers, siftContrastThreshold,
                   siftEdgeThreshold, siftSigma) {
  /*loadTemplatesFromFolder(R"(D:\2. Area\facultate\card-recognision\cards_photos\set1)");*/
  /*loadTemplatesFromFolder(R"(D:\2. Area\facultate\card-recognision\cards_photos\set2)");*/
  /*loadTemplatesFromFolder(R"(D:\2. Area\facultate\card-recognision\cards_photos\set3)");*/
  loadTemplatesFromFolder("D:\\2. Area\\facultate\\card-recognision\\cards_photos\\set4");
}

void CardRecognizer::loadTemplatesFromFolder(const std::string &folderPath) {
  for (const auto &entry : fs::directory_iterator(folderPath)) {
    if (entry.path().extension() == ".png") {
      cv::Mat templ = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
      if (templ.empty()) {
        std::cerr << "Template-ul nu a fost incarcat: " << entry.path()
                  << std::endl;
        continue;
      }

      templates.push_back(templ);
      std::vector<cv::KeyPoint> kp;
      cv::Mat desc;
      siftDetector.detectAndCompute(templ, kp, desc);
      templateKeyPoints.push_back(kp);
      templateDescriptors.push_back(desc);
      templateNames.push_back(entry.path().stem().string());
    }
  }
}

std::string CardRecognizer::recognize(const cv::Mat &inputImage) const {
  cv::Mat harrisResponse = harrisDetector.detect(inputImage);
  cv::Mat grayResponse;
  cv::cvtColor(harrisResponse, grayResponse, cv::COLOR_BGR2GRAY);

  cv::Mat harrisThresh;
  cv::threshold(grayResponse, harrisThresh, harrisDetector.getThreshold(), 255, cv::THRESH_BINARY);
  harrisThresh.convertTo(harrisThresh, CV_8U);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(harrisThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::string bestMatchName = "Unknown";

  for (auto &contour : contours) {
    std::vector<cv::Point> approx;
    double peri = cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);

    if (approx.size() == 4 && cv::contourArea(approx) > 500) {
      cv::Mat cardImage = fourPointTransform(inputImage, approx);

      std::vector<cv::KeyPoint> cardKeypoints;
      cv::Mat cardDescriptors;
      siftDetector.detectAndCompute(cardImage, cardKeypoints, cardDescriptors);

      int bestMatchesCount = 0;
      for (size_t i = 0; i < templateDescriptors.size(); i++) {
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<cv::DMatch> matches;

        if (cardDescriptors.empty() || templateDescriptors[i].empty())
          continue;

        matcher.match(cardDescriptors, templateDescriptors[i], matches);

        int goodMatches = 0;
        for (auto &m : matches) {
          if (m.distance < 300) {
            goodMatches++;
          }
        }

        if (goodMatches > bestMatchesCount) {
          bestMatchesCount = goodMatches;
          bestMatchName = templateNames[i];
        }
      }
      
    }
  }
  return bestMatchName;
}

cv::Mat
CardRecognizer::fourPointTransform(const cv::Mat &image,
                                   const std::vector<cv::Point> &points) {
  std::vector<cv::Point2f> rect = orderPoints(points);
  float widthA = std::hypot(rect[2].x - rect[3].x, rect[2].y - rect[3].y);
  float widthB = std::hypot(rect[1].x - rect[0].x, rect[1].y - rect[0].y);
  float maxWidth = std::max(widthA, widthB);
  float heightA = std::hypot(rect[1].x - rect[2].x, rect[1].y - rect[2].y);
  float heightB = std::hypot(rect[0].x - rect[3].x, rect[0].y - rect[3].y);
  float maxHeight = std::max(heightA, heightB);

  std::vector<cv::Point2f> dst = {{0, 0},
                                  {maxWidth - 1, 0},
                                  {maxWidth - 1, maxHeight - 1},
                                  {0, maxHeight - 1}};

  cv::Mat M = cv::getPerspectiveTransform(rect, dst);
  cv::Mat warped;
  cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));
  return warped;
}

std::vector<cv::Point2f>
CardRecognizer::orderPoints(const std::vector<cv::Point> &pts) {
  std::vector<cv::Point2f> points;
  for (auto &p : pts)
    points.emplace_back(p.x, p.y);

  std::sort(
      points.begin(), points.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });

  std::vector<cv::Point2f> ordered(4);
  std::vector<cv::Point2f> leftPoints(points.begin(), points.begin() + 2);
  std::vector<cv::Point2f> rightPoints(points.begin() + 2, points.end());

  std::sort(
      leftPoints.begin(), leftPoints.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
  std::sort(
      rightPoints.begin(), rightPoints.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });

  ordered[0] = leftPoints[0];  // top-left
  ordered[3] = leftPoints[1];  // bottom-left
  ordered[1] = rightPoints[0]; // top-right
  ordered[2] = rightPoints[1]; // bottom-right

  return ordered;
}

} // namespace grg

//
// Created by grig on 03/04/2025.
//

#ifndef CARDRECOGNIZER_H
#define CARDRECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include "HarrisCornerDetector.h"
#include "SIFTDetector.h"

namespace fs = std::filesystem;

namespace grg {

class CardRecognizer {
public:
    CardRecognizer(
        float harrisK,
        double harrisThreshold,
        int harrisBlockSize,
        double harrisSigma,
        int siftFeatures = 0,
        int siftOctaveLayers = 3,
        double siftContrastThreshold = 0.04,
        double siftEdgeThreshold = 10,
        double siftSigma = 1.6
    );

    void loadTemplatesFromFolder(const std::string& folderPath);
    std::string recognize(const cv::Mat &inputImage) const;

private:
    static cv::Mat fourPointTransform(const cv::Mat &image, const std::vector<cv::Point> &points);
    static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point> &pts);

    HarrisCornerDetector harrisDetector;
    SIFTDetector siftDetector;

    std::vector<cv::Mat> templates;
    std::vector<std::vector<cv::KeyPoint>> templateKeyPoints;
    std::vector<cv::Mat> templateDescriptors;
    std::vector<std::string> templateNames;
};

} // grg

#endif //CARDRECOGNIZER_H

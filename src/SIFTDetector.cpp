//
// Created by grig on 03/04/2025.
//

#include "SIFTDetector.h"

namespace grg {
    SIFTDetector::SIFTDetector(
        const int noFeatures,
        const int nOctaveLayers,
        const double contrastThreshold,
        const double edgeThreshold,
        const double sigma
    ) : noFeatures_(noFeatures),
        nOctaveLayers_(nOctaveLayers),
        contrastThreshold_(contrastThreshold),
        edgeThreshold_(edgeThreshold),
        sigma_(sigma)
    {
        sift_ = cv::SIFT::create(noFeatures_, nOctaveLayers_, contrastThreshold_, edgeThreshold_, sigma_);
    }

    void SIFTDetector::detectAndCompute(
        const cv::Mat &inputImage,
        std::vector<cv::KeyPoint> &keyPoints,
        cv::Mat &descriptors
    ) const
    {
        cv::Mat gray;

        if (inputImage.channels() == 3)
            cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
        else
            gray = inputImage.clone();

        sift_->detectAndCompute(gray, cv::Mat(), keyPoints, descriptors);
    }

    cv::Mat SIFTDetector::drawKeyPointsOnImage(const cv::Mat &inputImage, const std::vector<cv::KeyPoint> &keyPoints)
    {
        cv::Mat output;

        cv::drawKeypoints(
            inputImage,
            keyPoints,
            output,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
        );

        return output;
    }

} // grg
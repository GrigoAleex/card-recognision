//
// Created by grig on 03/04/2025.
//

#ifndef SIFTDETECTOR_H
#define SIFTDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

namespace grg {

    class SIFTDetector {
    public:
        explicit SIFTDetector(
            int noFeatures = 0,
            int nOctaveLayers = 3,
            double contrastThreshold = 0.04,
            double edgeThreshold = 10,
            double sigma = 1.6
        );

        void detectAndCompute(const cv::Mat &inputImage, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors) const;
        static cv::Mat drawKeyPointsOnImage(const cv::Mat &inputImage, const std::vector<cv::KeyPoint> &keyPoints);

    private:
        int noFeatures_;
        int nOctaveLayers_;
        double contrastThreshold_;
        double edgeThreshold_;
        double sigma_;
        cv::Ptr<cv::SIFT> sift_;
    };

} // grg

#endif //SIFTDETECTOR_H

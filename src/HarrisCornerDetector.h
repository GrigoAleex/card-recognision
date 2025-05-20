//
// Created by grig on 03/04/2025.
//

#ifndef HARRISCORNERDETECTOR_H
#define HARRISCORNERDETECTOR_H

#include <opencv2/opencv.hpp>

namespace grg {

class HarrisCornerDetector {
    public:
        explicit HarrisCornerDetector(float k = 0.04f, double threshold = 100, int blockSize = 3, double sigma = 2.0);

        cv::Mat detect(const cv::Mat& inputImage) const;
        double getThreshold() const;

    private:
        float k_;
        double threshold_;
        int blockSize_;
        double sigma_;
};

} // grg

#endif //HARRISCORNERDETECTOR_H

//
// Created by grig on 03/04/2025.
//

#include "HarrisCornerDetector.h"

namespace grg {

    HarrisCornerDetector::HarrisCornerDetector(const float k, const double threshold, const int blockSize, const double sigma)
        : k_(k), threshold_(threshold), blockSize_(blockSize), sigma_(sigma) { }

    cv::Mat HarrisCornerDetector::detect(const cv::Mat& inputImage) const
    {
        cv::Mat src = inputImage.clone();

        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);

        cv::Mat Ix, Iy;
        cv::Sobel(gray, Ix, CV_32F, 1, 0, 3);
        cv::Sobel(gray, Iy, CV_32F, 0, 1, 3);

        cv::Mat Ix2 = Ix.mul(Ix);
        cv::Mat Iy2 = Iy.mul(Iy);
        cv::Mat Ixy = Ix.mul(Iy);

        cv::Mat Sx2, Sy2, Sxy;
        cv::GaussianBlur(Ix2, Sx2, cv::Size(blockSize_, blockSize_), sigma_);
        cv::GaussianBlur(Iy2, Sy2, cv::Size(blockSize_, blockSize_), sigma_);
        cv::GaussianBlur(Ixy, Sxy, cv::Size(blockSize_, blockSize_), sigma_);

        cv::Mat R = cv::Mat::zeros(gray.size(), CV_32F);
        for (int y = 0; y < gray.rows; y++) {
            for (int x = 0; x < gray.cols; x++) {
                float a = Sx2.at<float>(y, x);
                float b = Sxy.at<float>(y, x);
                float c = Sy2.at<float>(y, x);

                float det = a * c - b * b;
                float trace = a + c;
                R.at<float>(y, x) = det - k_ * trace * trace;
            }
        }

        cv::Mat R_norm, R_norm_scaled;
        cv::normalize(R, R_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::convertScaleAbs(R_norm, R_norm_scaled);

        for (int y = 0; y < R_norm.rows; y++) {
            for (int x = 0; x < R_norm.cols; x++) {
                if (R_norm_scaled.at<uchar>(y, x) > threshold_) {
                    cv::circle(src, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), 1, 8, 0);
                }
            }
        }

        return src;
    }

    double HarrisCornerDetector::getThreshold() const
    {
        return threshold_;
    }
} // grg
#include <iostream>
#include <opencv2/opencv.hpp>

#define TESTING_IMAGE_PATH "D:\\2. Area\\facultate\\card-recognision\\cards_photos\\set1\\2_of_hearts.png"

class HarrisCornerDetector {
public:
   explicit HarrisCornerDetector(const float k = 0.04f,const double threshold = 100,const int blockSize = 3,const double sigma = 2.0)
        : k_(k), threshold_(threshold), blockSize_(blockSize), sigma_(sigma) {}

    cv::Mat detect(const cv::Mat& inputImage) const
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

private:
    float k_;             // Parametrul k din formula Harris
    double threshold_;    // Pragul pentru detecția colțurilor
    int blockSize_;       // Dimensiunea blocului pentru filtrarea Gaussian
    double sigma_;        // Sigma pentru filtrul Gaussian
};

int main() {
    const cv::Mat image = cv::imread(TESTING_IMAGE_PATH, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Image not found!" << std::endl;
        return -1;
    }

    cv::imshow("Original", image);

    const HarrisCornerDetector harris(0.04f, 80.0);
    const cv::Mat result = harris.detect(image);

    cv::imshow("Harris Corners", result);

    cv::waitKey(0);
    return 0;
}

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "src/CardRecognizer.h"
#include "src/HarrisCornerDetector.h"
#include "src/SIFTDetector.h"

#define TESTING_IMAGE_PATH "D:\\2. Area\\facultate\\card-recognision\\cards_photos\\test\\king_of_diamonds.png"

int main() {
    const cv::Mat image = cv::imread(TESTING_IMAGE_PATH, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Image not found!" << std::endl;
        return -1;
    }

    cv::imshow("Original", image);

    const grg::HarrisCornerDetector harris(0.04f, 80.0);
    const cv::Mat result = harris.detect(image);

    cv::imshow("Harris Corners", result);

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    grg::SIFTDetector siftDetector(0, 3, 0.04, 10, 1.6);

    siftDetector.detectAndCompute(image, keyPoints, descriptors);
    const cv::Mat outputImage = siftDetector.drawKeyPointsOnImage(image, keyPoints);

    cv::imshow("SIFT key points", outputImage);

    // Creăm obiectul CardRecognizer cu parametrii doriti
    grg::CardRecognizer recognizer(0.04f, 100.0, 3, 2.0, 0, 3, 0.04, 10, 1.6);

    // Recunoaștem cardurile din imagine
    cv::Mat result2 = recognizer.recognize(image);

    // Afișăm rezultatul
    cv::imshow("Recunoașterea cardurilor", result2);

    cv::waitKey(0);
    return 0;
}

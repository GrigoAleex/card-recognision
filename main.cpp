#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "src/CardRecognizer.h"
#include "src/HarrisCornerDetector.h"
#include "src/SIFTDetector.h"

#define TESTING_IMAGE_PATH                                                     \
  "/Users/grig/Documents/2.Areas/college/card-recognision/cards_photos/set4"

int test_image(grg::HarrisCornerDetector harris, grg::SIFTDetector siftDetector,
               grg::CardRecognizer recognizer, std::string input) {
  const cv::Mat image = cv::imread(input, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cout << "Image not found!" << std::endl;
    return -1;
  }

  // cv::imshow("Original", image);

  const cv::Mat result = harris.detect(image);
  std::string windowName = "Harris Corners" + input;
  cv::imshow(windowName, result);

  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keyPoints;

  siftDetector.detectAndCompute(image, keyPoints, descriptors);
  const cv::Mat outputImage =
      siftDetector.drawKeyPointsOnImage(image, keyPoints);

  windowName = "SIFT key points" + input;
  cv::imshow(windowName, outputImage);

  // Recunoaștem cardurile din imagine
  cv::Mat result2 = recognizer.recognize(image);

  windowName = "Rezultat" + input;
  // Afișăm rezultatul
  cv::imshow(windowName, result2);
}

namespace fs = std::filesystem;

int main() {
  const grg::HarrisCornerDetector harris(0.04f, 130.0, 3, 1.0);
  const grg::SIFTDetector siftDetector(0, 3, 0.24, 10, 1.6);
  const grg::CardRecognizer recognizer(0.04f, 100.0, 3, 2.0, 0, 3, 0.04, 10,
                                       1.6);

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

// #include <opencv2/opencv.hpp>
// #include <filesystem>
// #include <iostream>
// #include <random>
//
// namespace fs = std::filesystem;
//
// // Funcție pentru a modifica aleator H, S și L ale fiecărui pixel
// cv::Mat modifyHSL(const cv::Mat& img) {
//     cv::Mat imgHLS;
//     cv::cvtColor(img, imgHLS, cv::COLOR_BGR2HLS); // OpenCV folosește HLS, nu
//     HSL exact
//
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dH(-30, 30);  // modificări pentru Hue
//     std::uniform_int_distribution<> dL(-40, 40);  // modificări pentru
//     Lightness std::uniform_int_distribution<> dS(-40, 40);  // modificări
//     pentru Saturation
//
//     for (int y = 0; y < imgHLS.rows; ++y) {
//         for (int x = 0; x < imgHLS.cols; ++x) {
//             cv::Vec3b& pixel = imgHLS.at<cv::Vec3b>(y, x);
//
//             int h = pixel[0] + dH(gen);
//             int l = pixel[1] + dL(gen);
//             int s = pixel[2] + dS(gen);
//
//             pixel[0] = cv::saturate_cast<uchar>((h + 180) % 180); // Hue în
//             [0, 180] pixel[1] = cv::saturate_cast<uchar>(l); pixel[2] =
//             cv::saturate_cast<uchar>(s);
//         }
//     }
//
//     cv::Mat result;
//     cv::cvtColor(imgHLS, result, cv::COLOR_HLS2BGR);
//     return result;
// }
//
// int main() {
//     std::string input_folder = "D:\\2.
//     Area\\facultate\\card-recognision\\cards_photos\\set4"; std::string
//     output_folder = "D:\\2.
//     Area\\facultate\\card-recognision\\cards_photos\\test_hls";
//
//     // Creează folderul de ieșire dacă nu există
//     fs::create_directory(output_folder);
//
//     for (const auto& entry : fs::directory_iterator(input_folder)) {
//         if (entry.is_regular_file()) {
//             std::string filename = entry.path().filename().string();
//             std::string input_path = entry.path().string();
//             std::string output_path = output_folder + "/" + filename;
//
//             cv::Mat img = cv::imread(input_path);
//             if (img.empty()) {
//                 std::cerr << "Eroare la citirea: " << input_path <<
//                 std::endl; continue;
//             }
//
//             cv::Mat modified = modifyHSL(img);
//             cv::imwrite(output_path, modified);
//
//             std::cout << "Salvat: " << output_path << std::endl;
//         }
//     }
//
//     return 0;
// }

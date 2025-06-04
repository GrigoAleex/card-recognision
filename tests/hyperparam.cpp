
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include "../src/CardRecognizer.h"

namespace fs = std::filesystem;

struct HyperParams {
  float harrisK;
  double harrisThreshold;
  int harrisBlockSize;
  double harrisSigma;
  int siftFeatures;
  int siftOctaveLayers;
  double siftContrastThreshold;
  double siftEdgeThreshold;
  double siftSigma;
};

std::vector<HyperParams> generateParamGrid() {
  std::vector<HyperParams> grid;
  for (float k : {0.04f, 0.05f, 0.055f})
  for (double thresh : {80.0, 100.0})
  for (int block : {2, 3})
  for (double sigma : {1.0, 1.4})
  for (int layers : {2, 3})
  for (double contrast : {0.03, 0.04})
  for (double edge : {5.0, 10.0})
  for (double siftSigma : {1.2, 1.6}) {
    grid.push_back({k, thresh, block, sigma, 0, layers, contrast, edge, siftSigma});
  }
  return grid;
}

int main() {
  std::string input_folder = "D:\\2. Area\\facultate\\card-recognision\\cards_photos\\set4";
  auto paramGrid = generateParamGrid();

  float bestAccuracy = -1.0f;
  HyperParams bestParams;

  for (const auto& params : paramGrid) {
    const grg::CardRecognizer recognizer(
      params.harrisK,
      params.harrisThreshold,
      params.harrisBlockSize,
      params.harrisSigma,
      params.siftFeatures,
      params.siftOctaveLayers,
      params.siftContrastThreshold,
      params.siftEdgeThreshold,
      params.siftSigma
    );

    size_t successCounter = 0, sampleSize = 0;

    for (const auto& entry : fs::directory_iterator(input_folder)) {
      if (!entry.is_regular_file()) continue;
      std::string filename = entry.path().filename().string();
      filename = filename.substr(0, filename.size() - 4); // remove .png

      cv::Mat img = cv::imread(entry.path().string());
      if (img.empty()) continue;

      sampleSize++;
      std::string matchFound = recognizer.recognize(img);
      if (matchFound == filename) successCounter++;
    }

    float accuracy = 1.0f * successCounter / sampleSize;
    std::cout << "Accuracy: " << accuracy * 100 << "% | Params => "
              << "K=" << params.harrisK
              << ", T=" << params.harrisThreshold
              << ", B=" << params.harrisBlockSize
              << ", σ=" << params.harrisSigma
              << ", Layers=" << params.siftOctaveLayers
              << ", Contrast=" << params.siftContrastThreshold
              << ", Edge=" << params.siftEdgeThreshold
              << ", SiftSigma=" << params.siftSigma << std::endl;

    if (accuracy > bestAccuracy) {
      bestAccuracy = accuracy;
      bestParams = params;
    }
  }

  std::cout << "\n=== Best Accuracy: " << bestAccuracy * 100.0f << "% ===\n";
  std::cout << "Best HyperParams:\n"
            << "  HarrisK: " << bestParams.harrisK << "\n"
            << "  HarrisThreshold: " << bestParams.harrisThreshold << "\n"
            << "  HarrisBlockSize: " << bestParams.harrisBlockSize << "\n"
            << "  HarrisSigma: " << bestParams.harrisSigma << "\n"
            << "  SiftOctaveLayers: " << bestParams.siftOctaveLayers << "\n"
            << "  SiftContrastThreshold: " << bestParams.siftContrastThreshold << "\n"
            << "  SiftEdgeThreshold: " << bestParams.siftEdgeThreshold << "\n"
            << "  SiftSigma: " << bestParams.siftSigma << "\n";

  return 0;
}

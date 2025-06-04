#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "../src/CardRecognizer.h"
namespace fs = std::filesystem;

int main() {
  std::cout << "Testing..." << std::endl;

  std::string input_folder = "D:\\2. Area\\facultate\\card-recognision\\cards_photos\\set4"; 
  const grg::CardRecognizer recognizer(0.055f, 100.0, 3, 1.4, 0, 3, 0.04, 10, 1.6);
  std::vector<std::pair<std::string, std::string>> errors;
  size_t successCounter = 0;
  size_t sampleSize = 0;

  for (const auto& entry : fs::directory_iterator(input_folder)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      std::string input_path = entry.path().string();
      filename = filename.substr(0, filename.size() - 4); /** remove the .png extension */

      cv::Mat img = cv::imread(input_path);

      if (img.empty()) {
        std::cerr << "Eroare la citirea: " << input_path << std::endl;
        continue;
      }

      sampleSize++;
      std::string matchFound = recognizer.recognize(img);

      if (filename == matchFound) {
        std::cout << ".";
        successCounter++;
      } else {
        std::cout << "x";
        errors.push_back(std::make_pair(filename, matchFound));
      }
    }
  }

  std::cout << std::endl;
  for (const auto &p : errors) {
    std::cout << "--- Expected: " << p.first << std::endl;
    std::cout << "+++ Actual: " << p.second << std::endl << std::endl;
  }

  std::cout << "[DEBUG]: successCounter: " << successCounter << std::endl;
  std::cout << "[DEBUG]: sampleSize: " << sampleSize << std::endl;

  float accuracy = 1.0f * successCounter / sampleSize;
  std::cout << "END! Final accuracy: " << accuracy * 100.0f << "%" << std::endl;

  return 0;
}

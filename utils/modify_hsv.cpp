
 #include <opencv2/opencv.hpp>
 #include <filesystem>
 #include <iostream>
 #include <random>

 namespace fs = std::filesystem;

 cv::Mat modifyHSL(const cv::Mat& img) {
     cv::Mat imgHLS;
     cv::cvtColor(img, imgHLS, cv::COLOR_BGR2HLS); // OpenCV folosește HLS, nu
     HSL exact

     std::random_device rd;
     std::mt19937 gen(rd());
     std::uniform_int_distribution<> dH(-30, 30);  // modificări pentru Hue
     std::uniform_int_distribution<> dL(-40, 40);  // modificări pentru
     Lightness std::uniform_int_distribution<> dS(-40, 40);  // modificări
     pentru Saturation

     for (int y = 0; y < imgHLS.rows; ++y) {
         for (int x = 0; x < imgHLS.cols; ++x) {
             cv::Vec3b& pixel = imgHLS.at<cv::Vec3b>(y, x);

             int h = pixel[0] + dH(gen);
             int l = pixel[1] + dL(gen);
             int s = pixel[2] + dS(gen);

             pixel[0] = cv::saturate_cast<uchar>((h + 180) % 180); // Hue în
             [0, 180] pixel[1] = cv::saturate_cast<uchar>(l); pixel[2] =
             cv::saturate_cast<uchar>(s);
         }
     }

     cv::Mat result;
     cv::cvtColor(imgHLS, result, cv::COLOR_HLS2BGR);
     return result;
 }

 int main() {
     std::string input_folder = "D:\\2.
     Area\\facultate\\card-recognision\\cards_photos\\set4"; std::string
     output_folder = "D:\\2.
     Area\\facultate\\card-recognision\\cards_photos\\test_hls";

     // Creează folderul de ieșire dacă nu există
     fs::create_directory(output_folder);

     for (const auto& entry : fs::directory_iterator(input_folder)) {
         if (entry.is_regular_file()) {
             std::string filename = entry.path().filename().string();
             std::string input_path = entry.path().string();
             std::string output_path = output_folder + "/" + filename;

             cv::Mat img = cv::imread(input_path);
             if (img.empty()) {
                 std::cerr << "Eroare la citirea: " << input_path <<
                 std::endl; continue;
             }

             cv::Mat modified = modifyHSL(img);
             cv::imwrite(output_path, modified);

             std::cout << "Salvat: " << output_path << std::endl;
         }
     }

     return 0;
 }

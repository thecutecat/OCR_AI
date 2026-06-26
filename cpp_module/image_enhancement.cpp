#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

class ImageEnhancerCPP {
public:
    // Fast and efficient image enhancement using C++
    static cv::Mat enhanceImage(const cv::Mat& inputImage) {
        cv::Mat gray, enhanced, denoised, sharpened;
        
        // Convert to grayscale
        cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
        
        // Apply CLAHE
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, enhanced);
        
        // Fast denoising
        cv::fastNlMeansDenoising(enhanced, denoised, 10);
        
        // Unsharp masking for sharpening
        cv::GaussianBlur(denoised, sharpened, cv::Size(3, 3), 1.0);
        cv::addWeighted(denoised, 1.5, sharpened, -0.5, 0, sharpened);
        
        return sharpened;
    }
    
    // Advanced contour detection for better document detection
    static std::vector<cv::Point> detectDocument(const cv::Mat& image) {
        cv::Mat gray, edges, dilated;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> bestContour;
        double maxArea = 0;
        
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        cv::Canny(gray, edges, 50, 150);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(edges, dilated, kernel);
        
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            std::vector<cv::Point> approx;
            double perimeter = cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
            
            if (approx.size() == 4 && area > image.size().area() * 0.05 && area > maxArea) {
                maxArea = area;
                bestContour = approx;
            }
        }
        
        return bestContour;
    }
};

// Python binding using pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(image_enhancement_cpp, m) {
    m.doc() = "C++ image enhancement module for Smart Document Scanner";
    
    m.def("enhance_image", [](py::array_t<uint8_t>& input) {
        // Convert numpy array to OpenCV Mat
        py::buffer_info buf = input.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
        
        cv::Mat result = ImageEnhancerCPP::enhanceImage(mat);
        
        // Convert back to numpy array
        return py::array_t<uint8_t>(
            {result.rows, result.cols, 1},
            {result.step[0], result.step[1], 1},
            result.data
        );
    }, "Enhance image using C++ optimized algorithms");
    
    m.def("detect_document", [](py::array_t<uint8_t>& input) {
        py::buffer_info buf = input.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
        
        std::vector<cv::Point> points = ImageEnhancerCPP::detectDocument(mat);
        
        // Convert points to Python list of tuples
        py::list result;
        for (const auto& pt : points) {
            result.append(py::make_tuple(pt.x, pt.y));
        }
        return result;
    }, "Detect document corners using C++");
}
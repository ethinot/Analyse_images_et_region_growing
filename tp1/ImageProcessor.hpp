#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

class ImageProcessor {
public:
    cv::Mat imageRgb;
    cv::Mat imageHsv;
    cv::Mat imageGray;

    ImageProcessor();
    ~ImageProcessor() = default;

    cv::Mat get_image_rgb() const;

    cv::Mat get_image_hsv() const;

    cv::Mat get_image_gray() const;

    void set_image_rgb(const cv::Mat &);

    void set_image_hsv(const cv::Mat &);

    void set_image_gray(const cv::Mat &);

    void process_image(const char* );
};

ImageProcessor::ImageProcessor() : imageRgb(cv::Mat()), imageHsv(cv::Mat()), imageGray(cv::Mat()) { }

cv::Mat ImageProcessor::get_image_rgb() const {
    return imageRgb;
}

cv::Mat ImageProcessor::get_image_hsv() const {
    return imageHsv;
}

cv::Mat ImageProcessor::get_image_gray() const {
    return imageGray;
}

void ImageProcessor::set_image_rgb(const cv::Mat &image) {
    imageRgb = image;
}

void ImageProcessor::set_image_hsv(const cv::Mat &image) {
    imageHsv = image;
}

void ImageProcessor::set_image_gray(const cv::Mat &image) {
    imageGray = image;
}

void ImageProcessor::process_image(const char* imagePath) {
    imageRgb = cv::imread(imagePath);
    if (!imageRgb.data) {
        printf("No image data\n");
        return;
    }

    cv::cvtColor(imageRgb, imageHsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(imageRgb, imageGray, cv::COLOR_BGR2GRAY);
}

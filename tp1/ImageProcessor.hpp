#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cmath>

class ImageProcessor {
public:
    cv::Mat originalImage;
    cv::Mat imageRgb;
    cv::Mat imageHsv;
    cv::Mat imageGray;

    ImageProcessor();
    ~ImageProcessor() = default;

    cv::Mat get_image_original() const;

    cv::Mat get_image_rgb() const;

    cv::Mat get_image_hsv() const;

    cv::Mat get_image_gray() const;

    void set_image_original(const cv::Mat &);

    void set_image_rgb(const cv::Mat &);

    void set_image_hsv(const cv::Mat &);

    void set_image_gray(const cv::Mat &);

    void process_image(const char* );

    cv::Mat get_part_of_image(const cv::Point &, const cv::Point &) const;

    void filter_image_noise(int kernelSize);

};

ImageProcessor::ImageProcessor() : originalImage(cv::Mat()), imageRgb(cv::Mat()), imageHsv(cv::Mat()), imageGray(cv::Mat()) { }

cv::Mat ImageProcessor::get_image_original() const {
    return originalImage;
}

cv::Mat ImageProcessor::get_image_rgb() const {
    return imageRgb;
}

cv::Mat ImageProcessor::get_image_hsv() const {
    return imageHsv;
}

cv::Mat ImageProcessor::get_image_gray() const {
    return imageGray;
}

void ImageProcessor::set_image_original(const cv::Mat &image) {
    originalImage = image;
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

// For the Kernel Size : 3 or 5 is a good values
void ImageProcessor::filter_image_noise(int kernelSize) {
    /* Todo : Teste other filter
     *  cv::medianBlur(image, imageTmp, 3);
     *  cv::bilateralFilter(imageTmp, imageFiltree, -1, 40, .?.)
     */
    cv::GaussianBlur(imageRgb, imageRgb, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::GaussianBlur(imageHsv, imageHsv, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::GaussianBlur(imageGray, imageGray, cv::Size(kernelSize, kernelSize), 0, 0);
}

void ImageProcessor::process_image(const char* imagePath) {
    originalImage = cv::imread(imagePath);
    if (!originalImage.data) {
        printf("No image data\n");
        return;
    }

    set_image_rgb(originalImage.clone());
    cv::cvtColor(originalImage, imageHsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(originalImage, imageGray, cv::COLOR_BGR2GRAY);

    filter_image_noise(5);
}

cv::Mat ImageProcessor::get_part_of_image(const cv::Point &top_left, const cv::Point &bottom_right) const {
   return imageRgb(cv::Rect(top_left, bottom_right));
}

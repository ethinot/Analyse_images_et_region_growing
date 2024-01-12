#pragma once

#include "GermsPositioning.hpp"

#include <list>


class ImageUtil {
public:
    void framing(unsigned int, unsigned int, int&, int&, int&, int&);

    double calculate_channel_variance(const cv::Mat &, int);

    double get_hsv_variance(const cv::Mat &);

    double get_grayscale_variance(const cv::Mat &);

    double calculate_variance(const cv::Mat &, const bool &);

    double calculate_region_variance(const cv::Mat &, cv::Point &, cv::Point &);

    float pixel_surface(cv::Point, cv::Point) const;

    cv::Point calculate_middle_point(const cv::Point&, const cv::Point&);


private:
    cv::Scalar calculate_mean_intensity(const cv::Mat &channel);

    cv::Mat get_channel_diff_from_mean_intensity(const cv::Mat &channel, const cv::Scalar &mean_intensity);
};

// Implementation :

cv::Scalar ImageUtil::calculate_mean_intensity(const cv::Mat &channel) {
    return cv::mean(channel);
}

cv::Mat ImageUtil::get_channel_diff_from_mean_intensity(const cv::Mat &channel, const cv::Scalar &mean_intensity) {
    cv::Mat diff;
    cv::absdiff(channel, mean_intensity, diff);
    diff = diff.mul(diff);
    return diff;
}

void ImageUtil::framing(unsigned int imWidth, unsigned int imHeight, int& numCaseW, int& numCaseH, int& caseWidth, int& caseHeight)
{
    numCaseW = (int)(std::log2f((float)imWidth));
    numCaseH = (int)(std::log2f((float)imHeight));
    caseWidth = imWidth / numCaseW;
    caseHeight = imHeight / numCaseH;
}

double ImageUtil::calculate_channel_variance(const cv::Mat & hsvImage, int channelIndex) {
    if (channelIndex < 0 || channelIndex >= hsvImage.channels()) {
        std::cerr << "Index de canal invalide." << std::endl;
        return -1.0;
    }

    cv::Mat channel;
    cv::extractChannel(hsvImage, channel, channelIndex);

    cv::Scalar meanIntensity = cv::mean(channel);
    cv::Mat diff;

    cv::absdiff(channel, meanIntensity, diff);
    diff = diff.mul(diff);
    cv::Scalar variance = cv::mean(diff);

    return variance.val[0];
}

double ImageUtil::get_hsv_variance(const cv::Mat &imageRgb) {
    cv::Mat hsvImage;
    cv::cvtColor(imageRgb, hsvImage, cv::COLOR_BGR2HSV);
    double varianceHue = calculate_channel_variance(hsvImage, 0);
    double varianceSaturation = calculate_channel_variance(hsvImage, 1);
    double varianceValue = calculate_channel_variance(hsvImage, 2);
    return (varianceHue + varianceSaturation + varianceValue) / 3.0;
}

double ImageUtil::get_grayscale_variance(const cv::Mat &image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Scalar meanIntensity = calculate_mean_intensity(grayImage);
    cv::Mat diff = get_channel_diff_from_mean_intensity(grayImage, meanIntensity);
    cv::Scalar variance = cv::mean(diff);

    return variance.val[0];
}

double ImageUtil::calculate_variance(const cv::Mat &image, const bool &isHsv) {
    return (isHsv) ? get_hsv_variance(image) : get_grayscale_variance(image);
}

/**
 * @brief Calculates the variance of the grayscale region in an image.
 *
 * The function calculates the variance of the grayscale region specified by the top-left and
 * bottom-right points in the given image. If the region points are invalid (i.e., outside the
 * image boundaries), an error message is printed to the standard error stream and -1.0 is returned.
 *
 * @param image     The input image.
 * @param topLeft   The top-left point of the region.
 * @param bottomRight   The bottom-right point of the region.
 * @return The variance of the grayscale region. If the region points are invalid, -1.0 is returned.
 */
double ImageUtil::calculate_region_variance(const cv::Mat& image, cv::Point & topLeft, cv::Point & bottomRight) {
    if (topLeft.x < 0 || topLeft.y < 0 || bottomRight.x > image.cols || bottomRight.y > image.rows) {

        std::cerr << "Points de région invalides. TopLeft: " << topLeft << ", BottomRight: " << bottomRight << "\n Car :" << image.cols << "--" << image.rows << std::endl;
        return -1.0;
    }

    cv::Mat roi = image(cv::Rect(topLeft, bottomRight));

    return calculate_variance(roi, 1);
}

float ImageUtil::pixel_surface(cv::Point point1, cv::Point point2) const {
    float dX = std::abs(point1.x - point2.x);
    float dY = std::abs(point1.y - point2.y);

    return dX * dY;
}

cv::Point ImageUtil::calculate_middle_point(const cv::Point& point1, const cv::Point& point2) {
    return cv::Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2);
}

class GermsDisplay {
public:
    void draw_framing(cv::Mat &, int, cv::Scalar);
    void display_germs(cv::Mat const &, cv::Mat &, std::vector<cv::Point> const &);
    void display_segmented_regions(cv::Mat const &, cv::Mat const &, const std::list<SegmentedRegion> &, cv::Scalar);
};

// Implementation :

void GermsDisplay::draw_framing(cv::Mat & image, int thickness = 2, cv::Scalar color = cv::Scalar(0, 0, 0)) {
    unsigned int rows = image.rows;
    unsigned int cols = image.cols;

    int numCols, numRows, caseWidth, caseHeight;
    ImageUtil imageUtil;
    imageUtil.framing(cols, rows, numCols, numRows, caseWidth, caseHeight);

    cv::Point start(0, 0);
    cv::Point end(cols, 0);
    for (int r = 0; r < numRows; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.y += caseHeight;
        end.y += caseHeight;
    }

    start = cv::Point(0, 0);
    end = cv::Point(0, rows);
    for (int r = 0; r < numCols; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.x += caseWidth;
        end.x += caseWidth;
    }
}

void GermsDisplay::display_germs(cv::Mat const & src, cv::Mat & dst, std::vector<cv::Point> const & germs) {
    dst = src.clone();
    for(auto& germ : germs) {
        cv::Point center(germ.x, germ.y); // (col,row)
        int radius = 10;
        cv::Scalar line_color(0,0,255);
        int thickness = 1;
        cv::circle(dst, center, radius, line_color, thickness);
    }
}

void GermsDisplay::display_segmented_regions(cv::Mat const & src, cv::Mat const & dst, const std::list<SegmentedRegion> & segmentedRegions, cv::Scalar color) {
    for (const auto& region : segmentedRegions) {
        cv::rectangle(dst, region.getTopLeftPoint(), region.getBottomRightPoint(), color);
    }
}
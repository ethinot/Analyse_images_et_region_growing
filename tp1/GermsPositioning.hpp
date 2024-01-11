#pragma once

#include "SegmentedRegion.hpp"
#include "ImageUtil.hpp"

#include "opencv2/imgproc.hpp"

#include <iostream>
#include <list>
#include <vector>
#include <random>


class GermsPositioningV1 {
private:
    ImageUtil imageUtil;

public:
    void generate_germ(std::vector<std::pair<int, int>> &, unsigned int, unsigned int, unsigned int);

    std::pair<int, int> rand_germ_position(int, int, int, int);
};

// GermsPositioningV1 implementations :

void GermsPositioningV1::generate_germ(std::vector<std::pair<int, int>> & buffer, unsigned int width, unsigned int height, unsigned int numOfGerm = 10) {
    int numCaseW, numCaseH, caseWidth, caseHeight;
    imageUtil.framing(width, height, numCaseW, numCaseH, caseWidth, caseHeight);
    for(unsigned int i = 0; i < numOfGerm; ++i)
    buffer.push_back(rand_germ_position(numCaseW, numCaseH, caseWidth, caseHeight));
}

std::pair<int, int> GermsPositioningV1::rand_germ_position(int numCaseW, int numCaseH, int caseWidth, int caseHeight) {
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distribNCaseW(0, numCaseW - 1);
    std::uniform_int_distribution<> distribNCaseH(0, numCaseH - 1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    std::uniform_int_distribution<> distribPosX(caseWidth * i, caseWidth * i + caseWidth);
    std::uniform_int_distribution<> distribPosY(caseHeight * j, caseHeight * j + caseHeight);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return std::pair<int, int>(py, px); // (row,col)
}

class GermsPositioningV2 {
private:
    std::list<SegmentedRegion> germsRegions;
    ImageUtil imageUtil;

public:
    const std::list<SegmentedRegion>& get_germs_regions() const;

    void set_germs_regions(const std::list<SegmentedRegion> &);

    void add_germ(cv::Point &topLeft, cv::Point &bottomRight, double variance);

    void delete_germ(const std::list<SegmentedRegion>::iterator &);

    void divide_image(const cv::Mat &, cv::Point &, cv::Point &, int);

    void process_high_variance_region(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight, int iterationLimit, int &iterationCounter);
};

const std::list<SegmentedRegion>& GermsPositioningV2::get_germs_regions() const {
    return germsRegions;
}

void GermsPositioningV2::add_germ(cv::Point &topLeft, cv::Point &bottomRight, double variance) {
    germsRegions.push_back(SegmentedRegion(topLeft, bottomRight, variance));
}

void GermsPositioningV2::delete_germ(const std::list<SegmentedRegion>::iterator & it) {
    germsRegions.erase(it);
}

void GermsPositioningV2::process_high_variance_region(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight,
                                                      int iterationLimit, int &iterationCounter) {
    int midX = (topLeft.x + bottomRight.x) / 2;
    int midY = (topLeft.y + bottomRight.y) / 2;
    ++iterationCounter;
    cv::Point mid = cv::Point(midX, midY);
    cv::Point midTop = cv::Point(midX, topLeft.y);
    cv::Point midRight = cv::Point(bottomRight.x, midY);
    cv::Point leftMid = cv::Point(topLeft.x, midY);
    cv::Point midBottom = cv::Point(midX, bottomRight.y);
    divide_image(image, topLeft, mid, iterationLimit);
    divide_image(image, midTop, midRight, iterationLimit);
    divide_image(image, leftMid, midBottom, iterationLimit);
    divide_image(image, mid, bottomRight, iterationLimit);
    --iterationCounter;
}

void GermsPositioningV2::divide_image(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight, int iterationLimit) {
    static int iterationCounter = 0;
    double variance = imageUtil.calculate_region_variance(image, topLeft, bottomRight);

    if (iterationCounter < iterationLimit && variance > 50.0) {
        if (topLeft.x < bottomRight.x && topLeft.y < bottomRight.y) {
            process_high_variance_region(image, topLeft, bottomRight, iterationLimit, iterationCounter);
        } else {
            add_germ(topLeft, bottomRight, variance);
        }
    } else {
        add_germ(topLeft, bottomRight, variance);
    }
}

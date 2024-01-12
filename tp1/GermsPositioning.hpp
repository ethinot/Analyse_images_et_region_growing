#pragma once

#include "SegmentedRegion.hpp"
#include "ImageUtil.hpp"

#include "opencv2/imgproc.hpp"

#include <iostream>
#include "ostream"
#include <list>
#include <vector>
#include <random>

class GermsPositioningV1 {
private:
    ImageUtil imageUtil;

public:
    cv::Point rand_germ_position(int, int, int, int);

    void generate_seed(std::vector<cv::Point>&, uint32_t, uint32_t, uint32_t);
};

// GermsPositioningV1 implementations :

cv::Point GermsPositioningV1::rand_germ_position(int numCaseW, int numCaseH, int caseWidth, int caseHeight) {
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distribNCaseW(0, numCaseW - 1);
    std::uniform_int_distribution<> distribNCaseH(0, numCaseH - 1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    std::uniform_int_distribution<> distribPosX(caseWidth * i, caseWidth * i + caseWidth);
    std::uniform_int_distribution<> distribPosY(caseHeight * j, caseHeight * j + caseHeight);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return {px,py}; // (row,col)
}

void GermsPositioningV1::generate_seed(std::vector<cv::Point>& seeds, uint32_t w, uint32_t h, uint32_t numSeeds=10) {
    int numC, numR, caseW, caseH;
    imageUtil.framing(w, h, numC, numR, caseW, caseH);
    for(uint32_t i = 0; i < numSeeds; ++i) {
        cv::Point seed;
        seed = rand_germ_position(numC, numR, caseW, caseH);
        seeds.push_back(seed);
    }
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

    bool variance_criterion(const double & variance, const double limit) const;
    bool iteration_criterion(const int iterationLimit, const int iterationCounter) const;
    bool surface_criterion(const cv::Point & topLeft, const cv::Point & bottomRight, const float limit) const;

    bool separation_criterion(const double &, const int, const int, const cv::Point &, const cv::Point &) const;

    void divide_image(const cv::Mat &, cv::Point &, cv::Point &, int);

    void process_high_variance_region(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight, int iterationLimit, int &iterationCounter);

    void add_region_germ(std::vector<cv::Point> &);

    void position_germs(cv::Mat&, int, std::vector<cv::Point> &);

    friend std::ostream& operator<<(std::ostream&, const GermsPositioningV2&);
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

bool GermsPositioningV2::variance_criterion(const double & variance, const double limit) const {
    return variance >= limit;
}

bool GermsPositioningV2::iteration_criterion(const int iterationLimit, const int iterationCounter) const {
    return (iterationCounter <= iterationLimit);
}

bool GermsPositioningV2::surface_criterion(const cv::Point & topLeft, const cv::Point & bottomRight, const float limit) const {
    float surface = imageUtil.pixel_surface(topLeft, bottomRight);
    return surface >= limit;
}

// Return true if the separation is possible
bool GermsPositioningV2::separation_criterion(const double & variance, const int iterationLimit, const int iterationCounter,
                                              const cv::Point & topLeft, const cv::Point & bottomRight) const {
    bool isVariance = variance_criterion(variance, 40.0);
    bool isIteration = iteration_criterion(iterationLimit, iterationCounter);
    bool isSurface = surface_criterion(topLeft, bottomRight, 30);
    return isVariance && isIteration && isSurface;
}

void GermsPositioningV2::process_high_variance_region(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight,
                                                      int iterationLimit, int &iterationCounter) {

    int midX = (topLeft.x + bottomRight.x) / 2;
    int midY = (topLeft.y + bottomRight.y) / 2;

    cv::Point mid = cv::Point(midX, midY);
    cv::Point midTop = cv::Point(midX, topLeft.y);
    cv::Point midRight = cv::Point(bottomRight.x, midY);
    cv::Point leftMid = cv::Point(topLeft.x, midY);
    cv::Point midBottom = cv::Point(midX, bottomRight.y);

    ++iterationCounter;

    if(iteration_criterion(iterationLimit, iterationCounter)) {
        divide_image(image, topLeft, mid, iterationLimit);
        divide_image(image, midTop, midRight, iterationLimit);
        divide_image(image, leftMid, midBottom, iterationLimit);
        divide_image(image, mid, bottomRight, iterationLimit);
    } else {
        add_germ(topLeft, bottomRight, -1);
    }

    --iterationCounter;

}

void GermsPositioningV2::divide_image(const cv::Mat &image, cv::Point &topLeft, cv::Point &bottomRight, int iterationLimit) {
    static int iterationCounter = 0;
    double variance = imageUtil.calculate_region_variance(image, topLeft, bottomRight);

    //std::cout << "Iteration - "<< iterationCounter << "\n -- Try to split : "<< topLeft << "," << bottomRight << std::endl;
    //std::cout << "  Variance --> "<< variance << std::endl;
    //std::cout << "  Surface --> "<< imageUtil.pixel_surface(topLeft, bottomRight) << std::endl;


    bool criterion = separation_criterion(variance, iterationLimit, iterationCounter, topLeft, bottomRight);
    //std::cout << "  Can split ? --> " << (criterion ? " yes ! \n" : " no \n");

    if (criterion) {
        if (topLeft.x < bottomRight.x && topLeft.y < bottomRight.y) {
            process_high_variance_region(image, topLeft, bottomRight, iterationLimit, iterationCounter);
        } else {
            std::cerr << "Top left and bottom right pixels are not respecting the condition: topLeft.x < bottomRight.x and topLeft.y < bottomRight.y\n";
            return;
        }
    } else {
        add_germ(topLeft, bottomRight, variance);
    }
}

void GermsPositioningV2::add_region_germ(std::vector<cv::Point> & seeds) {
    for (const auto& germ : get_germs_regions()) {
        seeds.push_back(imageUtil.calculate_middle_point(germ.getTopLeftPoint(), germ.getBottomRightPoint()));
    }
}

void GermsPositioningV2::position_germs(cv::Mat& image, int maxDivision, std::vector<cv::Point> & seeds) {
    cv::Point initialTopLeft(0, 0);
    cv::Point initialBottomRight(image.cols, image.rows);

    divide_image(image, initialTopLeft, initialBottomRight, maxDivision);

    add_region_germ(seeds);
}

std::ostream& operator<<(std::ostream& os, const GermsPositioningV2& gpv2)
{
    for (const auto& germ : gpv2.get_germs_regions()) {
        os << "\n -------------------------------- \n";
        os << germ;
        os << "\n -------------------------------- \n";

    }
    return os;
}
#pragma once

#include "opencv2/imgproc.hpp"

#include <iostream>

class SegmentedRegion {
private:
    cv::Point topLeft;
    cv::Point bottomRight;
    double variance;

public:
    SegmentedRegion();

    SegmentedRegion(cv::Point &, cv::Point &, double &);
    
    ~SegmentedRegion();

    const double getVariance() const;
    void setVariance(double &);

    const cv::Point getTopLeftPoint() const;
    void setTopLeftPoint(cv::Point &);

    const cv::Point getBottomRightPoint() const;
    void setBottomRightPoint(cv::Point &);

};

SegmentedRegion::SegmentedRegion() : topLeft(0, 0), bottomRight(0, 0), variance(0.0) { }

SegmentedRegion::SegmentedRegion(cv::Point & _topLeft, cv::Point & _bottomRight, double & _variance) : topLeft(_topLeft), bottomRight(_bottomRight), variance(_variance) { }

SegmentedRegion::~SegmentedRegion() { }

const double SegmentedRegion::getVariance() const {
    return this->variance;
}

void SegmentedRegion::setVariance(double & newVariance) {
    this->variance = newVariance;
}

const cv::Point SegmentedRegion::getTopLeftPoint() const {
    return this->topLeft;
}
void SegmentedRegion::setTopLeftPoint(cv::Point & newTopLeft) {
    this->topLeft = newTopLeft;
}

const cv::Point SegmentedRegion::getBottomRightPoint() const {
    return this->bottomRight;

}

void SegmentedRegion::setBottomRightPoint(cv::Point & newBottomRight) {
    this->bottomRight = newBottomRight;
}
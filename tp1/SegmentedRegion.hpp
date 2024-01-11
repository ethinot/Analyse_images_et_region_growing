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

    SegmentedRegion(cv::Point &, cv::Point &, double);
    
    ~SegmentedRegion();

    friend std::ostream& operator<<(std::ostream&, const SegmentedRegion&);

    const double getVariance() const;
    void setVariance(double &);

    const cv::Point getTopLeftPoint() const;
    void setTopLeftPoint(cv::Point &);

    const cv::Point getBottomRightPoint() const;
    void setBottomRightPoint(cv::Point &);

};

SegmentedRegion::SegmentedRegion() : topLeft(0, 0), bottomRight(0, 0), variance(0.0) { }

SegmentedRegion::SegmentedRegion(cv::Point & _topLeft, cv::Point & _bottomRight, double _variance) : topLeft(_topLeft), bottomRight(_bottomRight), variance(_variance) { }

SegmentedRegion::~SegmentedRegion() { }

std::ostream& operator<<(std::ostream& os, const SegmentedRegion& sr) {
    os << "Segmented Region Info:\n"
       << "Top Left Point : [" << sr.topLeft.x << ", " << sr.topLeft.y << "]\n"
       << "Bottom Right Point : [" << sr.bottomRight.x << ", " << sr.bottomRight.y << "]\n"
       << "Variance : " << sr.variance;
    return os;
}

const double SegmentedRegion::getVariance() const {
    return variance;
}

void SegmentedRegion::setVariance(double & newVariance) {
    variance = newVariance;
}

const cv::Point SegmentedRegion::getTopLeftPoint() const {
    return topLeft;
}

void SegmentedRegion::setTopLeftPoint(cv::Point & newTopLeft) {
    topLeft = newTopLeft;
}

const cv::Point SegmentedRegion::getBottomRightPoint() const {
    return bottomRight;
}

void SegmentedRegion::setBottomRightPoint(cv::Point & newBottomRight) {
    bottomRight = newBottomRight;
}
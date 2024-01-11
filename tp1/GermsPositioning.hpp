#pragma once

#include "opencv2/imgproc.hpp"
#include "SegmentedRegion.hpp"
#include <iostream>
#include <list>
#include <vector>

class GermsPositioningV1 {
public:
    void frame(unsigned int imWidth, unsigned int imHeight, int &numCaseW, int &numCaseH, int &caseWidth, int &caseHeight);

    void generateGerm(std::vector<std::pair<int, int>> &, unsigned int, unsigned int, unsigned int);

    std::pair<int, int> rand_germ_position(int, int, int, int);
};

class GermsPositioningV2 {
private:
    std::list<SegmentedRegion> germsRegions;

public:
    const std::list<SegmentedRegion>& get_germs_regions() const;
    void set_germs_regions(const std::list<SegmentedRegion> &);
    void divide_image(const cv::Mat &, cv::Point &, cv::Point &, int);
};
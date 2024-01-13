#pragma once

#include <random>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
#include<queue>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

std::mt19937 generator{ std::random_device{}() };

class GrowAndMerge {
private:

    using region_type = std::pair<std::list<cv::Point>, std::vector<cv::Scalar>>;
    using region_container = std::unordered_map<int, region_type>;

    region_container regions;

    bool randomColorization = false;

    int numSeeds = 10;

    // O(1)
    int bgr_to_hex(cv::Vec3b const&);

    cv::Vec3b hex_to_bgr(int);

    // O(1)
    std::pair<cv::Scalar, cv::Scalar> interval_bounds(cv::Scalar const&);

    // O(1)
    bool predicate(cv::Scalar const&, cv::Scalar const&, cv::Scalar const&);

    // O(1)
    void update_mean(region_type &, cv::Scalar const&);

    // O(1)
    cv::Scalar componentwise_min(cv::Scalar const&, cv::Scalar const&);

    cv::Scalar componentwise_max(cv::Scalar const&, cv::Scalar const&);

    void update_buffer(cv::Mat &, std::list<cv::Point> const&, int);

    // O(min(size1, size2))
    void merge(region_container &, cv::Mat &, int &, int &);

    // O(1)
    void process(region_container &, std::vector<cv::Mat> const&, cv::Mat &, std::queue<cv::Point> &, cv::Point const&, int &);

    void growing(region_container &, std::vector<cv::Mat> const&, cv::Mat &, cv::Point const&, int);

    std::vector<cv::Vec3b> generate_random_unique_BGR(size_t);

    uchar check_bounds(uchar);

    std::vector<int> generate_unique_BGR(cv::Mat const&, std::vector<cv::Point> const&);

    void fill_mask(cv::Mat const&, cv::Mat &);

    bool is_edge(cv::Mat const&, cv::Point const&);

    void edge_mask(cv::Mat const&, cv::Mat &);

    double coverage(region_container const&, uint32_t, uint32_t);

    void seg(cv::Mat const&, cv::Mat &, std::vector<cv::Point> const&, region_container &);

public:
    const region_container& get_regions() const;

    void set_regions(const region_container&);

    int get_num_seeds() const; // getter method

    void set_num_seeds(int); // setter method

    void rg_seg(cv::Mat const&, cv::Mat &, std::vector<cv::Point> &, bool onlyEdge=false);
};

const GrowAndMerge::region_container& GrowAndMerge::get_regions() const {
    return regions;
}

void GrowAndMerge::set_regions(const GrowAndMerge::region_container& regions) {
    this->regions = regions;
}

int GrowAndMerge::get_num_seeds() const {
    return numSeeds;
}

void GrowAndMerge::set_num_seeds(int seeds) {
    numSeeds = seeds;
}

int GrowAndMerge::bgr_to_hex(cv::Vec3b const& bgr) {
    return (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
}

cv::Vec3b GrowAndMerge::hex_to_bgr(int hexValue) {
    uchar blue = hexValue & 0xFF;
    uchar green = (hexValue >> 8) & 0xFF;
    uchar red = (hexValue >> 16) & 0xFF;

    return {blue, green, red};
}

std::pair<cv::Scalar, cv::Scalar> GrowAndMerge::interval_bounds(cv::Scalar const& hsv) {
    double th;
    if (hsv[2] <= 25) { // Black
        th = 30;
        double lowerv = ((hsv[2] - th) < 0) ? 0 : hsv[2] - th;
        double upperv = hsv[2] + th;
        return std::make_pair(cv::Scalar(0, 0, lowerv), cv::Scalar(180, 255, upperv));
    } else if (hsv[1] <= 70) {
        if (hsv[2] <= 175) { // Gray
            th = 40;
            double lowerv = hsv[2] - th;
            double upperv = hsv[2] + th;
            return std::make_pair(cv::Scalar(0, 0, lowerv), cv::Scalar(180, 45, upperv));
        } else { // White
            th = 40;
            double lowerv = hsv[2] - th;
            double upperv = ((hsv[2] + th) > 255) ? 255 : hsv[2] + th;
            return std::make_pair(cv::Scalar(0, 0, lowerv), cv::Scalar(180, 70, upperv));
        }
    }
    th = 10;
    double lowerh = (hsv[0] - th < 0) ? 0 : hsv[0] - th;
    double upperh = (hsv[0] + th > 180) ? 180 : hsv[0] + th;
    return std::make_pair(cv::Scalar(lowerh, 70, 50), cv::Scalar(upperh, 255, 255));
}

bool GrowAndMerge::predicate(cv::Scalar const& lowerb, cv::Scalar const& upperb, cv::Scalar const& value) {
    // lowerb <= value <= upperb
    return  value[0] >= lowerb[0] && value[0] <= upperb[0] &&
            value[1] >= lowerb[1] && value[1] <= upperb[1] &&
            value[2] >= lowerb[2] && value[2] <= upperb[2];
}

void GrowAndMerge::update_mean(region_type & region, cv::Scalar const& addedValue) {
    cv::Scalar oldMean = region.second[2];
    int size = (int)region.first.size();
    cv::Scalar newMean = (oldMean * size + addedValue) / (size + 1);
    region.second[2] = newMean;
}

cv::Scalar GrowAndMerge::componentwise_min(cv::Scalar const& sa, cv::Scalar const& sb) {
    cv::Scalar ret;
    ret[0] = std::min(sa[0], sb[0]);
    ret[1] = std::min(sa[1], sb[1]);
    ret[2] = std::min(sa[2], sb[2]);
    return ret;
}

cv::Scalar GrowAndMerge::componentwise_max(cv::Scalar const& sa, cv::Scalar const& sb) {
    cv::Scalar ret;
    ret[0] = std::max(sa[0], sb[0]);
    ret[1] = std::max(sa[1], sb[1]);
    ret[2] = std::max(sa[2], sb[2]);
    return ret;
}

void GrowAndMerge::update_buffer(cv::Mat & buffer, std::list<cv::Point> const& points, int newValue) {
    for (auto const& point : points) {

        buffer.at<int>(point) = newValue;
    }
}

void GrowAndMerge::merge(region_container & regions, cv::Mat & buffer, int & r1Key, int & r2Key) {
    std::vector<cv::Scalar> hsv1 = regions[r1Key].second;
    std::vector<cv::Scalar> hsv2 = regions[r2Key].second;
    std::vector<cv::Scalar> hsv(3, 0);
    hsv[0] = componentwise_min(hsv1[0], hsv2[0]);
    hsv[1] = componentwise_max(hsv1[1], hsv2[1]);

    int size1 = (int)regions[r1Key].first.size();
    int size2 = (int)regions[r2Key].first.size();
    hsv[2] = (hsv1[2]*size1 + hsv2[2]*size2) / (size1 + size2);

    if (size1 < size2) {
        update_buffer(buffer, regions[r1Key].first, r2Key);
        regions[r2Key].first.splice(
                regions[r2Key].first.end(), regions[r1Key].first);
        regions.erase(regions.find(r1Key));
        regions[r2Key].second = hsv;
        r1Key = r2Key;
    } else { // R2 is smaller than R1
        update_buffer(buffer, regions[r2Key].first, r1Key);
        regions[r1Key].first.splice(
                regions[r1Key].first.end(), regions[r2Key].first);
        regions.erase(regions.find(r2Key));
        regions[r1Key].second = hsv;
        r2Key = r1Key;
    }
}

void GrowAndMerge::process(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, std::queue<cv::Point> & queue, cv::Point const& current, int & currentKey) {
    cv::Scalar lowerb = regions[currentKey].second[0];
    cv::Scalar upperb = regions[currentKey].second[1];

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i != 0 || j != 0) {
                cv::Point neighbor(current.x + i, current.y + j);
                if (neighbor.x >= 0 && neighbor.x < hsvChannels[0].cols &&
                    neighbor.y >= 0 && neighbor.y < hsvChannels[0].rows) {
                    int neighborKey = buffer.at<int>(neighbor);
                    if (neighborKey == 0) {
                        auto neighborH = (double) hsvChannels[0].at<uchar>(neighbor);
                        auto neighborS = (double) hsvChannels[1].at<uchar>(neighbor);
                        auto neighborV = (double) hsvChannels[2].at<uchar>(neighbor);
                        cv::Scalar hsvNeighbor(neighborH, neighborS, neighborV);

                        if (predicate(lowerb, upperb, hsvNeighbor)) {
                            buffer.at<int>(neighbor) = currentKey;
                            update_mean(regions[currentKey], hsvNeighbor);
                            regions[currentKey].first.push_back(neighbor);
                            queue.push(neighbor);
                        }
                    } else if (currentKey != neighborKey) {
                        std::vector<cv::Scalar> hsvNeighborRegion =
                                regions[neighborKey].second;
                        cv::Scalar currentRegionHsvMean =
                                regions[currentKey].second[2];
                        if (predicate(lowerb, upperb, hsvNeighborRegion[2]) &&
                            predicate(hsvNeighborRegion[0], hsvNeighborRegion[1], currentRegionHsvMean)) {
                            merge(regions, buffer, currentKey, neighborKey);
                        } else {
                            std::cout << "";
                        }
                    }
                }
            }
        }
    }
}

void GrowAndMerge::growing(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, cv::Point const& seed, int currentKey) {
    std::queue<cv::Point> queue;
    queue.push(seed);

    auto seedH = (double)hsvChannels[0].at<uchar>(seed);
    auto seedS = (double)hsvChannels[1].at<uchar>(seed);
    auto seedV = (double)hsvChannels[2].at<uchar>(seed);

    cv::Scalar hsvSeed(seedH, seedS, seedV);
    std::pair<cv::Scalar, cv::Scalar> bounds = interval_bounds(hsvSeed);
    regions[currentKey].second.emplace_back(bounds.first);
    regions[currentKey].second.emplace_back(bounds.second);
    regions[currentKey].second.emplace_back(0);
    update_mean(regions[currentKey], hsvSeed);

    regions[currentKey].first.push_back(seed);
    buffer.at<int>(seed) = currentKey;

    while (!queue.empty()) {
        cv::Point current = queue.front();
        queue.pop();
        process(regions, hsvChannels, buffer, queue, current, currentKey);
    }
}

std::vector<cv::Vec3b> GrowAndMerge::generate_random_unique_BGR(size_t size) {
    std::uniform_int_distribution<int> dis(55, 255);

    std::unordered_set<int> usedColors;
    std::vector<cv::Vec3b> randomColorList;

    for (long unsigned int i = 0; i < size; ++i) {
        int colorValue;
        cv::Vec3b color;
        do {
            // Generate random values for RGB
            color[0] = static_cast<uchar>(dis(generator));
            color[1] = static_cast<uchar>(dis(generator));
            color[2] = static_cast<uchar>(dis(generator));

            // Convert RGB to a single integer for uniqueness check
            colorValue = bgr_to_hex(color);
        } while (usedColors.count(colorValue) > 0);

        usedColors.insert(colorValue);
        randomColorList.push_back(color);
    }

    return randomColorList;
}

uchar GrowAndMerge::check_bounds(uchar value) {
    return (value + 10 > 255) ? ((value - 10 < 0) ? value + 10 : value - 10) : value - 10;
}

std::vector<int> GrowAndMerge::generate_unique_BGR(cv::Mat const& img, std::vector<cv::Point> const& seeds) {
    std::unordered_set<int> usedColors;
    std::vector<int> colorList;
    colorList.reserve(seeds.size());

    for (auto seed : seeds) {
        cv::Vec3b color = img.at<cv::Vec3b>(seed);
        int hexColor = bgr_to_hex(color);

        if (hexColor == 0) {
            uchar bValue = check_bounds(color[0]);
            uchar gValue = check_bounds(color[1]);
            uchar rValue = check_bounds(color[2]);
            color = {bValue, gValue, rValue};
            hexColor = bgr_to_hex(color);
        }

        int cpt = 0;
        while (usedColors.count(hexColor) > 0) {
            ++cpt;
            uchar bValue = check_bounds(color[0]);
            uchar gValue = check_bounds(color[1]);
            uchar rValue = check_bounds(color[2]);
            color = {bValue, gValue, rValue};
            hexColor = bgr_to_hex(color);
        }

        usedColors.insert(hexColor);
        colorList.push_back(hexColor);
    }

    return colorList;
}

void GrowAndMerge::fill_mask(cv::Mat const& buffer, cv::Mat & mask) {
    for (int i = 0; i < buffer.rows; ++i) {
        for (int j = 0; j < buffer.cols; ++j) {
            mask.at<cv::Vec3b>(i, j) = hex_to_bgr(buffer.at<int>(i, j));
        }
    }
}

bool GrowAndMerge::is_edge(cv::Mat const& buffer, cv::Point const& pixel) {
    if (pixel.x-1 >= 0 && buffer.at<int>(pixel.y, pixel.x-1) != buffer.at<int>(pixel)) {
        return true;
    }
    if (pixel.x+1 < buffer.cols && buffer.at<int>(pixel.y, pixel.x+1) != buffer.at<int>(pixel)) {
        return true;
    }
    if (pixel.y-1 >= 0 && buffer.at<int>(pixel.y-1, pixel.x) != buffer.at<int>(pixel)) {
        return true;    
    }
    if (pixel.y+1 < buffer.rows && buffer.at<int>(pixel.y+1, pixel.x) != buffer.at<int>(pixel)) {
        return true;
    }
    return false;
}

void GrowAndMerge::edge_mask(cv::Mat const& buffer, cv::Mat & mask) {
    for (int i = 0; i < buffer.rows; ++i) {
        for (int j = 0; j < buffer.cols; ++j) {
            cv::Point pixel(j, i);
            if (is_edge(buffer, pixel)) {
                mask.at<cv::Vec3b>(i, j) = hex_to_bgr(buffer.at<int>(i, j));
            }
        }
    }
}

double GrowAndMerge::coverage(region_container const& regions, uint32_t cols, uint32_t rows) {
    size_t count = 0;
    for (auto const& [key, value]: regions) {
        count += value.first.size();
    }

    return (double)count / ((double)cols*(double)rows);
}

void GrowAndMerge::seg(cv::Mat const& src, cv::Mat & dst, std::vector<cv::Point> const& seeds, region_container & regions) {
    cv::Mat hsvImg;
    cv::cvtColor(src, hsvImg, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImg, hsvChannels);

    size_t numSeeds = seeds.size();

    std::vector<cv::Vec3b> rdmColorList;
    std::vector<int> colorList;
    if (randomColorization) {
        rdmColorList = generate_random_unique_BGR(numSeeds);
    } else {
        colorList = generate_unique_BGR(src, seeds);
    }

    for (size_t i = 0; i < numSeeds; ++i) {
        if (dst.at<int>(seeds[i]) == 0) {
            if (randomColorization) {
                int currentKey = bgr_to_hex(rdmColorList[i]);
                growing(regions, hsvChannels, dst, seeds[i], currentKey);
            } else {
                growing(regions, hsvChannels, dst, seeds[i], colorList[i]);
            }
        }
    }
}

// Public method implementation :

void GrowAndMerge::rg_seg(cv::Mat const& src, cv::Mat & dst, std::vector<cv::Point> & seeds, bool onlyEdge) {
    cv::Mat buffer = cv::Mat::zeros(src.size(), CV_32S);

    seg(src, buffer, seeds, regions);

    if (onlyEdge) {
        edge_mask(buffer, dst);
    } else {
        fill_mask(buffer, dst);
    }

    std::cout << "Coverage percentage: " << coverage(regions, src.cols, src.rows) << "%" << std::endl;
}
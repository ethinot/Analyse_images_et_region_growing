#include <iostream>
#include <random>
#include <cstdlib> // strtol, rand, srand ... 
#include <vector>
#include<queue>
#include <chrono>
#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <list>

// opencv libs
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point stop;

std::chrono::microseconds duration;

#define MEASURE_TIME(func) \
        start = std::chrono::high_resolution_clock::now(); \
        func; \
        stop = std::chrono::high_resolution_clock::now(); \
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        std::cout << "Time taken by " << #func << ": " << (duration.count() / 1000.0) << "ms" << std::endl; \


std::mt19937 generator{ std::random_device{}() };

// --------------- Seed stuff -------------------

void framing(uint32_t imW, uint32_t imH, int& numC, int& numR, int& caseW, int& caseH)
{
    numC = (int)(std::log2f((float)imW));
    numR = (int)(std::log2f((float)imH));
    caseW = imW / numC;
    caseH = imH / numR;
}

cv::Point rand_seed_position(int numC, int numR, int caseW, int caseH)
{
    std::uniform_int_distribution<> distribNCaseW(0, numC-1);
    std::uniform_int_distribution<> distribNCaseH(0, numR-1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    std::uniform_int_distribution<> distribPosX(caseW*i,caseW*i+caseW-1);
    std::uniform_int_distribution<> distribPosY(caseH*j,caseH*j+caseH-1);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return {px,py}; // (col,row)
}

void generate_seed(std::vector<cv::Point>& seeds, uint32_t w, uint32_t h, uint32_t numSeeds=10)
{
    int numC, numR, caseW, caseH;
    framing(w, h, numC, numR, caseW, caseH);
    for(uint32_t i = 0; i < numSeeds; ++i) {
        cv::Point seed;
        seed = rand_seed_position(numC, numR, caseW, caseH);
        seeds.push_back(seed);
    }
}

void display_seeds(cv::Mat const& src, cv::Mat & dst, std::vector<cv::Point> const& seeds)
{
    dst = src.clone();
    for(auto& seed : seeds) {
        cv::Point center(seed); // (col,row)
        int radius = 0;
        cv::Scalar line_color(0,0,255);
        int thickness = 1;
        cv::circle(dst, center, radius, line_color, thickness);
    }
}

// --------------------------------------------------

// O(1)
int bgr_to_hex(cv::Vec3b const& bgr)
{
    return (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
}

cv::Vec3b hex_to_bgr(int hexValue) {
    uchar blue = hexValue & 0xFF;
    uchar green = (hexValue >> 8) & 0xFF;
    uchar red = (hexValue >> 16) & 0xFF;

    return {blue, green, red};
}

// O(1)
std::pair<cv::Scalar, cv::Scalar> interval_bounds(cv::Scalar const& hsv)
{
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

// O(1)
bool predicate(cv::Scalar const& lowerb, cv::Scalar const& upperb, cv::Scalar const& value)
{
    // lowerb <= value <= upperb
    return  value[0] >= lowerb[0] && value[0] <= upperb[0] &&
            value[1] >= lowerb[1] && value[1] <= upperb[1] &&
            value[2] >= lowerb[2] && value[2] <= upperb[2];
}

using region_type = std::pair<std::list<cv::Point>, std::vector<cv::Scalar>>;
using region_container = std::unordered_map<int, region_type>;

// O(1)
void update_mean(region_type & region, cv::Scalar const& addedValue)
{
    cv::Scalar oldMean = region.second[2];
    int size = (int)region.first.size();
    cv::Scalar newMean = (oldMean * size + addedValue) / (size + 1);
    region.second[2] = newMean;
}

// O(1)
cv::Scalar componentwise_min(cv::Scalar const& sa, cv::Scalar const& sb)
{
    cv::Scalar ret;
    ret[0] = std::min(sa[0], sb[0]);
    ret[1] = std::min(sa[1], sb[1]);
    ret[2] = std::min(sa[2], sb[2]);
    return ret;
}

// O(1)
cv::Scalar componentwise_max(cv::Scalar const& sa, cv::Scalar const& sb)
{
    cv::Scalar ret;
    ret[0] = std::max(sa[0], sb[0]);
    ret[1] = std::max(sa[1], sb[1]);
    ret[2] = std::max(sa[2], sb[2]);
    return ret;
}

void update_buffer(cv::Mat & buffer, std::list<cv::Point> const& points, int newValue)
{
    for (auto const& point : points) {

        buffer.at<int>(point) = newValue;
    }
}

// O(min(size1, size2))
void merge(region_container & regions, cv::Mat & buffer, int & r1Key, int & r2Key)
{
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

// O(1)
void process(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, std::queue<cv::Point> & queue, cv::Point const& current, int & currentKey)
{
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

// O(w) avec w nombre de pixels parcourus
void growing(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, cv::Point const& seed, int currentKey)
{
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

std::vector<cv::Vec3b> generate_random_unique_BGR(size_t size) {
    std::uniform_int_distribution<int> dis(55, 255);

    std::unordered_set<int> usedColors;
    std::vector<cv::Vec3b> randomColorList;

    for (int i = 0; i < size; ++i) {
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

uchar check_bounds(uchar value)
{
    return (value + 10 > 255) ? ((value - 10 < 0) ? value + 10 : value - 10) : value - 10;
}

std::vector<int> generate_unique_BGR(cv::Mat const& img, std::vector<cv::Point> const& seeds) {
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

void fill_mask(cv::Mat const& buffer, cv::Mat & mask)
{
    for (int i = 0; i < buffer.rows; ++i) {
        for (int j = 0; j < buffer.cols; ++j) {
            mask.at<cv::Vec3b>(i, j) = hex_to_bgr(buffer.at<int>(i, j));
        }
    }
}

double coverage(region_container const& regions, uint32_t cols, uint32_t rows)
{
    size_t count = 0;
    for (auto const& [key, value]: regions) {
        count += value.first.size();
    }

    return (double)count / ((double)cols*(double)rows);
}

bool randomColorization = false;

void seg(cv::Mat const& src, cv::Mat & dst, std::vector<cv::Point> const& seeds, region_container & regions)
{
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

int numSeeds = 10;

void rg_seg(cv::Mat const& src, cv::Mat & dst)
{
    std::vector<cv::Point> seeds;
    MEASURE_TIME(generate_seed(seeds, src.cols, src.rows, numSeeds));

    region_container regions;
    cv::Mat buffer = cv::Mat::zeros(src.size(), CV_32S);
    MEASURE_TIME(seg(src, buffer, seeds, regions));

    MEASURE_TIME(fill_mask(buffer, dst));

    std::cout << "Coverage percentage: " << coverage(regions, src.cols, src.rows) << "%" << std::endl;
}

int main(int argc, char** argv) {
//    auto start = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        printf("usage: DisplayImage.out <Image_Path> (<num of seeds>)\n");
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (!image.data) {
        printf("No image data \n");
        return -1;
    }

    if (argc == 3) {
        numSeeds = strtol(argv[2], nullptr, 10);
    } 

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);
    rg_seg(image, mask);
    

//    cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
    cv::imshow("Segmentation", mask);
    cv::waitKey(0);
    return 0;
}
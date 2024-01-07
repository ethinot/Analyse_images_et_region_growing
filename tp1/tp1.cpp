#include <iostream>
#include <random>
#include <cstdlib> // strtol, rand, srand ... 
#include <vector>
#include<queue>
#include <chrono>
#include <variant>
#include <unordered_map>
#include <unordered_set>

// opencv libs
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

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
cv::Scalar min(cv::Scalar const& sa, cv::Scalar const& sb)
{
    cv::Scalar ret;
    ret[0] = std::min(sa[0], sb[0]);
    ret[1] = std::min(sa[1], sb[1]);
    ret[2] = std::min(sa[2], sb[2]);
    return ret;
}

// O(1)
cv::Scalar max(cv::Scalar const& sa, cv::Scalar const& sb)
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
    hsv[0] = min(hsv1[0], hsv2[0]);
    hsv[1] = max(hsv1[1], hsv2[1]);

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
    } else {
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

/*
lower {0, 0, 105}
upper {180, 45, 200}
mean  {91.3, 9.18, 160.07}

lower {0, 0, 167}
upper {180, 70, 227}
mean  {130, 12, 197}
*/


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

        while (usedColors.count(hexColor) > 0) {
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

void fill_mask(cv::Mat & mask, region_container & regions)
{
    for (auto& [key, value] : regions) {
        cv::Vec3b color = hex_to_bgr(key);
        for (auto& coords : value.first) {
            mask.at<cv::Vec3b>(coords) = color;
        }
    }
}

double coverage(region_container const& regions, uint32_t cols, uint32_t rows)
{
    uint32_t count = 0;
    for (auto const& [key, value]: regions) {
        count += value.first.size();
    }

    return (double)count / (double)(cols*rows);
}

bool randomColorization = false;

void seg(cv::Mat const& src, cv::Mat & dst, std::vector<cv::Point> const& seeds)
{
    cv::Mat hsvImg;
    cv::cvtColor(src, hsvImg, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImg, hsvChannels);

    region_container regions;

    cv::Mat buffer = cv::Mat::zeros(src.size(), CV_32S);

    size_t numSeeds = seeds.size();

    std::vector<cv::Vec3b> rdmColorList;
    std::vector<int> colorList;
    if (randomColorization) {
        rdmColorList = generate_random_unique_BGR(numSeeds);
    } else {
        colorList = generate_unique_BGR(src, seeds);
    }

    for (size_t i = 0; i < numSeeds; ++i) {
        if (randomColorization) {
            int currentKey = bgr_to_hex(rdmColorList[i]);
            growing(regions, hsvChannels, buffer, seeds[i], currentKey);
        } else {
            growing(regions, hsvChannels, buffer, seeds[i], colorList[i]);
        }
    }

    fill_mask(dst, regions);
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

    std::vector<cv::Point> seeds;
    int num = 10;
    if (argc == 3) {
        num = strtol(argv[2], nullptr, 10);
        generate_seed(seeds, image.cols, image.rows, num);
    } else {
        generate_seed(seeds, image.cols, image.rows);
    }

    if (argc == 4) {
        randomColorization = strtol(argv[3], nullptr, 10);
    }

//    cv::Mat imageWithseeds;
//    display_seeds(image, imageWithseeds, seeds);
//    cv::namedWindow("seeds", cv::WINDOW_NORMAL);
//    cv::imshow("seeds", imageWithseeds);
    /*
    cv::Mat mask70 = cv::Mat::zeros(image.size(), CV_8UC3);
    seg(image, mask70, seeds, 0.70);

    cv::Mat mask80 = cv::Mat::zeros(image.size(), CV_8UC3);
    seg(image, mask80, seeds, 0.80);

    cv::Mat mask90 = cv::Mat::zeros(image.size(), CV_8UC3);
    seg(image, mask90, seeds, 0.90);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / 1000.0) << "ms" << std::endl;

    cv::Mat displayedseeds;
    display_seeds(image, displayedseeds, seeds);

    std::vector<cv::Mat> hImages1 = { displayedseeds, mask70 };
    std::vector<cv::Mat> hImages2 = { mask80, mask90 };

    cv::Mat row1;
    cv::hconcat(hImages1, row1);
    cv::Mat row2;
    cv::hconcat(hImages2, row2);

    std::vector<cv::Mat> vImages = { row1, row2 };

    cv::Mat finalOutput;
    cv::vconcat(vImages, finalOutput);

    cv::imshow("Segmentation avec différents seuil (70%, 80%, 90%)", finalOutput);

    cv::waitKey(0);*/

//    cv::Mat hsvImg;
//    cv::cvtColor(image, hsvImg, cv::COLOR_BGR2HSV);
//
//    std::vector<cv::Mat> hsvChannels;
//    cv::split(hsvImg, hsvChannels);
//
//    cv::Point coords(7, 2);
//    uchar H = hsvChannels[0].at<uchar>(coords);
//    uchar S = hsvChannels[1].at<uchar>(coords);
//    uchar V = hsvChannels[2].at<uchar>(coords);
//
//    std::pair<cv::Scalar, cv::Scalar> bounds =
//            interval_bounds({double(H), double(S), double(V)});
//
//    cv::Mat mask;
//    cv::inRange(hsvImg, bounds.first, bounds.second, mask);
//
//    cv::Mat result;
//    cv::bitwise_and(image, image, result, mask);
//
//    cv::namedWindow("Seg with HSV space color test", cv::WINDOW_NORMAL);
//    cv::imshow("Seg with HSV space color test", result);
//
//    cv::waitKey(0);

//    cv::Point test(206,274);
//    uchar valueH = hsvChannels[0].at<uchar>(test);
//    uchar valueS = hsvChannels[1].at<uchar>(test);
//    uchar valueV = hsvChannels[2].at<uchar>(test);
//
//    cv::Scalar value(valueH, valueS, valueV);

//    bool res = predicate(lowerb, upperb, value);

//    region_container regions;
//
//    int hexR1 = bgr_to_hex(cv::Vec3b(255, 0, 0));
//    int hexR2 = bgr_to_hex(cv::Vec3b(0, 0, 255));
//
//    cv::Point pos1(173,255);
//    auto H = (double)hsvChannels[0].at<uchar>(pos1);
//    auto S = (double)hsvChannels[1].at<uchar>(pos1);
//    auto V = (double)hsvChannels[2].at<uchar>(pos1);
//    cv::Scalar hsv1(H,S,V);
//    regions[hexR1].second = {cv::Scalar(0), cv::Scalar(0), cv::Scalar(0)};
//    std::pair<cv::Scalar, cv::Scalar> bounds = interval_bounds(hsv1);
//    regions[hexR1].second[0] = bounds.first;
//    regions[hexR1].second[1] = bounds.second;
//    update_mean(regions[hexR1], hsv1);
//    regions[hexR1].first.push_back(pos1);
//
//    cv::Point pos2(202,279);
//    H = (double)hsvChannels[0].at<uchar>(pos2);
//    S = (double)hsvChannels[1].at<uchar>(pos2);
//    V = (double)hsvChannels[2].at<uchar>(pos2);
//    cv::Scalar hsv2(H,S,V);
//    regions[hexR2].second = {cv::Scalar(0), cv::Scalar(0), cv::Scalar(0)};
//    bounds = interval_bounds(hsv2);
//    regions[hexR2].second[0] = bounds.first;
//    regions[hexR2].second[1] = bounds.second;
//    update_mean(regions[hexR2], hsv2);
//    regions[hexR2].first.push_back(pos2);
//
//    for (int i = 0; i < 50; ++i) {
//        cv::Point p1(8, 7);
//        H = (double)hsvChannels[0].at<uchar>(p1);
//        S = (double)hsvChannels[1].at<uchar>(p1);
//        V = (double)hsvChannels[2].at<uchar>(p1);
//        cv::Scalar tmp_hsv1(H,S,V);
//
//        update_mean(regions[hexR1], tmp_hsv1);
//        regions[hexR1].first.push_back(p1);
//    }
//
//    for (int i = 0; i < 100; ++i) {
//        cv::Point p2(8, 7);
//        H = (double)hsvChannels[0].at<uchar>(p2);
//        S = (double)hsvChannels[1].at<uchar>(p2);
//        V = (double)hsvChannels[2].at<uchar>(p2);
//        cv::Scalar tmp_hsv2(H,S,V);
//
//        update_mean(regions[hexR2], tmp_hsv2);
//        regions[hexR2].first.push_back(p2);
//    }
//
//    cv::Mat buffer = cv::Mat::zeros(image.size(), CV_32S);
//    merge(regions, buffer, hexR1, hexR2);
//
//    for (int i = 0; i < 50; ++i) {
//        cv::Point p1(8, 7);
//        H = (double)hsvChannels[0].at<uchar>(p1);
//        S = (double)hsvChannels[1].at<uchar>(p1);
//        V = (double)hsvChannels[2].at<uchar>(p1);
//        cv::Scalar tmp_hsv1(H,S,V);
//
//        update_mean(regions[hexR1], tmp_hsv1);
//        regions[hexR1].first.push_back(p1);
//    }

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);

    auto start = std::chrono::high_resolution_clock::now();
    seg(image, mask, seeds);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / 1000.0) << "ms" << std::endl;

//    cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
    cv::imshow("Segmentation", mask);
    cv::waitKey(0);
    return 0;
}
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
    std::uniform_int_distribution<> distribPosX(caseW*i,caseW*i+caseW);
    std::uniform_int_distribution<> distribPosY(caseH*j,caseH*j+caseH);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return cv::Point(px,py); // (col,row)
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
        int radius = 10;
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

// O(1)
std::pair<cv::Scalar, cv::Scalar> interval_bounds(cv::Scalar const& hsv)
{
    if (hsv[2] <= 30) { // Black
        return std::make_pair(cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 40));
    } else if (hsv[1] <= 70) {
        if (hsv[2] <= 175) { // Gray
            return std::make_pair(cv::Scalar(0, 0, hsv[2]-40), cv::Scalar(180, 45, hsv[2]+40));
        } else { // White
            return std::make_pair(cv::Scalar(0, 0, 180), cv::Scalar(180, 70, 255));
        }
    }
    return std::make_pair(cv::Scalar(hsv[0]-10, 70, 50), cv::Scalar(hsv[0]+10, 255, 255));
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

// O(1)
void merge(region_container & regions, int hexR1, int hexR2)
{
    std::vector<cv::Scalar> hsv1 = regions[hexR1].second;
    std::vector<cv::Scalar> hsv2 = regions[hexR2].second;
    hsv1[0] = min(hsv1[0], hsv2[0]);
    hsv1[1] = max(hsv1[1], hsv2[1]);

    int size1 = (int)regions[hexR1].first.size();
    int size2 = (int)regions[hexR2].first.size();
    hsv1[2] = (hsv1[2]*size1 + hsv2[2]*size2) / (size1 + size2);
    regions[hexR1].second = hsv1;

    regions[hexR1].first.splice(
            regions[hexR1].first.end(), regions[hexR2].first);

    regions.erase(hexR2);
}

// O(1)
void process(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, std::queue<cv::Point> & queue, cv::Point const& current, int hexKey)
{
    cv::Scalar lowerb = regions[hexKey].second[0];
    cv::Scalar upperb = regions[hexKey].second[1];

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i != 0 || j != 0) {
                cv::Point neighbor(current.x + i, current.y + j);
                if (neighbor.x >= 0 && neighbor.x < hsvChannels[0].cols &&
                    neighbor.y >= 0 && neighbor.y < hsvChannels[0].rows) {
                    int hexNeighbor = buffer.at<int>(neighbor);
                    if (hexNeighbor == 0) {
                        auto neighborH = (double) hsvChannels[0].at<uchar>(neighbor);
                        auto neighborS = (double) hsvChannels[1].at<uchar>(neighbor);
                        auto neighborV = (double) hsvChannels[2].at<uchar>(neighbor);
                        cv::Scalar hsvNeighbor(neighborH, neighborS, neighborV);

                        if (predicate(lowerb, upperb, hsvNeighbor)) {
                            buffer.at<int>(neighbor) = hexKey;
                            update_mean(regions[hexKey], hsvNeighbor);
                            regions[hexKey].first.push_back(neighbor);
                            queue.push(neighbor);
                        }
                    } else if (hexKey != hexNeighbor) {
                        std::vector<cv::Scalar> hsvNeighborRegion =
                                regions[hexNeighbor].second;
                        cv::Scalar currentRegionHsvMean =
                                regions[hexKey].second[2];
                        if (predicate(lowerb, upperb, hsvNeighborRegion[2]) &&
                            predicate(hsvNeighborRegion[0], hsvNeighborRegion[1], currentRegionHsvMean)) {
                            merge(regions, hexKey, hexNeighbor);
                        }
                    }
                }
            }
        }
    }
}

// O(w) avec w nombre de pixels parcourus
void growing(region_container & regions, std::vector<cv::Mat> const& hsvChannels,
             cv::Mat & buffer, cv::Point const& seed, int hexKey)
{
    std::queue<cv::Point> queue;
    queue.push(seed);

    auto seedH = (double)hsvChannels[0].at<uchar>(seed);
    auto seedS = (double)hsvChannels[1].at<uchar>(seed);
    auto seedV = (double)hsvChannels[2].at<uchar>(seed);

    cv::Scalar hsvSeed(seedH, seedS, seedV);
    std::pair<cv::Scalar, cv::Scalar> bounds = interval_bounds(hsvSeed);
    regions[hexKey].second.push_back(bounds.first);
    regions[hexKey].second.push_back(bounds.second);
    regions[hexKey].second.emplace_back(0);
    update_mean(regions[hexKey], hsvSeed);

    regions[hexKey].first.push_back(seed);

    while (!queue.empty()) {
        cv::Point current = queue.front();
        queue.pop();
        process(regions, hsvChannels, buffer, queue, current, hexKey);
    }
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

    /*std::vector<cv::Point> seeds;
    int num = 10;
    if (argc == 3) {
        num = strtol(argv[2], nullptr, 10);
        generate_seed(seeds, image.cols, image.rows, num);
    } else {
        generate_seed(seeds, image.cols, image.rows);
    }

    cv::Mat imageWithseeds;
    display_seeds(image, imageWithseeds, seeds);

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

    cv::Mat hsvImg;
    cv::cvtColor(image, hsvImg, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImg, hsvChannels);

//    cv::Point coords(349,29);
//    uchar H = hsvChannels[0].at<uchar>(coords);
//    uchar S = hsvChannels[1].at<uchar>(coords);
//    uchar V = hsvChannels[2].at<uchar>(coords);
//
//    std::pair<cv::Scalar, cv::Scalar> bounds =
//            interval_bounds({double(H), double(S), double(V)});

//    bounds.first = cv::Scalar(0, 0, V-30);
//    bounds.second = cv::Scalar(180, 45, V+30);

//    cv::Mat mask;
//    cv::inRange(hsvImg, bounds.first, bounds.second, mask);
//
//    cv::Mat result;
//    cv::bitwise_and(image, image, result, mask);
//
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

    region_container regions;

    int hexR1 = bgr_to_hex(cv::Vec3b(255, 0, 0));
    int hexR2 = bgr_to_hex(cv::Vec3b(0, 0, 255));

    cv::Point pos1(173,255);
    auto H = (double)hsvChannels[0].at<uchar>(pos1);
    auto S = (double)hsvChannels[1].at<uchar>(pos1);
    auto V = (double)hsvChannels[2].at<uchar>(pos1);
    cv::Scalar hsv1(H,S,V);
    regions[hexR1].second = {cv::Scalar(0), cv::Scalar(0), cv::Scalar(0)};
    std::pair<cv::Scalar, cv::Scalar> bounds = interval_bounds(hsv1);
    regions[hexR1].second[0] = bounds.first;
    regions[hexR1].second[1] = bounds.second;
    update_mean(regions[hexR1], hsv1);
    regions[hexR1].first.push_back(pos1);

    cv::Point pos2(202,279);
    H = (double)hsvChannels[0].at<uchar>(pos2);
    S = (double)hsvChannels[1].at<uchar>(pos2);
    V = (double)hsvChannels[2].at<uchar>(pos2);
    cv::Scalar hsv2(H,S,V);
    regions[hexR2].second = {cv::Scalar(0), cv::Scalar(0), cv::Scalar(0)};
    bounds = interval_bounds(hsv2);
    regions[hexR2].second[0] = bounds.first;
    regions[hexR2].second[1] = bounds.second;
    update_mean(regions[hexR2], hsv2);
    regions[hexR2].first.push_back(pos2);

    for (int i = 0; i < 100; ++i) {
        cv::Point p1(8, 7);
        H = (double)hsvChannels[0].at<uchar>(p1);
        S = (double)hsvChannels[1].at<uchar>(p1);
        V = (double)hsvChannels[2].at<uchar>(p1);
        cv::Scalar tmp_hsv1(H,S,V);

        cv::Point p2(8, 7);
        H = (double)hsvChannels[0].at<uchar>(p2);
        S = (double)hsvChannels[1].at<uchar>(p2);
        V = (double)hsvChannels[2].at<uchar>(p2);
        cv::Scalar tmp_hsv2(H,S,V);

        update_mean(regions[hexR1], tmp_hsv1);
        regions[hexR1].first.push_back(p1);
        update_mean(regions[hexR2], tmp_hsv2);
        regions[hexR2].first.push_back(p2);
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / 1000.0) << "ms" << std::endl;
    merge(regions, hexR1, hexR2);

    return 0;
}
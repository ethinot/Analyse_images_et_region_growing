#include <iostream>
#include <random>
#include <cstdlib> // strtol, rand, srand ... 
#include <fstream>
#include <string>
#include <vector>
#include<cmath>
#include<queue>
#include <execution>
#include <chrono>
#include <variant>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>

// opencv libs
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

float convolution(cv::Mat const& img, const cv::Mat& h, int x, int y)
{
    float sum = 0.0;
    for (int u = -1; u <= 1; ++u) {
        for (int v = -1; v <= 1; ++v) {
            sum += h.at<float>(1 + u, 1 + v) * img.at<uchar>(y + u, x + v);
        }
    }
    return sum;
}

// Fonction qui applique une matrice de filtrage sur une image via l'utilisation du produit de convolution
void filtering(const cv::Mat& src, cv::Mat& dst, const cv::Mat& h)
{
    assert(h.rows == 3 && h.cols == 3);
    int height = src.rows;
    int width = src.cols;
    dst = cv::Mat::zeros(src.size(), src.type());

    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(convolution(src, h, c, r));
        }
    }
}

void edge_detection(cv::Mat const& src, cv::Mat & dst, uint32_t ch=0)
{
    std::vector<cv::Mat> chls;
    cv::split(src, chls);
    dst = src.clone();
    float edf[9] =
            { -1, -1, -1,
              -1,  8, -1,
              -1, -1, -1};

    cv::Mat h(3, 3, CV_32F, edf);

    filtering(chls[ch], dst, h);
}

void framing(uint32_t imW, uint32_t imH, int& numC, int& numR, int& caseW, int& caseH)
{
    numC = (int)(std::log2f((float)imW));
    numR = (int)(std::log2f((float)imH));
    caseW = imW / numC;
    caseH = imH / numR;
}

// void draw_framing(cv::Mat & image, int thickness=2, cv::Scalar color=cv::Scalar(0, 0, 0))
// {
//     uint32_t rows = image.rows;
//     uint32_t cols = image.cols;

//     int num_cols, num_rows, caseW, caseH;
//     framing(cols, rows, num_cols, num_rows, caseW, caseH);

//     cv::Point start(0, 0);
//     cv::Point end(cols, 0);
//     for (uint32_t r = 0; r < num_rows; ++r) {
//         cv::line(image, start, end, color, thickness, cv::LINE_8);
//         start.y += caseH;
//         end.y += caseH;
//     }

//     start = cv::Point(0, 0);
//     end = cv::Point(0, rows);
//     for (uint32_t r = 0; r < num_cols; ++r) {
//         cv::line(image, start, end, color, thickness, cv::LINE_8);
//         start.x += caseW;
//         end.x += caseW;
//     }
// }

cv::Point rand_seed_position(int numC, int numR, int caseW, int caseH)
{
    std::mt19937 generator{ std::random_device{}() };
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

void generate_seed(cv::Mat const& img, std::vector<cv::Point>& seeds, uint32_t w, uint32_t h, uint32_t numSeeds=10)
{
    int numC, numR, caseW, caseH;
    framing(w, h, numC, numR, caseW, caseH);
    float dist = 25;
    for(uint32_t i = 0; i < numSeeds; ++i) {
        cv::Point seed;
        do {
            seed = rand_seed_position(numC, numR, caseW, caseH);
        } while(img.at<cv::Vec3b>(seed) == cv::Vec3b(255));
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

// Calculate a "unique" hash for a given BGR value
int bgr_hash(uchar B, uchar G, uchar R)
{
    int hash = 0;
    hash += B*1+G*10+R*100;
    hash += B*100+G*1000+R*10000;
    hash += B*10000+G*100000+R*1000000;
    hash += (B-G*R);
    return hash;
}

// Return the ratio between a and b. This will be use
// for checking the proximity of two hash value.
double proximity_ratio(int a, int b)
{
    if (a<b) return double(a)/double(b);
    if (a>b) return double(b)/double(a);
    return 1.0;
}

// Check if the proximity ratio if above the threshold percentage (based on color value)
bool color_predicate(int hash_1, int hash_2, double threshold)
{
    return proximity_ratio(hash_1, hash_2) >= threshold;
}

// Predicate based on intensity values
bool intensity_predicate(uchar seedIntensity, uchar currentIntensity, uchar threshold) {
    return abs(currentIntensity - seedIntensity) <= threshold;
}

cv::Vec3b generate_random_color() {
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distrib(25, 255);
    return {(uchar)distrib(generator), (uchar)distrib(generator), (uchar)distrib(generator)}; // générer du noir
}

// Return the same color than the regionColor but darker
cv::Vec3b get_border_color(const cv::Vec3b & regionColor) {
    return {regionColor[0], regionColor[1], (uchar)(regionColor[2] - 90)}; // Attention valeur négatif
}

void region_growing(const cv::Mat& inputImage, cv::Mat& outputMask, cv::Point seedPoint,
                    std::variant<uchar, double> threshold, bool intensityBased=true, bool displaySeeds=false) { // based on color distance

    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seedPoint);

    if (displaySeeds) {
        int radius = 10;
        cv::Scalar line_color(0,0,255);
        int thickness = 1;
        cv::circle(outputMask, seedPoint, radius, line_color, thickness);
    }

    cv::Vec3b regionColor = generate_random_color();
    cv::Vec3b borderColor = get_border_color(regionColor);

    int seedHash;
    if (!intensityBased) {
        cv::Vec3b seed = inputImage.at<cv::Vec3b>(seedPoint);
        seedHash = bgr_hash(seed[0], seed[1], seed[2]);
    }

    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    while (!pixelQueue.empty()) {
        cv::Point currentPixel = pixelQueue.front();
        pixelQueue.pop();

        if (outputMask.at<cv::Vec3b>(currentPixel) == cv::Vec3b(0, 0, 0)) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    if (i != 0 || j != 0) { // not current pixel coords
                        cv::Point neighbor(currentPixel.x + i, currentPixel.y + j);

                        if (neighbor.x >= 0 && neighbor.x < inputImage.cols &&
                            neighbor.y >= 0 && neighbor.y < inputImage.rows)
                        {
                            bool valid_predicat = false;
                            if (intensityBased) {
                                uchar neighborTmp = grayImage.at<uchar>(neighbor); // intensity based
                                uchar seedTmp = grayImage.at<uchar>(seedPoint);
                                uchar threshTmp = std::get<uchar>(threshold);
                                valid_predicat = intensity_predicate(seedTmp, neighborTmp, threshTmp);
                            } else {
                                cv::Vec3b neighborTmp = inputImage.at<cv::Vec3b>(neighbor); // color based
                                int neighborHash = bgr_hash(neighborTmp[0], neighborTmp[1], neighborTmp[2]);
                                double threshTmp = std::get<double>(threshold);
                                valid_predicat = color_predicate(seedHash, neighborHash, threshTmp);
                            }
                            if (valid_predicat) {
                                outputMask.at<cv::Vec3b>(currentPixel) = regionColor;
                                pixelQueue.push(neighbor);
                            } else {
                                outputMask.at<cv::Vec3b>(currentPixel) = borderColor;
                                // on calcule tout puis border
                            }
                        }
                    }
                }
            }
        }
    }
}

void segmentation(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& seeds,
                  std::variant<uchar, double> threshold, bool mode=true, bool displaySeeds=false)
{
#define MT 0
#if MT
    std::for_each(std::execution::par, seeds.begin(), seeds.end(),
		[&, inputImage](const cv::Point& seed)
		{
            cv::Point testSeedPoint(seed);
            region_growing(src, dst, testSeedPoint, threshold, mode, displaySeeds);
		});
#else
    for (uint32_t i = 0; i < seeds.size(); ++i) {
        cv::Point testSeedPoint(seeds[i]);
        region_growing(src, dst, testSeedPoint, threshold, mode, displaySeeds);
    }
#endif
}

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

std::vector<cv::Vec3b> generate_random_unique_BGR(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(55, 255);

    std::unordered_set<int> usedColors;
    std::vector<cv::Vec3b> randomColorList;

    for (int i = 0; i < size; ++i) {
        int colorValue;
        cv::Vec3b color;
        do {
            // Generate random values for RGB
            color[0] = static_cast<uchar>(dis(gen));
            color[1] = static_cast<uchar>(dis(gen));
            color[2] = static_cast<uchar>(dis(gen));

            // Convert RGB to a single integer for uniqueness check
            colorValue = bgr_to_hex(color);
        } while (usedColors.count(colorValue) > 0);

        usedColors.insert(colorValue);
        randomColorList.push_back(color);
    }

    return randomColorList;
}

using region_type = std::pair<std::vector<cv::Point>, double>;
using region_container = std::unordered_map<int, region_type>;

// merge r1 into r2
void sub_merge(region_type const& r1, region_type & r2, size_t const& s1, size_t const& s2)
{
    for (auto& pxl : r1.first) {
        r2.first.push_back(pxl);
    }
    r2.second = ((double)s1 * r1.second + (double)s2 * r2.second) / (double)(s1 + s2);
}

// merge two regions
void merge(region_container & regions, int hexR1, int hexR2)
{
    region_type region1 = regions[hexR1];
    region_type region2 = regions[hexR2];

    size_t sizeR1 = region1.first.size();
    size_t sizeR2 = region2.first.size();
    if (sizeR1 < sizeR2) {
        sub_merge(region1, region2, sizeR1, sizeR2);
        regions.erase(hexR1);
        regions[hexR2] = region2;
    } else {
        sub_merge(region2, region1, sizeR1, sizeR2);
        regions.erase(hexR2);
        regions[hexR1] = region1;
    }
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

void update_hash_mean(region_type & region, int hash)
{
    double dsize = (double)((region.first.empty()) ? 0 : region.first.size());
    region.second = (dsize*region.second + hash) / (dsize+1.0);
}

void process_neighbors(cv::Point const& currentPixel, cv::Mat const& image,
                      std::queue<cv::Point> & queue, cv::Mat & buffer, region_container & regions, int hexColor, double threshold)
{
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i != 0 || j != 0) { // not current pixel coords
                cv::Point neighborPixel(currentPixel.x + i, currentPixel.y + j);
                if (neighborPixel.x >= 0 && neighborPixel.x < image.cols &&
                        neighborPixel.y >= 0 && neighborPixel.y < image.rows)
                {
                    int hexNeighbor = buffer.at<int>(neighborPixel);
                    if (hexNeighbor == 0) {
                        cv::Vec3b neighborBgr = image.at<cv::Vec3b>(neighborPixel);
                        int neighborHash = bgr_hash(neighborBgr[0], neighborBgr[1], neighborBgr[2]);
                        int hashMean = (int)regions[hexColor].second;
                        if (color_predicate(hashMean, neighborHash, threshold)) {
                            buffer.at<int>(neighborPixel) = hexColor;
                            update_hash_mean(regions[hexColor], neighborHash);
                            regions[hexColor].first.push_back(neighborPixel);
                            queue.push(neighborPixel);
                        } else {
                            // on calcule tout puis border
                        }
                    } else if (hexNeighbor != hexColor) {
                        int hashCurrentRegion = (int)regions[hexColor].second;
                        int hashNeighborRegion = (int)regions[hexNeighbor].second;
                        if (color_predicate(hashCurrentRegion, hashNeighborRegion, threshold)) {
                            merge(regions, hexColor, hexNeighbor);
                        }
                    }
                }
            }
        }
    }
}

void growing(cv::Mat const& image, cv::Mat & buffer, region_container & regions,
             cv::Point const& seedPoint, cv::Vec3b const& color, double threshold)
{
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seedPoint);

    cv::Vec3b seedBgr = image.at<cv::Vec3b>(seedPoint);
    int seedHash = bgr_hash(seedBgr[0], seedBgr[1], seedBgr[2]);

    int hexColor = bgr_to_hex(color);
    buffer.at<int>(seedPoint) = hexColor;

    update_hash_mean(regions[hexColor], seedHash);
    regions[hexColor].first.push_back(seedPoint);

    while (!pixelQueue.empty()) {
        cv::Point currentPixel = pixelQueue.front();
        pixelQueue.pop();
        process_neighbors(currentPixel, image, pixelQueue,
                          buffer, regions, hexColor, threshold);
    }
}

cv::Vec3b pickColor(std::vector<cv::Vec3b> & colorList)
{
    int size = (int)colorList.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, size-1);

    int index = dis(gen);
    cv::Vec3b color = colorList[index];
    colorList.erase(colorList.begin()+index);
    return color;
}

void seg(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& seeds, double threshold)
{
    cv::Mat buffer = cv::Mat::zeros(src.size(), CV_32S);
    std::vector<cv::Vec3b> rdmColorList =
            generate_random_unique_BGR(seeds.size());

    region_container regions;

#define MT 0
#if MT
    std::for_each(std::execution::par, seeds.begin(), seeds.end(),
		[&, src](cv::Point const& seed)
		{
            cv::Point testSeedPoint(seed);
            cv::Vec3b color = pickColor(rdmColorList);
            growing(src, buffer, regions, testSeedPoint, color, threshold);
		});
#else
    for (size_t i = 0; i < seeds.size(); ++i) {
        cv::Point testSeedPoint(seeds[i]);
        growing(src, buffer, regions, testSeedPoint, rdmColorList[i], threshold);
    }
#endif

    fill_mask(dst, regions);
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

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
        generate_seed(image, seeds, image.cols, image.rows, num);
    } else {
        generate_seed(image, seeds, image.cols, image.rows);
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

    cv::waitKey(0);

    return 0;
}
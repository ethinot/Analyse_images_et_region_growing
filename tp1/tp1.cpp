#include <iostream>
#include <random>
#include <cstdlib> // strtol, rand, srand ... 
#include <fstream>
#include <string>
#include <vector>
#include <utility> // pair
#include<cmath>
#include<queue>
#include <execution>

// opencv libs
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


void framing(unsigned int imWidth, unsigned int imHeight, int& numCaseW, int& numCaseH, int& caseWidth, int& caseHeight) 
{
    numCaseW = (int)(std::log2f((float)imWidth));
    numCaseH = (int)(std::log2f((float)imHeight));
    caseWidth = imWidth / numCaseW;
    caseHeight = imHeight / numCaseH;
} 

void draw_framing(cv::Mat & image, int thickness=2, cv::Scalar color=cv::Scalar(0, 0, 0)) 
{
    unsigned int rows = image.rows;
    unsigned int cols = image.cols;

    int numCols, numRows, caseWidth, caseHeight;
    framing(cols, rows, numCols, numRows, caseWidth, caseHeight);

    cv::Point start(0, 0);
    cv::Point end(cols, 0);
    for (int r = 0; r < numRows; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.y += caseHeight;
        end.y += caseHeight;
    }

    start = cv::Point(0, 0);
    end = cv::Point(0, rows);
    for (int r = 0; r < numCols; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.x += caseWidth;
        end.x += caseWidth;
    }
}

std::pair<int, int> rand_germ_position(int numCaseW, int numCaseH, int caseWidth, int caseHeight)
{
    // std::cout << "random sampling for i in [" << 0 << " " << numCaseW - 1 << "]\n";
    // std::cout << "random sampling for j in [" << 0 << " " << numCaseH - 1 << "]\n";
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distribNCaseW(0, numCaseW - 1);
    std::uniform_int_distribution<> distribNCaseH(0, numCaseH - 1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    // std::cout << "value of i: " << i << ", case width: " << caseWidth << "\n";
    // std::cout << "value of j: " << j << ", case height: " << caseHeight << "\n";
    // std::cout << "random sampling for px in [" << caseWidth * i << " " << caseWidth * i + case_width << "]\n";
    // std::cout << "random sampling for py in [" << caseHeight * j << " " << caseHeight * j + caseHeight << "]\n";
    std::uniform_int_distribution<> distribPosX(caseWidth * i, caseWidth * i + caseWidth);
    std::uniform_int_distribution<> distribPosY(caseHeight * j, caseHeight * j + caseHeight);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return std::pair<int, int>(py, px); // (row,col) 
} 

void generate_germ(std::vector<std::pair<int,int>>& buffer, unsigned int width, unsigned int height, unsigned int numOfGerm = 10) 
{
    int numCaseW, numCaseH, caseWidth, caseHeight;
    framing(width, height, numCaseW, numCaseH, caseWidth, caseHeight);
    for(unsigned int i = 0; i < numOfGerm; ++i) 
        buffer.push_back(rand_germ_position(numCaseW, numCaseH, caseWidth, caseHeight));
}

void color_germs(cv::Mat const& src, cv::Mat & dst, std::vector<std::pair<int, int>> const& germs) 
{
    dst = src.clone();
    for(auto& germ : germs) {
        cv::Point center(germ.second, germ.first); // (col,row)
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

// Check if the proximity ratio is above the threshold percentage 
bool growing_predicate(int hash1, int hash2, double thresholdPercentage)  
{
    return proximity_ratio(hash1, hash2) >= thresholdPercentage;
}


cv::Vec3b generate_random_color() {
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distrib(25, 255);
    return cv::Vec3b(distrib(generator), distrib(generator), distrib(generator)); // générer du noir
}

// Return the same color than the regionColor but darker
// cv::Vec3b get_border_color(const cv::Vec3b & regionColor) {
//     return cv::Vec3b(regionColor[0], regionColor[1], regionColor[2] - 90); // Attention valeur négatif
// }

void region_growing(const cv::Mat& inputImage, cv::Mat& outputMask, cv::Point seedPoint, double threshold, bool displayGerms=false) {
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seedPoint);
    
    if (displayGerms) {
        int radius = 10;
        cv::Scalar line_color(0,0,255);
        int thickness = 1;
        cv::circle(outputMask, seedPoint, radius, line_color, thickness);
    }

    cv::Vec3b regionColor = generate_random_color();
    cv::Vec3b borderColor(255, 255, 255);

    cv::Vec3b seed_bgr = inputImage.at<cv::Vec3b>(seedPoint);
    uint32_t seed_hash = bgr_hash(seed_bgr[0], seed_bgr[1], seed_bgr[2]);

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
                            cv::Vec3b neighbor_bgr = inputImage.at<cv::Vec3b>(neighbor);
                            uint32_t neighbor_hash = bgr_hash(neighbor_bgr[0], neighbor_bgr[1], neighbor_bgr[2]);
                            if (growing_predicate(seed_hash, neighbor_hash, threshold)) {
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

void segmentation(const std::vector<std::pair<int,int>>& germs, const cv::Mat& inputImage, cv::Mat& outputMask, double threshold, bool displayGerms=false) 
{
#define MT 0
#if MT
	std::for_each(std::execution::par, germs.begin(), germs.end(), 
		[&, inputImage](const std::pair<int, int>& germ)
		{
            cv::Point testSeedPoint(germ.second, germ.first);
            region_growing(inputImage, outputMask, testSeedPoint, threshold, displayGerms);
		});
#else 
    for (unsigned int i = 0; i < germs.size(); ++i) {
        cv::Point testSeedPoint(germs[i].second, germs[i].first);
        region_growing(inputImage, outputMask, testSeedPoint, threshold, displayGerms);
    }
#endif
}

void gnuPlot(const cv::Mat& hist, const std::string& fileName, const int histSize) {
    std::ofstream dataFile("./ressources/" + fileName + ".txt");
    for (int i = 0; i < histSize; i++) {
        dataFile << i << " " << hist.at<float>(i) << std::endl;
    }
    dataFile.close();

    FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "set title 'Histogramme de %s'\n", fileName.c_str());
        fprintf(gnuplotPipe, "plot './ressources/%s.txt' with boxes\n", fileName.c_str());
        fflush(gnuplotPipe);
        //getchar(); 
        pclose(gnuplotPipe);
    } else {
        std::cerr << "Erreur lors de l'ouverture de Gnuplot." << std::endl;
    }
}

std::vector<int> extract_histogram_max_alues(const cv::Mat& image, const cv::Rect& roi, int n) {
    cv::Mat roiImage = image(roi); // region of interest

    cv::Mat histogram;
    int histSize = 180; 
    float range[] = { 0, 179 }; 
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    cv::calcHist(&roiImage, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);

    std::vector<int> topNValues;
    for (int i = 0; i < n; ++i) {
        double minValue, maxValue;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(histogram, &minValue, &maxValue, &minLoc, &maxLoc);

        topNValues.push_back(maxLoc.x);
        
        histogram.at<float>(maxLoc) = 0;
    }

    return topNValues;
}

int main(int argc, char** argv) {
    if (argc < 2) { 
        printf("usage: DisplayImage.out <Image_Path> (<num of germs>)\n"); 
        return -1; 
    } 

    cv::Mat image_rgb, image_hsv, image_gray; 
    image_rgb = cv::imread(argv[1]); //, cv::IMREAD_COLOR

    if (!image_rgb.data) { 
        printf("No image data \n"); 
        return -1; 
    } 

    cv::cvtColor(image_rgb, image_hsv, cv::COLOR_BGR2HSV);

    cv::cvtColor(image_rgb, image_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> hsv_channels;
    cv::split(image_hsv, hsv_channels);

    // Calculez les histogrammes pour chaque canal
    int hist_size = 256; // Vous pouvez ajuster cela selon vos besoins
    float range[] = {0, 256};
    const float* hist_range = {range};

    cv::Mat h_hist, s_hist, v_hist;
    cv::calcHist(&hsv_channels[0], 1, 0, cv::Mat(), h_hist, 1, &hist_size, &hist_range, true, false);
    cv::calcHist(&hsv_channels[1], 1, 0, cv::Mat(), s_hist, 1, &hist_size, &hist_range, true, false);
    cv::calcHist(&hsv_channels[2], 1, 0, cv::Mat(), v_hist, 1, &hist_size, &hist_range, true, false);

    cv::Mat g_hist;
    cv::calcHist(&image_gray, 1, 0, cv::Mat(), g_hist, 1, &hist_size, &hist_range, true, false);

    //cv::imshow("Histo 2D hsv", h_hist);
    gnuPlot(h_hist, "Histo hsv - Teinte", hist_size);
    // gnuPlot(s_hist, "Histo hsv - Saturation", hist_size);
    // gnuPlot(v_hist, "Histo hsv - Luminaissance", hist_size);

    // gnuPlot(g_hist, "Histo greyscale", hist_size);



    // std::vector<std::pair<int,int>> germs;
    // int num = 10;
    // if (argc == 3) { 
    //     num = strtol(argv[2], nullptr, 10);
    //     generate_germ(germs, image.cols, image.rows, num);
    // } else {
    //     generate_germ(germs, image.cols, image.rows);
    // }
    // for(auto& germ: germs) std::cout << "[rows:" << germ.first << ", cols:" << germ.second << "]\n";

    // cv::Mat imageWithGerms;
    // color_germs(image, imageWithGerms, germs);

    cv::Mat framing = image_rgb.clone();
    draw_framing(framing, 2);

    cv::imshow("Cadrillage", framing);

    cv::waitKey(0); 

    return 0;
}
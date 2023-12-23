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

void framing(unsigned int im_width, unsigned int im_height, int& num_case_w, int& num_case_h, int& case_width, int& case_height) 
{
    num_case_w = (int)(std::log2f((float)im_width));
    num_case_h = (int)(std::log2f((float)im_height));
    case_width = im_width / num_case_w;
    case_height = im_height / num_case_h;
} 

void draw_framing(cv::Mat & image, int thickness=2, cv::Scalar color=cv::Scalar(0, 0, 0)) 
{
    unsigned int rows = image.rows;
    unsigned int cols = image.cols;

    int num_cols, num_rows, case_width, case_height;
    framing(cols, rows, num_cols, num_rows, case_width, case_height);

    cv::Point start(0, 0);
    cv::Point end(cols, 0);
    for (unsigned int r = 0; r < num_rows; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.y += case_height;
        end.y += case_height;
    }

    start = cv::Point(0, 0);
    end = cv::Point(0, rows);
    for (unsigned int r = 0; r < num_cols; ++r) {
        cv::line(image, start, end, color, thickness, cv::LINE_8);
        start.x += case_width;
        end.x += case_width;
    }
}

std::pair<int, int> rand_germ_position(int num_case_w, int num_case_h, int case_width, int case_height)
{
    // std::cout << "random sampling for i in [" << 0 << " " << num_case_w-1 << "]\n";
    // std::cout << "random sampling for j in [" << 0 << " " << num_case_h-1 << "]\n";
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distribNCaseW(0, num_case_w-1);
    std::uniform_int_distribution<> distribNCaseH(0, num_case_h-1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    // std::cout << "value of i: " << i << ", case width: " << case_width << "\n";
    // std::cout << "value of j: " << j << ", case height: " << case_height << "\n";
    // std::cout << "random sampling for px in [" << case_width*i << " " << case_width*i+case_width << "]\n";
    // std::cout << "random sampling for py in [" << case_height*j << " " << case_height*j+case_height << "]\n";
    std::uniform_int_distribution<> distribPosX(case_width*i,case_width*i+case_width);
    std::uniform_int_distribution<> distribPosY(case_height*j,case_height*j+case_height);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return std::pair<int, int>(py, px); // (row,col) 
} 

void generate_germ(std::vector<std::pair<int,int>>& buffer, unsigned int width, unsigned int height, unsigned int num_of_germ=10) 
{
    int num_case_w, num_case_h, case_width, case_height;
    framing(width, height, num_case_w, num_case_h, case_width, case_height);
    for(unsigned int i = 0; i < num_of_germ; ++i) 
        buffer.push_back(rand_germ_position(num_case_w, num_case_h, case_width, case_height));
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

// Check if the proximity ratio if above the threshold percentage 
bool growing_predicate(int hash_1, int hash_2, double threshold_percentage)  
{
    return proximity_ratio(hash_1, hash_2) >= threshold_percentage;
}

/* CM Slide 47 
 - Chaque pixel est décrit selon certains cannaux : R,G,B,H,S,V,…
R, G, B : Rouge, Vert, Bleu

H, S, V : Teinte (Hue), Saturation, Valeur (ou luminance)

Précision : utilisation de la structure "Vec3b" pour prendre en compte les 3 cannaux d'une image de couleur,
            respectivement bleu, vert et rouge.
            La structure Vec3b contient des canaux, chaque canal est de type uchar (0 à 255).

*/


bool growingPredicate(const cv::Vec3b& seedPixel, const cv::Vec3b& actualPixel, int threshold) {
    int diffBlue = std::abs(static_cast<int>(seedPixel[0]) - static_cast<int>(actualPixel[0]));
    int diffGreen = std::abs(static_cast<int>(seedPixel[1]) - static_cast<int>(actualPixel[1]));
    int diffRed = std::abs(static_cast<int>(seedPixel[2]) - static_cast<int>(actualPixel[2]));

    return (diffBlue < threshold) && (diffGreen < threshold) && (diffRed < threshold);
}

cv::Vec3b generateRandomColor() {
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distrib(25, 255);
    return cv::Vec3b(distrib(generator), distrib(generator), distrib(generator)); // générer du noir
}

// Return the same color than the regionColor but darker
cv::Vec3b getBorderColor(const cv::Vec3b & regionColor) {
    return cv::Vec3b(regionColor[0], regionColor[1], regionColor[2] - 90); // Attention valeur négatif
}

void regionGrowing(const cv::Mat& inputImage, cv::Mat& outputMask, cv::Point seedPoint, double threshold, bool displayGerms=false) {
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seedPoint);
    
    if (displayGerms) {
        int radius = 10;
        cv::Scalar line_color(0,0,255);
        int thickness = 1;
        cv::circle(outputMask, seedPoint, radius, line_color, thickness);
    }

    cv::Vec3b regionColor = generateRandomColor();
    cv::Vec3b borderColor = getBorderColor(regionColor);

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
            regionGrowing(inputImage, outputMask, testSeedPoint, threshold, displayGerms);
		});
#else 
    for (unsigned int i = 0; i < germs.size(); ++i) {
        cv::Point testSeedPoint(germs[i].second, germs[i].first);
        regionGrowing(inputImage, outputMask, testSeedPoint, threshold, displayGerms);
    }
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) { 
        printf("usage: DisplayImage.out <Image_Path> (<num of germs>)\n"); 
        return -1; 
    } 

    cv::Mat image; 
    image = cv::imread(argv[1], cv::IMREAD_COLOR); 

    if (!image.data) { 
        printf("No image data \n"); 
        return -1; 
    } 

    std::vector<std::pair<int,int>> germs;
    int num = 10;
    if (argc == 3) { 
        num = strtol(argv[2], nullptr, 10);
        generate_germ(germs, image.cols, image.rows, num);
    } else {
        generate_germ(germs, image.cols, image.rows);
    }
    for(auto& germ: germs) std::cout << "[rows:" << germ.first << ", cols:" << germ.second << "]\n";

    cv::Mat imageWithGerms;
    color_germs(image, imageWithGerms, germs);

    //cv::imshow("Image", image);
    // cv::imshow("Image with germs", imageWithGerms);
    
    // channel -> bleu, vert et rouge
    // std::vector<cv::Mat> channels;
    // cv::split(image, channels);

    // cv::imshow("Niveau de bleu", channels[0]);
    // cv::imshow("Niveau de vert", channels[1]);
    // cv::imshow("Niveau de rouge", channels[2]);

    // CV_8UC3 permet les différent canaux de couleur contrairement à CV_8U

    cv::Mat mask70 = cv::Mat::zeros(image.size(), CV_8UC3);
    segmentation(germs, image, mask70, 0.70);

    cv::Mat mask80 = cv::Mat::zeros(image.size(), CV_8UC3);
    segmentation(germs, image, mask80, 0.80);

    cv::Mat mask90 = cv::Mat::zeros(image.size(), CV_8UC3);
    segmentation(germs, image, mask90, 0.85);

    cv::Mat framing = image.clone();
    draw_framing(framing, 2);

    cv::imshow("Cadrillage", framing);

    std::vector<cv::Mat> hImages1 = { image, mask70 }; 
    std::vector<cv::Mat> hImages2 = { mask80, mask90 }; 

    cv::Mat row1;
    cv::hconcat(hImages1, row1);
    cv::Mat row2;
    cv::hconcat(hImages2, row2);

    std::vector<cv::Mat> vImages = { row1, row2 }; 

    cv::Mat finalOutput;
    cv::vconcat(vImages, finalOutput);

    cv::imshow("Segmentation avec différents seuil (70%, 80%, 90%)", finalOutput);
    // cv::imshow("Région Growing (threshold: 70%)", mask70);
    // cv::imshow("Région Growing (threshold: 80%)", mask80);
    // cv::imshow("Région Growing (threshold: 90%)", mask90);

    // int a = bgr_hash(241, 156, 234);
    // int b = bgr_hash(230, 160, 225);
    // double threshold = 0.97;
    // std::cout << "Hash for a=[241, 156, 234]: " << a << std::endl;
    // std::cout << "Hash for b=[230, 160, 225]: " << b << std::endl;
    // std::cout << "Proximity ratio between a & b: " << proximity_ratio(a,b)*100 << "%\n";
    // std::cout << "With a threshold percentage of value: " << threshold 
    //         << ", should a and b be on the same zone ? " 
    //         << (growing_predicate(a, b, threshold) ? "Definitly, yes!\n" : "Absolutly not!\n"); 

    cv::waitKey(0); 

    return 0;
}
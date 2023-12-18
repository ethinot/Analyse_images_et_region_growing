#include <iostream>
#include <random>
#include <cstdlib> // strtol, rand, srand ... 
#include <fstream>
#include <string>
#include <vector>
#include <utility> // pair
#include<cmath>
#include<queue>

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
    return cv::Vec3b((rand() % 156) + 100, (rand() % 156) + 100, (rand() % 156) + 100); // générer du noir
}

// Return the same color than the regionColor but darker
cv::Vec3b getBorderColor(const cv::Vec3b & regionColor) {
    return cv::Vec3b(regionColor[0], regionColor[1], regionColor[2] - 90); // Attention valeur négatif
}

void regionGrowing(const cv::Mat& inputImage, cv::Mat& outputMask, cv::Point seedPoint, int threshold) {
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seedPoint);

    int radius = 10;
    cv::Scalar line_color(0,0,255);
    int thickness = 1;
    cv::circle(outputMask, seedPoint, radius, line_color, thickness);

    cv::Vec3b regionColor = generateRandomColor();
    cv::Vec3b borderColor = getBorderColor(regionColor);

    while (!pixelQueue.empty()) {
        cv::Point currentPixel = pixelQueue.front();
        pixelQueue.pop();

        if (outputMask.at<cv::Vec3b>(currentPixel) == cv::Vec3b(0, 0, 0)) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    // currentPixel.x > 0 && currentPixel.x < tailleImage && pareil y
                    cv::Point neighbor(currentPixel.x + i, currentPixel.y + j); 

                    if (neighbor.x >= 0 && neighbor.x < inputImage.cols &&
                        neighbor.y >= 0 && neighbor.y < inputImage.rows)
                    {
                        if (growingPredicate(inputImage.at<cv::Vec3b>(seedPoint), inputImage.at<cv::Vec3b>(neighbor), threshold)) {
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

int main(int argc, char** argv) {
    std::srand(static_cast<unsigned>(time(nullptr)));
    
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
    if (argc == 3) { 
        int num = strtol(argv[2], nullptr, 10);
        generate_germ(germs, image.cols, image.rows, num);
    } else {
        generate_germ(germs, image.cols, image.rows);
    }
    for(auto& germ: germs) std::cout << "[rows:" << germ.first << ", cols:" << germ.second << "]\n";

    cv::Mat imageWithGerms;
    color_germs(image, imageWithGerms, germs);

    //cv::imshow("Image", image);
    cv::imshow("Image with germs", imageWithGerms);
    
    // channel -> bleu, vert et rouge
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::imshow("Niveau de bleu", channels[0]);
    cv::imshow("Niveau de vert", channels[1]);
    cv::imshow("Niveau de rouge", channels[2]);

    // CV_8UC3 permet les différent canaux de couleur contrairement à CV_8U
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);

    int threshold = 40;

    for (unsigned int i = 0; i < 10; ++i) {
            cv::Point testSeedPoint(germs[i].second, germs[i].first);
            regionGrowing(image, mask, testSeedPoint, threshold);

    }

    cv::imshow("Région Growing", mask);

    cv::waitKey(0); 

    return 0;
}
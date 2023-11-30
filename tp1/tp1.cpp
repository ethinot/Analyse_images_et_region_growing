#include <iostream>
#include <random>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdio>
#include <cmath> 

#include <fstream>
#include <string>
#include <vector>
#include <utility>


void framing(unsigned int im_width, unsigned int im_height, int& num_case_w, int& num_case_h, int& case_width, int& case_height) 
{
    num_case_w = (int)(std::log2f((float)im_width));
    num_case_h = (int)(std::log2f((float)im_height));
    case_width = im_width / num_case_w;
    case_height = im_height / num_case_h;
} 

std::pair<int, int> rand_germ_position(int num_case_w, int num_case_h, int case_width, int case_height)
{
    std::mt19937 generator{ std::random_device{}() };
    std::uniform_int_distribution<> distribNCaseW(0, 9-1);
    std::uniform_int_distribution<> distribNCaseH(0, 8-1);
    int i = distribNCaseW(generator);
    int j = distribNCaseH(generator);
    std::uniform_int_distribution<> distribPosX(56*i,56*i+56);
    std::uniform_int_distribution<> distribPosY(55*j,55*j+55);
    int px = distribPosX(generator);
    int py = distribPosY(generator);
    return std::pair<int, int>(px, py); 
} 

void generate_germ(std::vector<std::pair<int,int>>& buffer, unsigned int width, unsigned int height, unsigned int num_of_germ=1) 
{
    int num_case_w, num_case_h, case_width, case_height;
    framing(width, height, num_case_w, num_case_h, case_width, case_height);
    for(int i = 0; i < num_of_germ; ++i) buffer.push_back(rand_germ_position(num_case_w, num_case_h, case_width, case_width));
}

void color_germs(cv::Mat const& src, cv::Mat & dst, std::vector<std::pair<int, int>> const& germs) 
{
    dst = src.clone();
    for(auto& germ : germs) {
        dst.at<cv::Vec3b>(germ.first, germ.second)[0] = 0; 
        dst.at<cv::Vec3b>(germ.first, germ.second)[1] = 0; 
        dst.at<cv::Vec3b>(germ.first, germ.second)[2] = 255; 
    } 
} 

int main(int argc, char** argv)
{
    if (argc != 2) { 
        printf("usage: DisplayImage.out <Image_Path>\n"); 
        return -1; 
    } 

    cv::Mat image; 
    image = cv::imread(argv[1], cv::IMREAD_COLOR); 

    if (!image.data) { 
        printf("No image data \n"); 
        return -1; 
    } 

    std::vector<std::pair<int,int>> germs;
    generate_germ(germs, image.rows, image.cols, 20);
    for(auto& germ: germs) std::cout << "[" << germ.first << ", " << germ.second << "]\n";

    cv::Mat imageWithGerms;
    color_germs(image, imageWithGerms, germs);

    cv::imshow("Image", image);
    cv::imshow("Image with germs", imageWithGerms);

    // int num_case_w, num_case_h, case_width, case_height;
    // framing(512, 440, num_case_w, num_case_h, case_width, case_height);
    
    // while(1) {
    //     std::mt19937 generator{ std::random_device{}() };
    //     std::uniform_int_distribution<> distribNCaseW(0, 9-1);
    //     std::uniform_int_distribution<> distribNCaseH(0, 8-1);
    //     int i = distribNCaseW(generator);
    //     int j = distribNCaseH(generator);
    //     std::uniform_int_distribution<> distribPosX(56*i,56*i+56);
    //     std::uniform_int_distribution<> distribPosY(55*j,55*j+55);
    //     int px = distribPosX(generator);
    //     int py = distribPosY(generator);
    //     std::cout << i << ", " << j << ", " << px << ", " << py << '\n';
    //     std::cin.get();
    // }

    cv::waitKey(0); 

    return 0;
}
#include <iostream>
#include <random>
#include <cstdlib> // strtol 
#include <fstream>
#include <string>
#include <vector>
#include <utility> // pair

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
    for(int i = 0; i < num_of_germ; ++i) 
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

int main(int argc, char** argv)
{
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

    cv::imshow("Image", image);
    cv::imshow("Image with germs", imageWithGerms);

    cv::waitKey(0); 

    return 0;
}
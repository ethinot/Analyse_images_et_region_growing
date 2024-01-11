#include "ImageProcessor.hpp"
#include "SegmentedRegion.hpp"
#include "GermsPositioning.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Enter relative path to an image.\n");
        return -1;
    }

    ImageProcessor imageProcessor;
    imageProcessor.process_image(argv[1]);

    cv::Point point1(10, 20);
    cv::Point point2(30, 40);
    double variance = 3.8;

    SegmentedRegion pv1(point1, point2, variance);

    // Accès aux données
    std::cout << point1 << ", " << point2 << ", on une variance de : " << variance <<std::endl;

    return 0;
}
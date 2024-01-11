#include "ImageProcessor.hpp"
#include "SegmentedRegion.hpp"
#include "GermsPositioning.hpp"
#include "ImageUtil.hpp"

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

    ImageUtil imageUtil;
    double varianceGrayscale, varianceHsv;
    varianceGrayscale = imageUtil.calculate_variance(imageProcessor.get_image_rgb(), 0);
    varianceHsv = imageUtil.calculate_variance(imageProcessor.get_image_rgb(), 1);

    std::cout<<"Variance GrayScale : " << varianceGrayscale << std::endl;
    std::cout<<"Variance HSV : " << varianceHsv << std::endl;

    return 0;
}
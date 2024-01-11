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

    ImageUtil imageUtil;
    double variance;

    GermsPositioningV2 germsPositioning;

    cv::Point initialTopLeft(0, 0);
    cv::Point initialBottomRight(imageProcessor.get_image_rgb().cols, imageProcessor.get_image_rgb().rows);

    variance = imageUtil.calculate_region_variance(imageProcessor.get_image_rgb(), initialTopLeft, initialBottomRight);

    std::cout << " Variance la la region primaire : " << variance <<std::endl;

    germsPositioning.divide_image(imageProcessor.get_image_rgb(), initialTopLeft, initialBottomRight, 7);

    std::cout << "Nombre de sous division : " << germsPositioning.get_germs_regions().size() << std::endl;


    return 0;
}
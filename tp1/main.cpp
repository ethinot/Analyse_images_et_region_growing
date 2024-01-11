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
    double variance, varianceH, varianceS, varianceV;

    GermsPositioningV2 germsPositioning;

    varianceH = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 0);
    varianceS = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 1);
    varianceV = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 2);

    std::cout << " Variance de la region primaire : " << variance <<std::endl;
    std::cout << " Variance H (teinte) : " << varianceH <<std::endl;
    std::cout << " Variance S (Sat) : " << varianceS <<std::endl;
    std::cout << " Variance V (Value) : " << varianceV <<std::endl;

    cv::Mat image = imageProcessor.get_image_rgb();
    std::vector<cv::Point> seeds = germsPositioning.position_germs(image, 5);

    std::cout << "Nombre de sous division : " << germsPositioning.get_germs_regions().size() << std::endl;
    std::cout << "Nombre de germe : " << seeds.size() << std::endl;


    return 0;
}
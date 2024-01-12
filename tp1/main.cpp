#include "ImageProcessor.hpp"
#include "SegmentedRegion.hpp"
#include "GermsPositioning.hpp"
#include "GrowAndMerge.hpp"
#include "ImageUtil.hpp"

int main(int argc, char** argv) {
//    if (argc < 2) {
//        printf("Enter relative path to an image.\n");
//        return -1;
//    }
//
//    ImageProcessor imageProcessor;
//    imageProcessor.process_image(argv[1]);
//
//    ImageUtil imageUtil;
//    double varianceH, varianceS, varianceV;
//
//    GermsPositioningV2 germsPositioning;
//
//    varianceH = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 0);
//    varianceS = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 1);
//    varianceV = imageUtil.calculate_channel_variance(imageProcessor.get_image_hsv(), 2);
//
//    std::cout << " Variance H (teinte) : " << varianceH <<std::endl;
//    std::cout << " Variance S (Sat) : " << varianceS <<std::endl;
//    std::cout << " Variance V (Value) : " << varianceV <<std::endl;
//
//    cv::Mat image = imageProcessor.get_image_rgb();
//    std::vector<cv::Point> seeds = germsPositioning.position_germs(image, 4);
//
//    std::cout << "Nombre de sous division : " << germsPositioning.get_germs_regions().size() << std::endl;
//    std::cout << "Nombre de germe : " << seeds.size() << std::endl;

    //    auto start = std::chrono::high_resolution_clock::now();

    // Test growing

    GrowAndMerge growAndMerge;

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

    if (argc == 3) {
        growAndMerge.set_num_seeds(strtol(argv[2], nullptr, 10));
    }

    GermsPositioningV1 positioningV1;

    std::vector<cv::Point> seeds;

    positioningV1.generate_seed(seeds, image.cols, image.rows, growAndMerge.get_num_seeds());

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);
    growAndMerge.rg_seg(image, mask, seeds);


//    cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
    cv::imshow("Segmentation", mask);
    cv::waitKey(0);

    return 0;
}
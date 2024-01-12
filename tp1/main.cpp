#include "ImageProcessor.hpp"
#include "SegmentedRegion.hpp"
#include "GermsPositioning.hpp"
#include "GrowAndMerge.hpp"
#include "ImageUtil.hpp"

std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point stop;

std::chrono::microseconds duration;

#define MEASURE_TIME(func) \
        start = std::chrono::high_resolution_clock::now(); \
        func; \
        stop = std::chrono::high_resolution_clock::now(); \
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        std::cout << "Time taken by " << #func << ": " << (duration.count() / 1000.0) << "ms" << std::endl; \


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

    GermsPositioningV2 positioningV2;

    std::vector<cv::Point> seedsV1;

    std::vector<cv::Point> seedsV2;


    MEASURE_TIME(positioningV1.generate_seed(seedsV1, image.cols, image.rows, growAndMerge.get_num_seeds()));

    MEASURE_TIME(positioningV2.position_germs(image, 4, seedsV2));

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);

    MEASURE_TIME(growAndMerge.rg_seg(image, mask, seedsV2));

    GermsDisplay germsDisplay;

    cv::Mat germsAndRegion;

    germsDisplay.display_germs(image, germsAndRegion, seedsV2);

    germsDisplay.display_segmented_regions(image, germsAndRegion, positioningV2.get_germs_regions(), cv::Scalar(0, 150, 0));

    std::cout << "Nombre de germe V1 : " << seedsV1.size() << std::endl;
    std::cout << "Nombre de germe V2 : " << seedsV2.size() << std::endl;

    cv::imshow("Segmentation", mask);
    cv::imshow("Germs and regions", germsAndRegion);

    cv::waitKey(0);

    return 0;
}
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
    if (argc < 2) {
        printf("Enter relative path to an image.\n");
        return -1;
    }

    ImageProcessor imageProcessor;
    imageProcessor.process_image(argv[1]);

    // Test growing

    GrowAndMerge growAndMerge;

    if (argc < 3) {
        printf("usage: DisplayImage.out <Image_Path> <show_edge (0 or 1)> (<num of seeds>)\n");
        return -1;
    }

    if (argc == 4) {
        std::cout<<"Set the number of seed ... \n";
        growAndMerge.set_num_seeds(strtol(argv[3], nullptr, 10));
    }

    GermsPositioningV1 positioningV1;

    GermsPositioningV2 positioningV2;

    std::vector<cv::Point> seedsV1;

    std::vector<cv::Point> seedsV2;

    cv::Mat image = imageProcessor.get_image_rgb();

    MEASURE_TIME(positioningV1.generate_seed(seedsV1, image.cols, image.rows, growAndMerge.get_num_seeds()));

    MEASURE_TIME(positioningV2.position_germs(image, 4, seedsV2));

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);

    bool showEdge = std::stoi(argv[2]) != 0;
    MEASURE_TIME(growAndMerge.rg_seg(image, mask, seedsV2, showEdge));

    GermsDisplay germsDisplay;

    cv::Mat germsAndRegion;

    germsDisplay.display_germs(image, germsAndRegion, seedsV2);

    germsDisplay.display_segmented_regions(image, germsAndRegion, positioningV2.get_germs_regions(), cv::Scalar(0, 150, 0));

    std::cout << "Nombre de germe V1 : " << seedsV1.size() << std::endl;
    std::cout << "Nombre de germe V2 : " << seedsV2.size() << std::endl;

    cv::imshow("Segmentation", mask);
    cv::imshow("Germs and regions", germsAndRegion);

    cv::Mat original = imageProcessor.get_image_original();
    cv::Mat rgbFiltered = imageProcessor.get_image_rgb();

    cv::imshow("Original", original);
    cv::imshow("RGB filtered", rgbFiltered);

    ImageUtil imageUtil;
    double vO = imageUtil.calculate_variance(original, false);
    double vF = imageUtil.calculate_variance(rgbFiltered, false);

    std::cout << "Variance original : " << vO << std::endl;
    std::cout << "Variance filtrer : " << vF << std::endl;

    cv::waitKey(0);
    return 0;
}
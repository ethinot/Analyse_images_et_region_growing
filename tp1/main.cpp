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

    bool showEdge = false;

    if(argc > 2) {
        try {
            if (std::stoi(argv[2]) != 0) {
                showEdge = true;
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << '\n';
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << '\n';
        }
    }

    // Initialisation

    ImageProcessor imageProcessor;
    imageProcessor.process_image(argv[1]);

    cv::Mat image = imageProcessor.get_image_rgb();

    // Germs / seeds positioning

    GermsPositioningV2 positioningV2;

    std::vector<cv::Point> seeds;

    MEASURE_TIME(positioningV2.position_germs(image, 5, seeds)); // the second parameter can be change

    // Grow and merge parts

    GrowAndMerge growAndMerge;

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);
    MEASURE_TIME(growAndMerge.rg_seg(image, mask, seeds, showEdge));

    // Display solutions

    GermsDisplay germsDisplay;

    cv::Mat germsAndRegion;

    germsDisplay.display_germs(image, germsAndRegion, seeds);
    germsDisplay.display_segmented_regions(image, germsAndRegion, positioningV2.get_germs_regions(), cv::Scalar(0, 150, 0));

    cv::imshow("Segmentation", mask);
    cv::imshow("Germs and regions", germsAndRegion);

    cv::waitKey(0);
    return 0;
}
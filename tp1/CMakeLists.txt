cmake_minimum_required(VERSION 3.12)
project(RegionGrowing)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Add your source files
set(SOURCES
    main.cpp
    ImageProcessor.hpp
    SegmentedRegion.hpp
    GermsPositioning.hpp
    ImageUtil.hpp
    GrowAndMerge.hpp
)

# Create the executable
add_executable(seg ${SOURCES})

# Include OpenCV headers
target_include_directories(seg PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV
target_link_libraries(seg PRIVATE ${OpenCV_LIBS})
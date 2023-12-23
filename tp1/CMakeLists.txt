cmake_minimum_required(VERSION 3.12)
project(TP1)

find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
add_executable(seg tp1.cpp)
target_link_libraries(seg ${OpenCV_LIBS} TBB::tbb)
cmake_minimum_required(VERSION 3.12)
project(TP1)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(seg tp1.cpp)
target_link_libraries(seg ${OpenCV_LIBS})
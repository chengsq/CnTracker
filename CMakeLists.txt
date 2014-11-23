#author shiqing cheng
#e-mail chengsq@gmail.com
#date   2014-11-23

cmake_minimum_required(VERSION 2.8)
project(CnTracker)

set(SRC main.cpp color_attributes_tracker.cpp config.cpp)

set(SRC_LIST ${SRC})

INCLUDE_DIRECTORIES(/usr/include)

add_executable(cntracker ${SRC_LIST})
TARGET_LINK_LIBRARIES(cntracker opencv_core opencv_imgproc opencv_highgui)

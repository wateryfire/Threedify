cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Export)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (Export Export.cc)
target_link_libraries (Export ${PCL_LIBRARIES})

add_executable (PcdToPly PcdToPly.cc)
target_link_libraries (PcdToPly ${PCL_LIBRARIES})

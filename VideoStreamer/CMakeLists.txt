cmake_minimum_required(VERSION 3.10)
project(stream_cctv)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(stream_cctv_cpp stream_cctv.cpp)
target_link_libraries(stream_cctv_cpp PRIVATE ${OpenCV_LIBS})

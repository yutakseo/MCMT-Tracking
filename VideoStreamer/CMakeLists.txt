cmake_minimum_required(VERSION 3.10)
project(stream_cctv LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(stream_cctv_cpp stream_cctv.cpp)
target_link_libraries(stream_cctv_cpp PRIVATE ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_target_properties(stream_cctv_cpp PROPERTIES
  BUILD_RPATH "/usr/lib/x86_64-linux-gnu"
  INSTALL_RPATH "/usr/lib/x86_64-linux-gnu"
)

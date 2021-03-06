############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

cmake_minimum_required(VERSION 3.5)

set (CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --expt-extended-lambda -O3" )

set(CMAKE_CXX_FLAGS "-O3")

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)


file( GLOB  hdr *.hpp *.h )

SET (HW5_files main.cu student.cu reference_calc.cpp)

CUDA_ADD_EXECUTABLE(HW5 ${HW5_files} ${hdr})


target_link_libraries( HW5 ${OpenCV_LIBS} )


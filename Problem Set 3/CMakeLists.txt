############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# minimum required cmake version

cmake_minimum_required(VERSION 3.5)

#to get all the commands so I am sure that we are passing O3 to both compilers
set( CMAKE_VERBOSE_MAKEFILE on )

set (CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --expt-extended-lambda -O3" )

#gcc flags
set(CMAKE_CXX_FLAGS "-O3")


find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)
    
CUDA_ADD_EXECUTABLE(HW3 ${HW3_files} ${hdr} ${cu})

target_link_libraries( HW3 ${OpenCV_LIBS} )

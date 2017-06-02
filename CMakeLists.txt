cmake_minimum_required(VERSION 2.8)
# Handling of relative directories by link_directories()
cmake_policy(SET CMP0015 NEW) 

project( image-segmenter )

find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
  message(">> OpenCV version: ${OpenCV_VERSION}")
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  link_directories( ${OpenCV_LIBS} )
endif(OpenCV_FOUND)

set(SRC source/main.cpp
		source/sort_permutation.h
		source/utility.h
)

add_executable( ${PROJECT_NAME} ${SRC} )

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
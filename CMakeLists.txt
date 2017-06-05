cmake_minimum_required(VERSION 2.8)
# Handling of relative directories by link_directories()
cmake_policy(SET CMP0015 NEW) 

project( image-segmenter )

find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
  message(STATUS ">> OpenCV version: ${OpenCV_VERSION}")
  message(STATUS ">>   Include dirs: ${OpenCV_INCLUDE_DIRS}")
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  message(STATUS ">>   Library dirs: ${OpenCV_LIBS}")
  link_directories( ${OpenCV_LIBS} )
endif(OpenCV_FOUND)

set(Boost_USE_STATIC_LIBS ON) # only find static libs
find_package(Boost COMPONENTS serialization)
if(Boost_FOUND)
  message(STATUS ">> Boost version: ${Boost_VERSION}")
  message(STATUS ">>   Include dirs: ${Boost_INCLUDE_DIRS}")
  include_directories( ${Boost_INCLUDE_DIRS} )
  message(STATUS ">>   Library dirs: ${Boost_LIBRARY_DIRS}")
  link_directories( ${Boost_LIBRARY_DIRS} )
endif(Boost_FOUND)

set(SRC source/main.cpp
		source/sort_permutation.h
		source/utility.h
)

add_executable( ${PROJECT_NAME} ${SRC} )

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES} )
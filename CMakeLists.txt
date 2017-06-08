cmake_minimum_required(VERSION 2.8.11)

# Options
option(build_tests "Build unit tests" OFF)

# Handling of relative directories by link_directories()
cmake_policy(SET CMP0015 NEW) 

project( image-segmenter )

################################################################
# External libraries 
################################################################

# OpenCV
find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
  message(STATUS ">> OpenCV version: ${OpenCV_VERSION}")
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  link_directories( ${OpenCV_LIBS} )
endif(OpenCV_FOUND)

# Boost
set(Boost_USE_STATIC_LIBS ON) # only find static libs
find_package(Boost COMPONENTS serialization)
if(Boost_FOUND)
  message(STATUS ">> Boost version: ${Boost_VERSION}")
  include_directories( ${Boost_INCLUDE_DIRS} )
  link_directories( ${Boost_LIBRARY_DIRS} )
endif(Boost_FOUND)

set(LIBS ${OpenCV_LIBS}
		 ${Boost_LIBRARIES})

################################################################
# Project structure
################################################################

set(PROJECT_LIB_NAME "${PROJECT_NAME}-lib")
set(PROJECT_CLITOOL_NAME "${PROJECT_NAME}-cli-tool")
set(PROJECT_TESTS_NAME "${PROJECT_NAME}-tests")
			 
set(SRC_LIB source/image_segmenter.h
			source/image_segmenter.cpp
			source/segment.h
			source/serialize_template_specializations.h
			source/utility.h
)

add_library( ${PROJECT_LIB_NAME} ${SRC_LIB})
target_link_libraries( ${PROJECT_LIB_NAME} ${LIBS} )
add_executable( ${PROJECT_CLITOOL_NAME} source/main.cpp )
target_link_libraries( ${PROJECT_CLITOOL_NAME} ${PROJECT_LIB_NAME} )

################################################################
# Unit tests
################################################################
if (build_tests)
	# GTest
	configure_file(CMakeLists.gtest.txt googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # Prevent overriding the parent project's compiler/linker settings on Windows
	add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
					 ${CMAKE_BINARY_DIR}/googletest-build)

	# Unit test project
	set(SRC_TESTS   test/main.cpp
					test/image-segmenter-tests.h
					test/image-segmenter-tests.cpp
					test/segment-tests.h
					test/segment-tests.cpp
	)
	add_executable( image-segmenter-tests ${SRC_TESTS} )
	target_link_libraries( image-segmenter-tests gtest gtest_main ${PROJECT_LIB_NAME} )
endif (build_tests)
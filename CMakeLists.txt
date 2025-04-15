cmake_minimum_required(VERSION 3.30)
project(card_recognition)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(card_recognition main.cpp
        src/SIFTDetector.cpp
        src/SIFTDetector.h
        src/HarrisCornerDetector.cpp
        src/HarrisCornerDetector.h
        src/CardRecognizer.cpp
        src/CardRecognizer.h
)

target_link_libraries(card_recognition PRIVATE ${OpenCV_LIBS})

set_target_properties(card_recognition PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

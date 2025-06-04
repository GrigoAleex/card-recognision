cmake_minimum_required(VERSION 3.30)
project(card_recognition)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build with tests/accuracy.cpp instead of main.cpp" OFF)

# Ensure build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

set(APP_SRCS 
    src/SIFTDetector.cpp
    src/HarrisCornerDetector.cpp
    src/CardRecognizer.cpp
)

if(BUILD_TESTING)
    message(STATUS ">>> Building TEST executable (using tests/accuracy.cpp)")
    set(MAIN_SRC tests/hyperparam.cpp)
else()
    message(STATUS ">>> Building normal executable (using main.cpp)")
    set(MAIN_SRC main.cpp)
endif()

add_executable(card_recognition
    ${APP_SRCS}
    ${MAIN_SRC}
)

target_link_libraries(card_recognition PRIVATE ${OpenCV_LIBS})

set_target_properties(card_recognition PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

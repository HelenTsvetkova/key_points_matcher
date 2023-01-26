cmake_minimum_required(VERSION 3.5)

project(key_points_represent_project LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

set(PROJECT_SOURCES
    main.cpp
)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)

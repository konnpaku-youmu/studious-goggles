cmake_minimum_required(VERSION 3.0)
add_compile_options(--std=c++17)

project(assignment)

find_package(OpenCV 4.5 REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/main.cc src/preprocess.cc)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
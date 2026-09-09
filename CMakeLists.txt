cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(F1HybridSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(f1-telemetry-sim
    src/physics.cu
    src/circuit_loader.cpp
    src/hud_renderer.cpp
    src/visualizer.cpp
    src/main.cpp 
)

target_link_libraries(f1-telemetry-sim sfml-graphics sfml-window sfml-system)
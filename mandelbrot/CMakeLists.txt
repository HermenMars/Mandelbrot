cmake_minimum_required(VERSION 3.24)

project(Mandelbrot
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# RTX 4060 = Ada Lovelace, compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(Mandelbrot
    main.cpp
    mandelbrot.cu
)

target_link_libraries(Mandelbrot PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

set_target_properties(Mandelbrot PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if (UNIX)
    target_link_libraries(Mandelbrot PRIVATE m)
endif()

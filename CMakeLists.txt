cmake_minimum_required(VERSION 3.20)

project(MandelbrotExplorer
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

find_program(GLSLC glslc REQUIRED)

set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADERS
    mandelbrot.vert
    mandelbrot.frag
)

foreach(SHADER ${SHADERS})
    add_custom_command(
        OUTPUT "${SHADER_BINARY_DIR}/${SHADER}.spv"

        COMMAND ${GLSLC}
            "${SHADER_SOURCE_DIR}/${SHADER}"
            -o "${SHADER_BINARY_DIR}/${SHADER}.spv"

        DEPENDS "${SHADER_SOURCE_DIR}/${SHADER}"

        COMMENT "Compiling ${SHADER}"
    )

    list(APPEND SPIRV_SHADERS
        "${SHADER_BINARY_DIR}/${SHADER}.spv"
    )
endforeach()

add_custom_target(shaders
    DEPENDS ${SPIRV_SHADERS}
)

add_executable(mandelbrot
    src/main.cpp
)

add_dependencies(mandelbrot shaders)

target_link_libraries(mandelbrot
    PRIVATE
        Vulkan::Vulkan
        glfw
)

target_compile_definitions(mandelbrot
    PRIVATE
        SHADER_DIR="${SHADER_BINARY_DIR}"
)

target_compile_options(mandelbrot
    PRIVATE
        -Wall
        -Wextra
)
# Mandelbrot Explorer

A GPU-accelerated Mandelbrot fractal explorer written in C++20 using Vulkan.

## Features

- GPU-accelerated Mandelbrot rendering
- Vulkan graphics pipeline
- GLSL fragment shaders
- Interactive zooming
- Camera movement
- Adjustable iteration count
- Smooth fractal coloring
- GLFW window management

## Technologies

- C++20
- Vulkan
- GLSL
- GLFW
- CMake
- Ninja

## Requirements

- C++20 compiler
- CMake
- Ninja
- Vulkan SDK components / Vulkan development libraries
- GLFW
- glslc shader compiler
- Vulkan-compatible GPU and driver

## Installation (Arch Linux)

Install dependencies:

```bash
sudo pacman -S --needed \
    base-devel \
    cmake \
    ninja \
    glfw \
    vulkan-headers \
    vulkan-icd-loader \
    vulkan-tools \
    shaderc
```

## Build

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Mandelbrot.git
cd Mandelbrot
```

Configure:

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
```

Compile:

```bash
cmake --build build
```

Run:

```bash
./build/mandelbrot
```

## Controls

| Input | Action |
|---|---|
| Mouse wheel | Zoom |
| W | Move up |
| A | Move left |
| S | Move down |
| D | Move right |
| + | Zoom in |
| - | Zoom out |
| Arrow up | Increase iterations |
| Arrow down | Decrease iterations |
| R | Reset view |

## Project Structure

```text
Mandelbrot/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── src/
│   └── main.cpp
└── shaders/
    ├── mandelbrot.vert
    └── mandelbrot.frag
```

## License

No License.
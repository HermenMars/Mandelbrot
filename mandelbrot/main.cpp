#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <omp.h>


struct MandelbrotParams
{
    double centerX;
    double centerY;

    double scale;

    int width;
    int height;

    int maxIterations;

    double aspectRatio;
};


extern "C"
bool cudaInitialize(
    int width,
    int height
);

extern "C"
bool cudaRender(
    uint8_t* pixels,
    const MandelbrotParams& params
);

extern "C"
void cudaShutdown();


enum class RenderMode
{
    GPU,
    CPU
};


/*
 * CPU Mandelbrot renderer.
 */
static void renderCPU(
    std::vector<uint8_t>& pixels,
    const MandelbrotParams& params
)
{
    const int width =
        params.width;

    const int height =
        params.height;

    const int maxIterations =
        params.maxIterations;


    /*
     * First calculate iteration counts.
     */
    std::vector<int> iterations(
        static_cast<size_t>(width) *
        static_cast<size_t>(height)
    );


#pragma omp parallel for schedule(static)
    for (int y = 0;
         y < height;
         ++y)
    {
        for (int x = 0;
             x < width;
             ++x)
        {
            const double normalizedX =
                (
                    static_cast<double>(x) -
                    static_cast<double>(width) * 0.5
                ) /
                static_cast<double>(width);

            const double normalizedY =
                (
                    static_cast<double>(y) -
                    static_cast<double>(height) * 0.5
                ) /
                static_cast<double>(width);

            const double cx =
                params.centerX +
                normalizedX *
                params.scale;

            const double cy =
                params.centerY +
                normalizedY *
                params.scale;


            double zx = 0.0;
            double zy = 0.0;

            int count = 0;

            for (; count < maxIterations; ++count)
            {
                const double zx2 =
                    zx * zx;

                const double zy2 =
                    zy * zy;

                if (zx2 + zy2 > 4.0)
                    break;

                zy =
                    2.0 * zx * zy + cy;

                zx =
                    zx2 - zy2 + cx;
            }


            const size_t index =
                static_cast<size_t>(y) *
                static_cast<size_t>(width) +
                static_cast<size_t>(x);

            if (count >= maxIterations)
                iterations[index] = -1;
            else
                iterations[index] = count;
        }
    }


    /*
     * Find minimum and maximum.
     */
    int minIterations =
        maxIterations;

    int maxFound =
        0;


#pragma omp parallel
    {
        int localMin =
            maxIterations;

        int localMax =
            0;


#pragma omp for nowait
        for (size_t i = 0;
             i < iterations.size();
             ++i)
        {
            const int value =
                iterations[i];

            if (value < 0)
                continue;

            localMin =
                std::min(
                    localMin,
                    value
                );

            localMax =
                std::max(
                    localMax,
                    value
                );
        }


#pragma omp critical
        {
            minIterations =
                std::min(
                    minIterations,
                    localMin
                );

            maxFound =
                std::max(
                    maxFound,
                    localMax
                );
        }
    }


    /*
     * Convert to colors.
     */
#pragma omp parallel for schedule(static)
    for (int y = 0;
         y < height;
         ++y)
    {
        for (int x = 0;
             x < width;
             ++x)
        {
            const size_t index =
                static_cast<size_t>(y) *
                static_cast<size_t>(width) +
                static_cast<size_t>(x);

            const int iteration =
                iterations[index];


            uint8_t r = 0;
            uint8_t g = 0;
            uint8_t b = 0;


            /*
             * Points inside the Mandelbrot set
             * are black.
             */
            if (iteration >= 0)
            {
                double t = 0.0;

                if (maxFound > minIterations)
                {
                    t =
                        static_cast<double>(
                            iteration -
                            minIterations
                        ) /
                        static_cast<double>(
                            maxFound -
                            minIterations
                        );
                }


                /*
                 * Classic palette.
                 */
                const double angle =
                    t *
                    6.283185307179586;


                r =
                    static_cast<uint8_t>(
                        (
                            0.5 +
                            0.5 *
                            std::cos(
                                angle + 4.0
                            )
                        ) *
                        t *
                        255.0
                    );


                g =
                    static_cast<uint8_t>(
                        (
                            0.5 +
                            0.5 *
                            std::cos(
                                angle + 2.0
                            )
                        ) *
                        t *
                        255.0
                    );


                b =
                    static_cast<uint8_t>(
                        (
                            0.5 +
                            0.5 *
                            std::cos(
                                angle
                            )
                        ) *
                        t *
                        255.0
                    );
            }


            const size_t pixel =
                index * 4;


            pixels[pixel + 0] = r;
            pixels[pixel + 1] = g;
            pixels[pixel + 2] = b;
            pixels[pixel + 3] = 255;
        }
    }
}


int main()
{
    constexpr unsigned int WIDTH =
        1280;

    constexpr unsigned int HEIGHT =
        720;


    sf::RenderWindow window(
        sf::VideoMode(
            {WIDTH, HEIGHT}
        ),
        "Mandelbrot"
    );


    window.setVerticalSyncEnabled(true);


    std::vector<uint8_t> pixels(
        static_cast<size_t>(WIDTH) *
        static_cast<size_t>(HEIGHT) *
        4
    );


    sf::Texture texture(
        {WIDTH, HEIGHT}
    );


    sf::Sprite sprite(
        texture
    );


    MandelbrotParams params;

    params.centerX =
        -0.743643887037151;

    params.centerY =
         0.131825904205330;

    params.scale =
        3.0;

    params.width =
        WIDTH;

    params.height =
        HEIGHT;

    params.maxIterations =
        500;

    params.aspectRatio =
        static_cast<double>(WIDTH) /
        static_cast<double>(HEIGHT);


    RenderMode renderMode =
        RenderMode::GPU;


    if (!cudaInitialize(
            WIDTH,
            HEIGHT))
    {
        std::cerr
            << "CUDA initialization failed.\n"
            << "Using CPU/OpenMP.\n";

        renderMode =
            RenderMode::CPU;
    }
    else
    {
        std::cout
            << "CUDA initialized.\n";
    }


    std::cout
        << "OpenMP threads: "
        << omp_get_max_threads()
        << '\n';


    bool needsRender =
        true;


    bool dragging =
        false;


    sf::Vector2i previousMouse;


    while (window.isOpen())
    {
        while (const std::optional event =
                   window.pollEvent())
        {
            /*
             * Window close.
             */
            if (event->is<
                    sf::Event::Closed>())
            {
                window.close();
            }


            /*
             * Keyboard.
             */
            if (const auto* key =
                    event->getIf<
                        sf::Event::KeyPressed>())
            {
                /*
                 * Escape.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::Escape)
                {
                    window.close();
                }


                /*
                 * GPU.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::G)
                {
                    renderMode =
                        RenderMode::GPU;

                    needsRender =
                        true;

                    std::cout
                        << "\nRenderer: GPU\n";
                }


                /*
                 * CPU.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::C)
                {
                    renderMode =
                        RenderMode::CPU;

                    needsRender =
                        true;

                    std::cout
                        << "\nRenderer: CPU/OpenMP\n";
                }


                /*
                 * Increase iterations.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::Up)
                {
                    params.maxIterations =
                        std::min(
                            params.maxIterations +
                            100,
                            10000
                        );

                    needsRender =
                        true;
                }


                /*
                 * Decrease iterations.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::Down)
                {
                    params.maxIterations =
                        std::max(
                            params.maxIterations -
                            100,
                            50
                        );

                    needsRender =
                        true;
                }


                /*
                 * Reset.
                 */
                if (key->scancode ==
                    sf::Keyboard::Scancode::R)
                {
                    params.centerX =
                        -0.743643887037151;

                    params.centerY =
                         0.131825904205330;

                    params.scale =
                        3.0;

                    params.maxIterations =
                        500;

                    needsRender =
                        true;
                }
            }


            /*
             * Mouse wheel zoom.
             */
            if (const auto* wheel =
                    event->getIf<
                        sf::Event::MouseWheelScrolled>())
            {
                const sf::Vector2i mouse =
                    sf::Mouse::getPosition(
                        window
                    );


                /*
                 * Complex coordinate before zoom.
                 */
                const double beforeX =
                    params.centerX +
                    (
                        (
                            static_cast<double>(
                                mouse.x
                            ) -
                            static_cast<double>(
                                WIDTH
                            ) * 0.5
                        ) /
                        static_cast<double>(
                            WIDTH
                        )
                    ) *
                    params.scale;


                const double beforeY =
                    params.centerY +
                    (
                        (
                            static_cast<double>(
                                mouse.y
                            ) -
                            static_cast<double>(
                                HEIGHT
                            ) * 0.5
                        ) /
                        static_cast<double>(
                            WIDTH
                        )
                    ) *
                    params.scale;


                /*
                 * Zoom.
                 */
                const double zoom =
                    wheel->delta > 0
                    ? 0.8
                    : 1.25;


                params.scale *=
                    zoom;


                /*
                 * Complex coordinate after zoom.
                 */
                const double afterX =
                    params.centerX +
                    (
                        (
                            static_cast<double>(
                                mouse.x
                            ) -
                            static_cast<double>(
                                WIDTH
                            ) * 0.5
                        ) /
                        static_cast<double>(
                            WIDTH
                        )
                    ) *
                    params.scale;


                const double afterY =
                    params.centerY +
                    (
                        (
                            static_cast<double>(
                                mouse.y
                            ) -
                            static_cast<double>(
                                HEIGHT
                            ) * 0.5
                        ) /
                        static_cast<double>(
                            WIDTH
                        )
                    ) *
                    params.scale;


                /*
                 * Keep the point under the mouse
                 * stationary.
                 */
                params.centerX +=
                    beforeX -
                    afterX;

                params.centerY +=
                    beforeY -
                    afterY;


                /*
                 * Increase iteration count
                 * as we zoom.
                 */
                const double zoomLevel =
                    3.0 /
                    params.scale;


                params.maxIterations =
                    std::clamp(
                        static_cast<int>(
                            300.0 +
                            80.0 *
                            std::log2(
                                std::max(
                                    1.0,
                                    zoomLevel
                                )
                            )
                        ),
                        300,
                        10000
                    );


                needsRender =
                    true;
            }


            /*
             * Start dragging.
             */
            if (const auto* mouse =
                    event->getIf<
                        sf::Event::MouseButtonPressed>())
            {
                if (mouse->button ==
                    sf::Mouse::Button::Left)
                {
                    dragging =
                        true;

                    previousMouse =
                        sf::Mouse::getPosition(
                            window
                        );
                }
            }


            /*
             * Stop dragging.
             */
            if (const auto* mouse =
                    event->getIf<
                        sf::Event::MouseButtonReleased>())
            {
                if (mouse->button ==
                    sf::Mouse::Button::Left)
                {
                    dragging =
                        false;
                }
            }
        }


        /*
         * Panning.
         */
        if (dragging)
        {
            const sf::Vector2i mouse =
                sf::Mouse::getPosition(
                    window
                );


            const int dx =
                mouse.x -
                previousMouse.x;

            const int dy =
                mouse.y -
                previousMouse.y;


            if (dx != 0 ||
                dy != 0)
            {
                params.centerX -=
                    static_cast<double>(dx) /
                    static_cast<double>(WIDTH) *
                    params.scale;


                params.centerY -=
                    static_cast<double>(dy) /
                    static_cast<double>(WIDTH) *
                    params.scale;


                previousMouse =
                    mouse;


                needsRender =
                    true;
            }
        }


        /*
         * Render ONLY when necessary.
         */
        if (needsRender)
        {
            if (renderMode ==
                RenderMode::GPU)
            {
                if (!cudaRender(
                        pixels.data(),
                        params))
                {
                    std::cerr
                        << "CUDA rendering failed.\n"
                        << "Switching to CPU.\n";

                    renderMode =
                        RenderMode::CPU;

                    renderCPU(
                        pixels,
                        params
                    );
                }
            }
            else
            {
                renderCPU(
                    pixels,
                    params
                );
            }


            texture.update(
                pixels.data()
            );


            needsRender =
                false;
        }


        /*
         * Display the already calculated image.
         *
         * No Mandelbrot calculation happens here.
         */
        window.clear();

        window.draw(
            sprite
        );

        window.display();
    }


    cudaShutdown();

    return 0;
}

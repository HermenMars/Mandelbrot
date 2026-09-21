#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>


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


/*
 * ------------------------------------------------------------
 * Utility functions
 * ------------------------------------------------------------
 */

__device__
__forceinline__
double clamp01(double value)
{
    if (value < 0.0)
        return 0.0;

    if (value > 1.0)
        return 1.0;

    return value;
}


/*
 * Smoothstep.
 */
__device__
__forceinline__
double smoothstep(
    double edge0,
    double edge1,
    double x
)
{
    double t =
        clamp01(
            (x - edge0) /
            (edge1 - edge0)
        );

    return t * t * (3.0 - 2.0 * t);
}


/*
 * ------------------------------------------------------------
 * Mandelbrot calculation
 * ------------------------------------------------------------
 *
 * Returns:
 *
 *   -1  = inside the Mandelbrot set
 *
 *   >=0 = smooth escape iteration
 *
 * Smooth iteration is important for getting the thin,
 * detailed coloring seen in the reference image.
 */
__device__
__forceinline__
double calculateIterations(
    double cx,
    double cy,
    int maxIterations
)
{
    double zx = 0.0;
    double zy = 0.0;


    for (int i = 0;
         i < maxIterations;
         ++i)
    {
        const double zx2 =
            zx * zx;

        const double zy2 =
            zy * zy;


        /*
         * Escape radius = 2.
         */
        if (zx2 + zy2 > 4.0)
        {
            /*
             * Smooth escape-time calculation.
             *
             * This removes the hard bands caused by
             * integer iteration counts.
             */
            const double magnitude =
                sqrt(
                    zx2 + zy2
                );


            /*
             * log(log(|z|) / log(2)) / log(2)
             */
            const double smooth =
                static_cast<double>(i)
                + 1.0
                -
                log(
                    log(magnitude) /
                    log(2.0)
                )
                /
                log(2.0);


            return smooth;
        }


        /*
         * z = z² + c
         */
        const double newZy =
            2.0 * zx * zy + cy;

        const double newZx =
            zx2 - zy2 + cx;


        zx = newZx;
        zy = newZy;
    }


    /*
     * Point is inside the set.
     */
    return -1.0;
}


/*
 * ------------------------------------------------------------
 * Reference-image color palette
 * ------------------------------------------------------------
 *
 * The palette is deliberately designed around the image
 * supplied by the user:
 *
 *     black
 *       ↓
 *     deep blue
 *       ↓
 *     bright blue
 *       ↓
 *     cyan
 *       ↓
 *     white
 *       ↓
 *     yellow/gold
 *
 * The exact distribution is controlled by the normalized
 * escape value.
 */
__device__
__forceinline__
void calculateColor(
    double t,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b
)
{
    /*
     * Mandelbrot interior.
     */
    if (t < 0.0)
    {
        r = 0;
        g = 0;
        b = 0;

        return;
    }


    /*
     * --------------------------------------------------------
     * COLOR CURVE
     * --------------------------------------------------------
     *
     * t is normalized:
     *
     * 0.0 = minimum escape value
     * 1.0 = maximum escape value
     *
     * We deliberately use several bands to resemble
     * the supplied image.
     */


    /*
     * Make the first portion extremely dark.
     *
     * This produces the dark blue region immediately
     * around the black Mandelbrot body.
     */
    if (t < 0.12)
    {
        const double x =
            t / 0.12;


        /*
         * Black -> deep blue.
         */
        r = static_cast<uint8_t>(
            1.0 * x
        );


        g = static_cast<uint8_t>(
            8.0 * x
        );


        b = static_cast<uint8_t>(
            70.0 +
            50.0 * x
        );


        return;
    }


    /*
     * --------------------------------------------------------
     * Deep blue -> bright blue
     * --------------------------------------------------------
     */
    if (t < 0.38)
    {
        const double x =
            (t - 0.12) /
            (0.38 - 0.12);


        r = static_cast<uint8_t>(
            1.0 +
            3.0 * x
        );


        g = static_cast<uint8_t>(
            8.0 +
            25.0 * x
        );


        b = static_cast<uint8_t>(
            120.0 +
            100.0 * x
        );


        return;
    }


    /*
     * --------------------------------------------------------
     * Bright blue -> cyan
     * --------------------------------------------------------
     */
    if (t < 0.62)
    {
        const double x =
            (t - 0.38) /
            (0.62 - 0.38);


        r = static_cast<uint8_t>(
            4.0 +
            25.0 * x
        );


        g = static_cast<uint8_t>(
            33.0 +
            150.0 * x
        );


        b = static_cast<uint8_t>(
            220.0 +
            35.0 * x
        );


        return;
    }


    /*
     * --------------------------------------------------------
     * Cyan -> almost white
     * --------------------------------------------------------
     */
    if (t < 0.80)
    {
        const double x =
            (t - 0.62) /
            (0.80 - 0.62);


        r = static_cast<uint8_t>(
            25.0 +
            220.0 * x
        );


        g = static_cast<uint8_t>(
            183.0 +
            72.0 * x
        );


        b = static_cast<uint8_t>(
            255.0
        );


        return;
    }


    /*
     * --------------------------------------------------------
     * White -> warm yellow
     * --------------------------------------------------------
     */
    if (t < 0.92)
    {
        const double x =
            (t - 0.80) /
            (0.92 - 0.80);


        r = static_cast<uint8_t>(
            245.0 +
            10.0 * x
        );


        g = static_cast<uint8_t>(
            255.0 -
            35.0 * x
        );


        b = static_cast<uint8_t>(
            255.0 -
            190.0 * x
        );


        return;
    }


    /*
     * --------------------------------------------------------
     * Yellow -> white/yellow highlight
     * --------------------------------------------------------
     */
    {
        const double x =
            (t - 0.92) /
            (1.0 - 0.92);


        r = 255;


        g = static_cast<uint8_t>(
            220.0 +
            35.0 * x
        );


        b = static_cast<uint8_t>(
            65.0 +
            190.0 * x
        );
    }
}


/*
 * ------------------------------------------------------------
 * Mandelbrot iteration kernel
 * ------------------------------------------------------------
 */
__global__
void mandelbrotIterationsKernel(
    double* iterations,
    MandelbrotParams params
)
{
    const int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;


    const int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;


    if (x >= params.width ||
        y >= params.height)
    {
        return;
    }


    /*
     * Convert pixel -> complex plane.
     *
     * Width is used as the reference dimension so
     * the fractal doesn't become distorted.
     */
    const double normalizedX =
        (
            static_cast<double>(x) -
            static_cast<double>(params.width) * 0.5
        )
        /
        static_cast<double>(params.width);


    const double normalizedY =
        (
            static_cast<double>(y) -
            static_cast<double>(params.height) * 0.5
        )
        /
        static_cast<double>(params.width);


    const double cx =
        params.centerX +
        normalizedX *
        params.scale;


    const double cy =
        params.centerY +
        normalizedY *
        params.scale;


    const double result =
        calculateIterations(
            cx,
            cy,
            params.maxIterations
        );


    iterations[
        static_cast<size_t>(y) *
        static_cast<size_t>(params.width) +
        static_cast<size_t>(x)
    ] = result;
}


/*
 * ------------------------------------------------------------
 * CUDA color kernel
 * ------------------------------------------------------------
 */
__global__
void colorKernel(
    uint8_t* pixels,
    const double* iterations,
    MandelbrotParams params,
    double minIterations,
    double maxIterations
)
{
    const int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;


    const int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;


    if (x >= params.width ||
        y >= params.height)
    {
        return;
    }


    const size_t index =
        static_cast<size_t>(y) *
        static_cast<size_t>(params.width) +
        static_cast<size_t>(x);


    const double iteration =
        iterations[index];


    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;


    /*
     * Mandelbrot interior remains black.
     */
    if (iteration >= 0.0)
    {
        double t = 0.0;


        /*
         * Normalize based on the actual minimum
         * and maximum escape values in this image.
         */
        if (maxIterations >
            minIterations)
        {
            t =
                (
                    iteration -
                    minIterations
                )
                /
                (
                    maxIterations -
                    minIterations
                );
        }


        t =
            clamp01(t);


        calculateColor(
            t,
            r,
            g,
            b
        );
    }


    const size_t pixel =
        index * 4;


    pixels[pixel + 0] = r;
    pixels[pixel + 1] = g;
    pixels[pixel + 2] = b;
    pixels[pixel + 3] = 255;
}


/*
 * ------------------------------------------------------------
 * CUDA buffers
 * ------------------------------------------------------------
 */

static uint8_t* d_pixels =
    nullptr;


static double* d_iterations =
    nullptr;


static size_t d_pixelsSize =
    0;


static size_t d_iterationsSize =
    0;


/*
 * ------------------------------------------------------------
 * CUDA initialization
 * ------------------------------------------------------------
 */
extern "C"
bool cudaInitialize(
    int width,
    int height
)
{
    const size_t pixelSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        4;


    const size_t iterationSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        sizeof(double);


    /*
     * Clean up existing buffers.
     */
    if (d_pixels != nullptr)
    {
        cudaFree(d_pixels);

        d_pixels =
            nullptr;
    }


    if (d_iterations != nullptr)
    {
        cudaFree(d_iterations);

        d_iterations =
            nullptr;
    }


    /*
     * Allocate pixel buffer.
     */
    if (cudaMalloc(
            reinterpret_cast<void**>(
                &d_pixels
            ),
            pixelSize
        ) != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "Failed to allocate CUDA pixel buffer.\n"
        );

        return false;
    }


    /*
     * Allocate iteration buffer.
     */
    if (cudaMalloc(
            reinterpret_cast<void**>(
                &d_iterations
            ),
            iterationSize
        ) != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "Failed to allocate CUDA iteration buffer.\n"
        );


        cudaFree(d_pixels);

        d_pixels =
            nullptr;


        return false;
    }


    d_pixelsSize =
        pixelSize;


    d_iterationsSize =
        iterationSize;


    return true;
}


/*
 * ------------------------------------------------------------
 * CUDA rendering
 * ------------------------------------------------------------
 */
extern "C"
bool cudaRender(
    uint8_t* hostPixels,
    const MandelbrotParams& params
)
{
    /*
     * 16x16 gives 256 CUDA threads per block.
     */
    const dim3 block(
        16,
        16
    );


    const dim3 grid(
        (
            params.width +
            block.x -
            1
        )
        /
        block.x,

        (
            params.height +
            block.y -
            1
        )
        /
        block.y
    );


    /*
     * --------------------------------------------------------
     * PASS 1
     * --------------------------------------------------------
     *
     * Calculate smooth iteration value for every pixel.
     */
    mandelbrotIterationsKernel<<<
        grid,
        block
    >>>(
        d_iterations,
        params
    );


    if (cudaGetLastError() !=
        cudaSuccess)
    {
        std::fprintf(
            stderr,
            "Mandelbrot CUDA kernel failed.\n"
        );

        return false;
    }


    if (cudaDeviceSynchronize() !=
        cudaSuccess)
    {
        std::fprintf(
            stderr,
            "CUDA synchronization failed.\n"
        );

        return false;
    }


    /*
     * Number of pixels.
     */
    const size_t pixelCount =
        static_cast<size_t>(
            params.width
        )
        *
        static_cast<size_t>(
            params.height
        );


    /*
     * Copy smooth iteration values to CPU.
     *
     * This allows us to calculate the exact
     * minimum and maximum for the current image.
     */
    double* hostIterations =
        new double[pixelCount];


    if (cudaMemcpy(
            hostIterations,
            d_iterations,
            d_iterationsSize,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess)
    {
        delete[] hostIterations;

        return false;
    }


    /*
     * --------------------------------------------------------
     * Find actual minimum and maximum.
     * --------------------------------------------------------
     */
    double minIterations =
        static_cast<double>(
            params.maxIterations
        );


    double maxIterations =
        0.0;


    for (size_t i = 0;
         i < pixelCount;
         ++i)
    {
        const double value =
            hostIterations[i];


        /*
         * Ignore Mandelbrot interior.
         */
        if (value < 0.0)
            continue;


        if (value <
            minIterations)
        {
            minIterations =
                value;
        }


        if (value >
            maxIterations)
        {
            maxIterations =
                value;
        }
    }


    /*
     * --------------------------------------------------------
     * PASS 2
     * --------------------------------------------------------
     *
     * Convert normalized escape values to colors.
     */
    colorKernel<<<
        grid,
        block
    >>>(
        d_pixels,
        d_iterations,
        params,
        minIterations,
        maxIterations
    );


    if (cudaGetLastError() !=
        cudaSuccess)
    {
        delete[] hostIterations;

        return false;
    }


    if (cudaDeviceSynchronize() !=
        cudaSuccess)
    {
        delete[] hostIterations;

        return false;
    }


    /*
     * Copy final image back to CPU.
     */
    if (cudaMemcpy(
            hostPixels,
            d_pixels,
            d_pixelsSize,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess)
    {
        delete[] hostIterations;

        return false;
    }


    delete[] hostIterations;


    return true;
}


/*
 * ------------------------------------------------------------
 * CUDA shutdown
 * ------------------------------------------------------------
 */
extern "C"
void cudaShutdown()
{
    if (d_pixels != nullptr)
    {
        cudaFree(d_pixels);

        d_pixels =
            nullptr;
    }


    if (d_iterations != nullptr)
    {
        cudaFree(d_iterations);

        d_iterations =
            nullptr;
    }


    d_pixelsSize =
        0;


    d_iterationsSize =
        0;
}

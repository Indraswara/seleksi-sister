#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../lib/gaussian.h"
#include <cmath>


/**
 * src: 
 *  - https://www.slideshare.net/slideshow/gaussian-image-blurring-in-cuda-c/56869492
 *  - https://github.com/NVIDIA/cuda-samples 
 *  - https://github.com/jIdle/GaussianBlur-CUDA
 *  - 
 */

/**
 * so yeah this is the gaussian blur kernel
 * @param inputImage input image
 * @param outputImage output image
 * @param width image width
 * @param height image height
 * @param channels image channels
 * @param filter gaussian filter
 * @param filterWidth filter width
 * @return void
 */
__global__ void gaussianBlurKernel(float* inputImage, float* outputImage, int width, int height, int channels, float* filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfFilterWidth = filterWidth / 2;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
            for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sum += inputImage[(iy * width + ix) * channels + c] * filter[(ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth)];
            }
        }
        outputImage[(y * width + x) * channels + c] = sum;
    }
}

/**
 * Apply Gaussian blur to an image
 * @param inputImage input image
 * @param outputImage output image 
 * @param width image width
 * @param height image height
 * @param channels image channels
 * @param sigma sigma value
 */

void gaussianBlur(float* inputImage, float* outputImage, int width, int height, int channels, float sigma) {
    int filterWidth = 2 * ceil(3 * sigma) + 1;
    float* filter = new float[filterWidth * filterWidth];
    float sum = 0.0f;

    for (int y = -filterWidth / 2; y <= filterWidth / 2; ++y) {
        for (int x = -filterWidth / 2; x <= filterWidth / 2; ++x) {
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma));
            filter[(y + filterWidth / 2) * filterWidth + (x + filterWidth / 2)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < filterWidth * filterWidth; ++i) {
        filter[i] /= sum;
    }

    float* d_inputImage;
    float* d_outputImage;
    float* d_filter;

    cudaMalloc(&d_inputImage, width * height * channels * sizeof(float));
    cudaMalloc(&d_outputImage, width * height * channels * sizeof(float));
    cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float));

    cudaMemcpy(d_inputImage, inputImage, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, d_filter, filterWidth);

    cudaMemcpy(outputImage, d_outputImage, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_filter);
    delete[] filter;
}
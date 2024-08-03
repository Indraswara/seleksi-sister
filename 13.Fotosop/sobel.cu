#include "sobel.h"
#include <cmath>

__global__ void sobelFilterKernel(float* inputImage, float* outputImage, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int halfFilterWidth = 1;
    float filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for(int c = 0; c < 3; ++c){
        float sumX = 0.0f;
        float sumY = 0.0f;
        for(int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky){
            for(int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx){
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sumX += inputImage[(iy * width + ix) * 3 + c] * filterX[ky + halfFilterWidth][kx + halfFilterWidth];
                sumY += inputImage[(iy * width + ix) * 3 + c] * filterY[ky + halfFilterWidth][kx + halfFilterWidth];
            }
        }
        outputImage[(y * width + x) * 3 + c] = sqrt(sumX * sumX + sumY * sumY);
    }
}

void sobelFilter(float* inputImage, float* outputImage, int width, int height){
    int size = width * height * 3 * sizeof(float); // Adjust size for RGB channels
    float* d_inputImage;
    float* d_outputImage;

    cudaMalloc(&d_inputImage, size);
    cudaMalloc(&d_outputImage, size);
    cudaMemcpy(d_inputImage, inputImage, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobelFilterKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(outputImage, d_outputImage, size, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
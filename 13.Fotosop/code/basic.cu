#include "../lib/basic.h"


#define CUDA_CHECK_ERROR(call)                                            \
    do {                                                                  \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Greyscale Kernel
__global__ void greyscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char r = input[3 * idx];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];
        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// Contrast Kernel
__global__ void contrastKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        for (int c = 0; c < 3; ++c) {
            int pixel = input[3 * idx + c];
            pixel = alpha * (pixel - 128) + 128;
            output[3 * idx + c] = min(max(pixel, 0), 255);
        }
    }
}

// Saturation Kernel
__global__ void saturationKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        float r = input[3 * idx] / 255.0f;
        float g = input[3 * idx + 1] / 255.0f;
        float b = input[3 * idx + 2] / 255.0f;
        float grey = 0.299f * r + 0.587f * g + 0.114f * b;
        r = grey + alpha * (r - grey);
        g = grey + alpha * (g - grey);
        b = grey + alpha * (b - grey);
        output[3 * idx] = min(max(r * 255.0f, 0.0f), 255.0f);
        output[3 * idx + 1] = min(max(g * 255.0f, 0.0f), 255.0f);
        output[3 * idx + 2] = min(max(b * 255.0f, 0.0f), 255.0f);
    }
}




// Greyscale Function
void greyscale(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t size = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, input.total() * input.elemSize());
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data, input.total() * input.elemSize(), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    greyscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

// Contrast Function
void contrast(const cv::Mat& input, cv::Mat& output, float alpha) {
    int width = input.cols;
    int height = input.rows;
    size_t size = input.total() * input.elemSize();
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    contrastKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, alpha);
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

// Saturation Function
void saturation(const cv::Mat& input, cv::Mat& output, float alpha) {
    int width = input.cols;
    int height = input.rows;
    size_t size = input.total() * input.elemSize();
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    saturationKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, alpha);
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

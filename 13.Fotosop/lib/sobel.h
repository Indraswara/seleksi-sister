#ifndef SOBEL_H
#define SOBEL_H
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>

/**
 * sobel filter function
 * @param inputImage : input image
 * @param outputImage : output image
 * @param width : width of the image
 * @param height : height of the image
 */
void sobelFilter(float* inputImage, float* outputImage, int width, int height);

#endif
#ifndef BASIC_H
#define BASIC_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include <math.h>


/**
 * src: https://github.com/KhosroBahrami/ImageFiltering_CUDA
 * 
 */

/**
 * spek wajib 
 * 1. Greyscale 
 * @param input : input image
 * @param output : output image
 * @param width : width of the image
 * @param height : height of the image
 */
__global__ void greyscaleKernel(unsigned char* input, unsigned char* output, int width, int height);

/**
 * 2. Contrast
 * @param input : input image
 * @param output : output image
 * @param width : width of the image
 * @param height : height of the image
 * @param alpha : contrast value
 */
__global__ void contrastKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha);

/**
 * 3. Saturation
 * @param input : input image
 * @param output : output image
 * @param width : width of the image
 * @param height : height of the image
 * @param alpha : saturation value
 */

__global__ void saturationKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha);

/**
 * greyscale function 
 * @param input : input image
 * @param output : output image
 */
void greyscale(const cv::Mat& input, cv::Mat& output);

/**
 * contrast function
 * @param input : input image
 * @param output : output image
 */
void contrast(const cv::Mat& input, cv::Mat& output, float alpha);

/**
 * saturation function
 * @param input : input image
 * @param output : output image
 */
void saturation(const cv::Mat& input, cv::Mat& output, float alpha);

#endif
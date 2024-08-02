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

__global__ void greyscaleKernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void contrastKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha);
__global__ void saturationKernel(unsigned char* input, unsigned char* output, int width, int height, float alpha);


void greyscale(const cv::Mat& input, cv::Mat& output);
void contrast(const cv::Mat& input, cv::Mat& output, float alpha);
void saturation(const cv::Mat& input, cv::Mat& output, float alpha);
#endif
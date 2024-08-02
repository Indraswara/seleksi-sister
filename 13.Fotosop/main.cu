#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include <math.h>
#include "basic.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " /path/to/file type (1: saturation, 2: contrast, 3: greyscale)" << std::endl;
        return -1;
    }
    std::string filePath = argv[1];
    int type = std::stoi(argv[2]);
    cv::Mat input = cv::imread(filePath);
    if (input.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    switch (type) {
        case 3: {
            cv::Mat greyOutput(input.rows, input.cols, CV_8UC1);
            greyscale(input, greyOutput);
            cv::imwrite("./output_greyscale.jpg", greyOutput);
            break;
        }
        case 2: {
            cv::Mat contrastOutput(input.rows, input.cols, CV_8UC3);
            float alpha = 1.5f; //value
            if(argv[3] != NULL){
                alpha = std::stof(argv[3]);
            }
            contrast(input, contrastOutput, alpha);
            cv::imwrite("./output_contrast.jpg", contrastOutput);
            break;
        }
        case 1: {
            cv::Mat saturationOutput(input.rows, input.cols, CV_8UC3);
            float alpha = 1.5f; // value
            if(argv[3] != NULL){
                alpha = std::stof(argv[3]);
            }
            saturation(input, saturationOutput, alpha);
            cv::imwrite("./output_saturation.jpg", saturationOutput);
            break;
        }
        default: {
            std::cerr << "Unknown type: " << type << std::endl;
            return -1;
        }
    }
    return 0;
}
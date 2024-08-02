#include "basic.h"
#include "gaussian.h"
#include "sobel.h"


using namespace std; 

int main(int argc, char** argv){
    if(argc < 3){
        cout << "Usage: " << argv[0] << " /path/to/file type(1: saturation, 2: contrast, 3: greyscale, 4: gaussian, 5: sobel)" << endl;
        cout << "Saturation: " << argv[0] << " /path/to/file 1 alpha" << endl;
        cout << "Contrast: " << argv[0] << " /path/to/file 2 alpha" << endl;
        cout << "Greyscale: " << argv[0] << " /path/to/file 3" << endl;
        cout << "Gaussian: " << argv[0] << " /path/to/file 4 sigma" << endl;
        cout << "Sobel: " << argv[0] << " /path/to/file 5" << endl;
        return -1;
    }
    string filePath = argv[1];
    int type = stoi(argv[2]);
    cv::Mat input = cv::imread(filePath);
    if(input.empty()){
        cerr << "Error loading image" << endl;
        return -1;
    }

    switch(type){
        //sobel
        case 5:{
            cv::Mat inputFloat;
            input.convertTo(inputFloat, CV_32FC3, 1.0 / 255.0);
            cv::Mat outputFloat(input.rows, input.cols, CV_32FC3); // Allocate memory for the output image
            sobelFilter((float*)inputFloat.ptr(), (float*)outputFloat.ptr(), input.cols, input.rows);
            cv::Mat output;
            outputFloat.convertTo(output, CV_8UC3, 255.0);
            cv::imwrite("./.output/output_sobel.jpg", output);
            break;
        }
        //gaussian
        case 4:{
           float sigma = 1.0f; // default sigma value
            if(argc > 3){
                sigma = std::stof(argv[3]);
            }
            cv::Mat inputFloat;
            input.convertTo(inputFloat, CV_32FC3);
            cv::Mat outputFloat(input.rows, input.cols, CV_32FC3); //allocate memory
            gaussianBlur((float*)inputFloat.ptr(), (float*)outputFloat.ptr(), input.cols, input.rows, input.channels(), sigma); //call func
            cv::Mat output; //convert
            outputFloat.convertTo(output, CV_8UC3); 
            cv::imwrite("./.output/output_gaussian.jpg", output); //save Image
            break; 
        }
        case 3:{
            cv::Mat greyOutput(input.rows, input.cols, CV_8UC1);
            greyscale(input, greyOutput);
            cv::imwrite("./.output/output_greyscale.jpg", greyOutput);
            break;
        }
        case 2:{
            cv::Mat contrastOutput(input.rows, input.cols, CV_8UC3);
            float alpha = 1.5f; // default value
            if(argv[3] != NULL){
                alpha = std::stof(argv[3]);
            }
            contrast(input, contrastOutput, alpha);
            cv::imwrite("./.output/output_contrast.jpg", contrastOutput);
            break;
        }
        case 1:{
            cv::Mat saturationOutput(input.rows, input.cols, CV_8UC3);
            float alpha = 1.5f; // default value
            if(argv[3] != NULL){
                alpha = std::stof(argv[3]);
            }
            saturation(input, saturationOutput, alpha);
            cv::imwrite("./.output/output_saturation.jpg", saturationOutput);
            break;
        }
        default: {
            std::cerr << "Unknown type: " << type << std::endl;
            return -1;
        }
    }
    return 0;
}
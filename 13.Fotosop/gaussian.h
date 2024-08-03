#ifndef GAUSSIAN_H
#define GAUSSIAN_H


/**
 * gaussian blur function
 * @param inputImage : input image
 * @param outputImage : output image
 * @param width : width of the image
 * @param height : height of the image
 * @param channels : number of channels
 * @param sigma : sigma value
 */
void gaussianBlur(float* inputImage, float* outputImage, int width, int height, int channels, float sigma);

#endif
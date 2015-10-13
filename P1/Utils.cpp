/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 13, 2015, 11:56 AM
 */

#include "Utils.h"

void gaussConvolution(Mat &im, Mat &mask, Mat &out) {

}

Mat myGetGaussianKernel1D(double sigma) {

    // Kernel size
    int ksize = 2 * sigma + 1;

    Mat kernel(1, ksize, CV_64F);

    // Compute the mask with the given sigma
    double sum = 0;
    for (int i = -sigma; i <= sigma; i++) {
        // i+sigma to start at index 0
        int index = i + sigma;
        kernel.at<double>(index) = (double) exp(-.5 * ((i * i) / sigma * sigma));
        sum += kernel.at<double>(index);
    }
    
    // Normalize the kernel to sum 1, it is a smooth kernel
    kernel = kernel * (1/sum);
    
    return kernel;
}
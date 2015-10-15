/* 
 * File:   main.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 12, 2015, 6:41 PM
 */

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Utils.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    // Exercise one, compute Gaussian kernel
    Mat kernel = myGetGaussianKernel1D(3);
    // Exercise two, compute convolution of a 1D signal vector and a kernel
//    Mat image = imread("./images/dog.bmp",IMREAD_GRAYSCALE);
//    drawImage(image, "antes");
    // create and initialize 3 channel Mat
    Mat vector(1,7, CV_64FC3, CV_RGB(50, 150, 255));
//    Mat vector(1,7, CV_64F);
//    vector.at<double>(0) = 10;
//    vector.at<double>(1) = 50;
//    vector.at<double>(2) = 60;
//    vector.at<double>(3) = 10;
//    vector.at<double>(4) = 20;
//    vector.at<double>(5) = 40;
//    vector.at<double>(6) = 30;
    Mat result = convolutionOperator1D(vector, kernel, BORDER_REFLECT);
    return 0;
}
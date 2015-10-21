/* 
 * File:   main.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 12, 2015, 6:41 PM
 */


#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Utils.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    bool debug = true;

    // Exercise one, compute Gaussian kernel
    double sigma = 50;
    Mat kernel = myGetGaussianKernel1D(sigma);
    if (!debug) {
        cout << "EXCERSICE ONE RESULT: (with sigma=" << sigma << ")" << endl;
        cout << setw(15) << "Kernel: " << kernel << endl;
        cout << "Press enter to show next exercise result" << endl;
        cin.get();
    }

    // Exercise two, compute convolution of a 1D signal vector and a kernel
    // create and initialize 3 channel Mat
    Mat vector(1, 7, CV_64FC3, CV_RGB(50, 150, 255));
    Mat result = convolutionOperator1D(vector, kernel, BORDER_CONSTANT);
    if (!debug) {
        cout << "EXCERSICE TWO RESULT: (with BORDER_CONSTANT)" << endl;
        cout << setw(15) << "Convol1D: " << result << endl;
        cout << "Press enter to show next example" << endl;
        cout << "EXCERSICE TWO RESULT: (with BORDER_REFLECT)" << endl;
        result = convolutionOperator1D(vector, kernel, BORDER_REFLECT);
        cout << setw(15) << "Convol1D: " << result << endl;
        cout << "Press enter to show next exercise result" << endl;
        cin.get();
    }

    // Exercise three
    if (debug) {
        cout << "EXCERSICE THREE RESULT:" << endl;
        Mat colorImage = imread("./images/dog.bmp", IMREAD_UNCHANGED);
        Mat grayImage = imread("./images/dog.bmp", IMREAD_GRAYSCALE);

        drawImage(colorImage, "Before Convolution | ColorImage");
        drawImage(grayImage, "Before Convolution | GrayImage");

        colorImage = computeConvolution(colorImage, sigma);
        grayImage = computeConvolution(grayImage, sigma);

        drawImage(colorImage, "After Convolution | ColorImage");
        drawImage(grayImage, "After Convolution | GrayImage");
        
        waitKey(0);
//        destroyWindow(windowName);
    }
    return 0;
}
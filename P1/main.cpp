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

int main() {

    bool debug = false;

    // Exercise one, compute Gaussian kernel
    double sigma = 3;
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
    if (!debug) {
        cout << "EXCERSICE THREE RESULT:" << endl;
        Mat colorImage = imread("./imagenes/dog.bmp", IMREAD_UNCHANGED);
        Mat grayImage = imread("./imagenes/dog.bmp", IMREAD_GRAYSCALE);

        drawImage(colorImage, "Before Convolution | ColorImage");
        drawImage(grayImage, "Before Convolution | GrayImage");

        colorImage = computeConvolution(colorImage, sigma);
        grayImage = computeConvolution(grayImage, sigma);

        drawImage(colorImage, "After Convolution | ColorImage");
        drawImage(grayImage, "After Convolution | GrayImage");

        cout << "Press enter to show next exercise result" << endl;
        cin.get();
    }

    // Exercise four
    if (!debug) {
        cout << "EXCERSICE FOUR RESULT:" << endl;

        double sigma1 = 7;
        double sigma2 = 10;
        Mat low = imread("./imagenes/dog.bmp", IMREAD_UNCHANGED);
        Mat high = imread("./imagenes/cat.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid = hybridImage(high, low, sigma1, sigma2);
        drawHybrid(hybrid);

        cout << "Press enter to show the next hybrid image" << endl;

        low = imread("./imagenes/marilyn.bmp", IMREAD_UNCHANGED);
        high = imread("./imagenes/einstein.bmp", IMREAD_UNCHANGED);

        sigma1 = 3;
        sigma2 = 5.5;
        std::vector<Mat> hybrid2 = hybridImage(high, low, sigma1, sigma2);
        drawHybrid(hybrid2);

        cout << "Press enter to show the next hybrid image" << endl;

        sigma1 = 3;
        sigma2 = 7;
        low = imread("./imagenes/fish.bmp", IMREAD_UNCHANGED);
        high = imread("./imagenes/submarine.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid3 = hybridImage(high, low, sigma1, sigma2);
        drawHybrid(hybrid3);
        cout << "Press enter to show the next hybrid image" << endl;

        sigma1 = 3;
        sigma2 = 7;
        low = imread("./imagenes/motorcycle.bmp", IMREAD_UNCHANGED);
        high = imread("./imagenes/bicycle.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid4 = hybridImage(high, low, sigma1, sigma2);
        drawHybrid(hybrid4);

        sigma1 = 5;
        sigma2 = 3;
        low = imread("./imagenes/bird.bmp", IMREAD_UNCHANGED);
        high = imread("./imagenes/plane.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid5 = hybridImage(high, low, sigma1, sigma2);
        drawHybrid(hybrid5);

        cout << "Press enter to show the next hybrid image" << endl;

        cout << "Press enter to show the next exercise result" << endl;
        cin.get();

        // Exercise five
        gaussianPyramid(hybrid.at(0), 7);
        gaussianPyramid(hybrid2.at(0), 7);
        gaussianPyramid(hybrid3.at(0), 7);
        gaussianPyramid(hybrid4.at(0), 7);
        gaussianPyramid(hybrid5.at(0), 7);
    }

    return 0;
}

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

    bool debug = false;

    // Exercise one, compute Gaussian kernel
    double sigma = 10;
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
        Mat colorImage = imread("./images/dog.bmp", IMREAD_UNCHANGED);
        Mat grayImage = imread("./images/dog.bmp", IMREAD_GRAYSCALE);

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

        // low = marylin, high = einstein, 25, 8
        // low = dog, high = cat, 20, 12
        // low = fish, high = submarine, 8, 12
        // low = motorcycle, high = bicycle, 8, 15
        // low = bird, high = plane, 6, 15

        Mat low = imread("./images/dog.bmp", IMREAD_UNCHANGED);
        Mat high = imread("./images/cat.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid = hybridImage(high, low, 12, 20);
        drawHybrid(hybrid);

        cout << "Press enter to show the next hybrid image" << endl;

        low = imread("./images/marilyn.bmp", IMREAD_UNCHANGED);
        high = imread("./images/einstein.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid2 = hybridImage(high, low, 25, 8);
        drawHybrid(hybrid2);

        cout << "Press enter to show the next hybrid image" << endl;

        low = imread("./images/fish.bmp", IMREAD_UNCHANGED);
        high = imread("./images/submarine.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid3 = hybridImage(high, low, 12, 8);
        drawHybrid(hybrid3);

        cout << "Press enter to show the next hybrid image" << endl;

        low = imread("./images/motorcycle.bmp", IMREAD_UNCHANGED);
        high = imread("./images/bicycle.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid4 = hybridImage(high, low, 15, 8);
        drawHybrid(hybrid4);

        low = imread("./images/bird.bmp", IMREAD_UNCHANGED);
        high = imread("./images/plane.bmp", IMREAD_UNCHANGED);

        std::vector<Mat> hybrid5 = hybridImage(high, low, 15, 6);
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
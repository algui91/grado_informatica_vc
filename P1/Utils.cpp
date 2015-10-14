/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 13, 2015, 11:56 AM
 */

#include "Utils.h"

void gaussConvolution(Mat &im, Mat &mask, Mat &out) {

}

//////////////////////////////////////////////////////

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
    kernel = kernel * (1 / sum);

    return kernel;
}

//////////////////////////////////////////////////////

Mat convolutionOperator1D(Mat &signalVector, Mat &kernel, BorderTypes border) {

    int extraBorder = kernel.cols / 2;

    Mat signalWithBorder(1, signalVector.cols + kernel.cols / 2, signalVector.type());
    // Add extra borders to the vector to solve boundary issue
    //TODO: parametrice border type
    copyMakeBorder(signalVector, signalWithBorder, 0, 0, extraBorder, extraBorder, BORDER_CONSTANT, Scalar(0));
    // Vector to store the convolution operation
    Mat filtered = signalWithBorder.clone();
//    kernel.at<double>(0) = (double) 1 / 3;
//    kernel.at<double>(1) = (double) 1 / 3;
//    kernel.at<double>(2) = (double) 1 / 3;
    // If the # channels is > 1, we need to split the vector into its channels
    if (signalVector.channels() == 1) {
        // Create a ROI to pass along the vector and compute convolution with the kernel
        Mat roi(signalWithBorder, Rect(0, 0, kernel.cols, 1));
        for (int i = extraBorder; i < signalWithBorder.cols - extraBorder; i++) {
            Mat r = roi.mul(kernel);
            cout << roi << endl;
            cout << r << endl;
            cout << *(sum(r).val) << endl;
            filtered.at<double>(i) = (double) *(sum(r).val);
            cout << filtered << endl;
            roi = roi.adjustROI(0, 0, -1, 1);
        }
    } else {
        //        Mat channels[signalVector.channels()];
        //        split(signalVector, channels);
        //        for (int i = 0; i < signalVector.channels(); i++) {
        //            channels = channels
        //        }
    }

    return filtered;
}

//////////////////////////////////////////////////////

void drawImage(Mat &m, string windowName) {
    if (!m.empty()) {
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, m);
        waitKey(0);
        destroyWindow(windowName);
    }
}
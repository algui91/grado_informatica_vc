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
    double ssum = 0;
    for (int i = -sigma; i <= sigma; i++) {
        // i+sigma to start at index 0
        int index = i + sigma;
        kernel.at<double>(index) = (double) exp(-.5 * ((i * i) / (sigma * sigma)));
        ssum += kernel.at<double>(index);
    }

    // Normalize the kernel to sum 1, it is a smooth kernel
    kernel = kernel * (1 / ssum);

    return kernel;
}

//////////////////////////////////////////////////////

Mat convolutionOperator1D(Mat &signalVector, Mat &kernel, BorderTypes border) {

    int extraBorder = kernel.cols / 2;

    vector<Mat> signalVectorByChannels(signalVector.channels());
    split(signalVector, signalVectorByChannels);

    Mat filtered;

    for (vector<Mat>::const_iterator it = signalVectorByChannels.begin(); it != signalVectorByChannels.end(); ++it) {
        Mat m = *(it);
        // Create a new Mat with the extra borders needed
        Mat signalWithBorder(1, m.cols + kernel.cols / 2, m.type());
        // Add extra borders to the vector to solve boundary issue
        copyMakeBorder(m, signalWithBorder, 0, 0, extraBorder, extraBorder, border, Scalar(0));
        // Vector to store the convolution result
        filtered = m.clone();
        // Create a ROI to pass along the vector and compute convolution with the kernel
        Mat roi(signalWithBorder, Rect(0, 0, kernel.cols, 1));
        for (int i = 0; i < m.cols; i++) {
            // Multiply the focused section by the kernel
            Mat r = roi.mul(kernel);
            // Sum the result of the above operation to the pixel at i
            filtered.at<double>(i) = (double) *(sum(r).val);
            // Move the Roi one position to the right
            roi = roi.adjustROI(0, 0, -1, 1);
        }
        filtered.copyTo(m);
    }
    // Merge the vectors into a multichannel Mat
    merge(signalVectorByChannels, filtered);
    return filtered;
}

//////////////////////////////////////////////////////

Mat computeConvolution(Mat &m, double sigma) {

    // TODO only allow C1 or C3  CV_Assert(src.channels() == 3);
    
    Mat result = m.clone();
    // Store type to restore it when the convolution is computed
    int type = result.channels() == 1 ? CV_64F : CV_64FC3;
    // Convert the image to a 64F type, (one or three channels)
    result.convertTo(result, type);
    Mat kernel = myGetGaussianKernel1D(sigma);
    // This kernel is separable, apply convolution for rows and columns separately
    for (int i = 0; i < result.rows; i++) {
        Mat row = result.row(i);
        row = convolutionOperator1D(row, kernel, BORDER_CONSTANT);
        row.copyTo(result.row(i));
    }
    for (int i = 0; i < result.cols; i++) {
        Mat col = result.col(i);
        col = convolutionOperator1D(col, kernel, BORDER_CONSTANT);
        col.copyTo(result.col(i));
    }
    result.convertTo(result, m.type());
    
    return result;
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
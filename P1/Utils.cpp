/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 13, 2015, 11:56 AM
 */

#include "Utils.h"

//////////////////////////////////////////////////////

Mat myGetGaussianKernel1D(double sigma, bool highpass) {

    // Kernel size
    int ksize = 2 * sigma + 1;

    Mat kernel(1, ksize, CV_64F);

    // Compute the mask with the given sigma
    double ssum = 0;
    for (int i = -sigma; i <= sigma; i++) {
        // i+sigma to start at index 0
        int index = i + sigma;
        double gaussian = (double) exp(-.5 * ((i * i) / (sigma * sigma)));
        kernel.at<double>(index) = highpass ? 1 - gaussian : gaussian;
        ssum += kernel.at<double>(index);
    }

    // Normalize the kernel to sum 1, it is a smooth kernel
    kernel = kernel * (1 / ssum);

    return kernel;
}

//////////////////////////////////////////////////////

Mat convolutionOperator1D(Mat &signalVector, Mat &kernel, BorderTypes border) {

    Mat filtered;
    bool was1col = false;

    if (!signalVector.empty() || !kernel.empty()) {
        // If we receive a signalvector with one column, transpose it
        if (signalVector.cols == 1) {
            signalVector = signalVector.t();
            was1col = true;
        }
        int extraBorder = kernel.cols / 2;

        vector<Mat> signalVectorByChannels(signalVector.channels());
        split(signalVector, signalVectorByChannels);


        for (vector<Mat>::const_iterator it = signalVectorByChannels.begin(); it != signalVectorByChannels.end(); ++it) {
            Mat m = *(it);
            // Create a new Mat with the extra borders needed
            Mat signalWithBorder(1, m.cols + extraBorder, m.type());
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
    }

    filtered = was1col ? filtered.t() : filtered;

    return filtered;
}

//////////////////////////////////////////////////////

Mat computeConvolution(Mat &m, double sigma, bool highpass) {

    // TODO only allow C1 or C3  CV_Assert(src.channels() == 3);
    Mat result;

    if (!m.empty()) {
        result = m.clone();
        // Store type to restore it when the convolution is computed
        int type = result.channels() == 1 ? CV_64F : CV_64FC3;
        // Convert the image to a 64F type, (one or three channels)
        result.convertTo(result, type);
        Mat kernel = myGetGaussianKernel1D(sigma, highpass);
        // This kernel is separable, apply convolution for rows and columns separately
        for (int i = 0; i < result.rows; i++) {
            Mat row = result.row(i);
            row = convolutionOperator1D(row, kernel, BORDER_REFLECT);
            row.copyTo(result.row(i));
        }
        for (int i = 0; i < result.cols; i++) {
            Mat col = result.col(i);
            col = convolutionOperator1D(col, kernel, BORDER_REFLECT);
            col.copyTo(result.col(i));
        }
        result.convertTo(result, m.type());
    }
    return result;
}

//////////////////////////////////////////////////////

vector<Mat> hybridImage(Mat &highFreq, Mat &lowFreq, double highSigma, double lowSigma) {

    vector<Mat> result;

    Mat highBlurred = computeConvolution(highFreq, highSigma, true); // high pass filter
    Mat lowBlurred = computeConvolution(lowFreq, lowSigma); // low pass filter

    // Get the high frequencies of the image
    Mat highH = highFreq - highBlurred;

    // Generate the hybrid image
    Mat H = lowBlurred + highH;

    result.push_back(H);
    result.push_back(highH);
    result.push_back(lowBlurred);

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

//////////////////////////////////////////////////////

void drawHybrid(const std::vector<Mat> &m) {
    if (!m.empty()) {
        int height = 0;
        int width = 0;

        // Get the size of the resulting window in which to draw the images
        // The window will be the sum of all width and the height of the greatest image
        for (std::vector<Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            Mat item = (*it);
            width += item.cols;
            if (item.rows > height) {
                height = item.rows;
            }
        }

        // Create a Mat to store all the images
        Mat result(height, width, CV_8UC3);
        // Black background
        result = 0;
        int x = 0;
        for (std::vector<Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            Mat item = (*it);
            // If a image is in grayscale or black and white, convert it to 3 channels 8 bit depth
            if (item.type() != CV_8UC3) {
                cvtColor(item, item, CV_GRAY2RGB);
            }
            Mat roi(result, Rect(x, 0, item.cols, item.rows));
            item.copyTo(roi);
            x += item.cols;
        }
        drawImage(result, "Ventana");
    }
}

//////////////////////////////////////////////////////

void gaussianPyramid(Mat &hybrid, int levels) {
    vector<Mat> pyramid;
    pyramid.push_back(hybrid);
    for (int i = 0; i < levels; i++) {
        Mat r;
        pyrDown(pyramid.at(i), r);
        pyramid.push_back(r);
    }
    drawHybrid(pyramid);
}

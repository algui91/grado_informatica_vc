/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on October 13, 2015, 11:56 AM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

/**
 * Get a 1D Gaussian kernel for the given parameters
 * 
 * @param sigma 
 * @param type Type of Mat
 * @param highpass True if we want a high pass kernel
 * @return The kernel
 */
Mat myGetGaussianKernel1D(double sigma, bool highpass = false);

/**
 * Computes the 1D convolution for the given signal vector, using the kernel
 * 
 * @param signalVector Vector in which apply the convolution.
 * @param kernel The mask to use
 * @param border Only allows BORDER_CONSTANT|BORDER_REFLECT
 * @return 
 */
Mat convolutionOperator1D(Mat &signalVector, Mat &kernel, BorderTypes border = BORDER_CONSTANT);

/**
 * Compute the convolution of an image using the given sigma
 * 
 * @param m The image to which apply convolution
 * @param sigma Sigma to compute the Gaussian kernel
 * @param highpass True if we want a high pass kernel
 * @return An image with the filter applied
 */
Mat computeConvolution(Mat &m, double sigma, bool highpass = false);

/**
 * Computes both high frequency and low frequency with the given sigmas for the
 * two images passed in
 * 
 * @param highFreq The image use as high frequency
 * @param lowFreq The image use as low frequency
 * @param highSigma Sigma value for the high frequency
 * @param lowSigma Sigma value for the low frequency
 * 
 * @return A vector with the hybrid image, low and high freq images
 */
vector<Mat> hybridImage(Mat &highFreq, Mat &lowFreq, double highSigma, double lowSigma);

/**
 * Shows an image in the screen
 * 
 * @param m Image to show
 * @param wn Name of the window
 */
void drawImage(Mat &m, string wn);

/**
 * Draws an hybrid image and the two original images
 * 
 * @param m Vector with images to draw
 */
void drawHybrid(const std::vector<Mat> &m);

/**
 * Draws a Gaussian pyramid with the given levels
 * 
 * @param hybrid Hybrid images
 * @param levels Levels of the pyramid
 */
void gaussianPyramid(Mat &hybrid, int levels);

#endif	/* UTILS_H */


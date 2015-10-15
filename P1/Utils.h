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
 * Computes the two-dimensional convolution of an image with a mask
 * 
 * @param im Input image
 * @param mask Mask to use
 * @param out Output image with the mask applied
 */
void gaussConvolution(Mat &im, Mat &mask, Mat &out);

/**
 * Get a 1D Gaussian kernel for the given parameters
 * 
 * @param sigma 
 * @param type Type of Mat
 * @return The kernel
 */
Mat myGetGaussianKernel1D(double sigma);

/**
 * Computes the 1D convolution for the given signal vector, using the kernel
 * 
 * @param signalVector Vector in which apply the convolution.
 * @param kernel The mask to use
 * @param border Only allows BORDER_CONSTANT|BORDER_REFLECT
 * @return 
 */
Mat convolutionOperator1D(Mat &signalVector, Mat &kernel, BorderTypes border);

/**
 * Compute the convolution of an image using the given sigma
 * 
 * @param m The image to which apply convolution
 * @param sigma Sigma to compute the Gaussian kernel
 * @return An image with the filter applied
 */
Mat computeConvolution(Mat &m, double sigma);

/**
 * Shows an image in the screen
 * 
 * @param m Image to show
 * @param wn Name of the window
 */
void drawImage(Mat &m, string wn);

#endif	/* UTILS_H */


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
 * @return The kernel
 */
Mat myGetGaussianKernel1D(double sigma);

#endif	/* UTILS_H */


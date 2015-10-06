/* 
 * File:   utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com>
 *
 * Created on October 5, 2015, 11:44 PM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <opencv2/opencv.hpp>

#include <opencv2/imgcodecs.hpp>

using namespace cv;

/**
 * Reads an image in GrayScale or color
 * 
 * @param name name Name of the image to read
 * @param flag If set to 0, reads the image in color, otherwise in grayscale, default 0.
 * @return A @ref cv::Mat matrix representing the image
 */
Mat leerImagen(std::string name, int flag);

/**
 * Shows an image in the screen
 * 
 * @param m Image to show
 * @param wn Name of the window
 */
void pintaI(Mat& m, std::string wn);

#endif	/* UTILS_H */


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
 * @param flag If set to 0, reads the image in grayscale, if set to 1 in color
 * @return A @ref cv::Mat matrix representing the image
 */
Mat leerImagen(std::string name, int flag);


#endif	/* UTILS_H */


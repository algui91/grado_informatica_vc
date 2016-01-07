/* 
 * File:   main.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on January 4, 2016, 7:00 PM
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Utils.h"
#include "../P2/Utils.h"

#define _DEBUG 1

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__ << " (" << __LINE__ << "): " << x << std::endl;
#else
#define LOG_MESSAGE(x)
#endif

int main() {

    uint i;

    // Exercise 1
    // a - Estimate a finite camera matrix P from a set of random points in [0,1]
    cv::Mat P = mu::estimatePMatrix();
    // b - Let be a 3D set of points (0,x1,x2) y (x2,x1,0), for x1=0.1:0.1:1 and x2=0.1:0.1:1
    std::vector<double> x1;
    std::vector<double> x2;
    std::vector<cv::Point3f> points;

    x1.push_back(0.1);
    x1.push_back(0.1);
    x1.push_back(1);
    x2.push_back(0.1);
    x2.push_back(0.1);
    x2.push_back(1);

    for (i = 0; i < x1.size(); i++) {
        points.push_back(cv::Point3f(0, x1.at(i), x2.at(i)));
        points.push_back(cv::Point3f(x2.at(i), x1.at(i), 0));
    }

    // Excercise 2 - Camera calibration using homographies


    // Excersise 3
    // 3.a estimate points
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Mat img1, img2;
    img1 = cv::imread("./imagenes/Vmort1.pgm", cv::IMREAD_GRAYSCALE);
    img2 = cv::imread("./imagenes/Vmort2.pgm", cv::IMREAD_GRAYSCALE);
    
    mu::runDetector(img1,
            img2,
            "ORB",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);
    std::vector<cv::DMatch> matchPoints = mu::matching(img1,
            img2,
            "FlannBased",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);

    

    return 0;
}


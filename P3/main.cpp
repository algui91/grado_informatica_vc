/* 
 * File:   main.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on January 4, 2016, 7:00 PM
 */

#include <iostream>

#include <opencv2/core.hpp>

#include "Utils.h"

#define _DEBUG 1
#define _RELEASE 0

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__ << " (" << __LINE__ << "): " << x << std::endl;
#else
#define LOG_MESSAGE(x)
#endif

int main() {


    // Exercise 1
    // a - Estimate a finite camera matrix P from a set of random points in [0,1]
    cv::Mat P = mu::estimatePMatrix();
    LOG_MESSAGE(P)
    return 0;
}


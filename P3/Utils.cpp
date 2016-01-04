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

cv::Mat mu::estimatePMatrix() {
    cv::Mat_<double> P(3, 4);
    cv::Mat_<double> M;
    do {
        cv::randu(P, 0, 1);
        M = P(cv::Rect(0, 0, 3, 3));
    } while (cv::determinant(M) < 0.2f);

    return P;
}
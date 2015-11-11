#include <opencv2/core.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Utils.h"

using namespace cv;
using namespace std;

cv::Mat mu::normalize(cv::Mat_<double>& p) {

    //    cv::Point2d m1c(0,0);
    //    
    //    // compute centers and average distances for each of the two point sets
    //    for(int i = 0; i < p.rows; i++ )
    //    {
    //        double x = p(i,0), y = p(i,1);
    //        m1c.x += x; m1c.y += y;
    //    }
    //    
    //    double t = 1./p.rows;
    //    m1c.x *= t; m1c.y *= t;
    //
    //    double scale1 = 0;
    //    for(int i = 0; i < p.rows; i++ )
    //    {
    //        double x = p(i,0) - m1c.x, y = p(i,1) - m1c.y;
    //        scale1 += std::sqrt(x*x + y*y);
    //    }
    //
    //    scale1 *= t;
    //
    //    scale1 = std::sqrt(2.)/scale1;
    cv::Mat_<double> T(3, 3, 0.0);

    // means
    double xmean = *(cv::mean(p.col(0)).val);
    double ymean = *(cv::mean(p.col(1)).val);

    // Scaling factor
    double s = (std::sqrt(2) * p.rows);
    double d = 0;

    for (int i = 0; i < p.rows; i++) {
        d += std::pow((p(i, 0) - xmean), 2) + std::pow((p(i, 1) - ymean), 2);
    }

    d = std::sqrt(d);
    s /= d;

    T(0) = 1;
    T(1) = 0;
    T(2) = -xmean;
    T(3) = 0;
    T(4) = 1;
    T(5) = -ymean;
    T(6) = 0;
    T(7) = 0;
    T(8) = 1 / s;

    T = s*T;

    for (int i = 0; i < p.rows; i++) {
        p.row(i) = p.row(i) * T;
    }

    return T;
}

cv::Mat mu::dlt(cv::Mat_<double> &p1, cv::Mat_<double> &p2) {

    cv::Mat A(2 * p1.rows, 9, CV_64F);

    double *p = A.ptr<double>(0);

    //Compute A
    for (int i = 0; i < p1.rows; ++i) {
        // TODO: Put matrix from theory

        cv::Point2f pp1(p1(i, 0), p1(i, 1));
        cv::Point2f pp2(p2(i, 0), p2(i, 1));

        // first row
        *p++ = -pp1.x;
        *p++ = -pp1.y;
        *p++ = -1;
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = pp2.x * pp1.x;
        *p++ = pp2.x * pp1.y;
        *p++ = pp2.x;

        // second row
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = -pp1.x;
        *p++ = -pp1.y;
        *p++ = -1;
        *p++ = pp2.y * pp1.x;
        *p++ = pp2.y * pp1.y;
        *p++ = pp2.y;
    }

    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat Hn = svd.vt.row(8).reshape(0, 3); // H12

    return Hn;
}

void mu::runDetector(const std::string &detectorType, cv::Mat &img1, cv::Mat &img2) {
    
    /**
     * Notes on params for BRISK
     * With a threshold of 30, too many points are detected, we've have increased 
     * this number up to 70 an seems to give good results, with octaves between 7 and 8.
     * 
     * This detector seems to detect good match more vertically than ORB
     * 
     * We pick 70,8
     */
    /**
     * Notes on params for ORB
     * ORB seems to work better with 13 octaves, and any number of nfeature, maybe with 1300 better
     * when nfeatures is 500, it is common to have failure matching. 
     * 
     * For this, we think a good param is 13 octaves, and above 1000 nfeatures
     * 
     * We pick 1500, 8
     */
    
    Ptr<Feature2D> detector;
    
    if (detectorType == "BRISK"){
        detector = BRISK::create(70,8);
    } else if (detectorType == "ORB"){
        detector = ORB::create(1500,1.2f, 13);
    }
    
    vector<KeyPoint> keypoints1, keypoints2;

    Mat descriptors1, descriptors2;
    
    detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1, false);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2, false);

    Mat result;    
    drawKeypoints(img1, keypoints1, result);
    
    namedWindow(detectorType, WINDOW_AUTOSIZE);
    imshow(detectorType, result);
    waitKey();
    
    drawKeypoints(img2, keypoints2, result);
    namedWindow(detectorType, WINDOW_AUTOSIZE);
    imshow(detectorType, result);
    waitKey();
}

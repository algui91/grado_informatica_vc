#include <iostream>

#include <opencv2/core.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Utils.h"

#define _DEBUG 1
#define _RELEASE 0

#if _DEBUG
#define LOG_MESSAGE(x) cout << __FILE__ << " (" << __LINE__ << "): " << x << endl;
#else
#define LOG_MESSAGE(x)
#endif

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

cv::Mat mu::dlt(const cv::Mat_<double> &p1, const cv::Mat_<double> &p2) {

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

void mu::runDetector(const std::string &detectorType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2) {

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

    cv::Mat img1 = cv::imread("./imagenes/Yosemite1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("./imagenes/Yosemite2.jpg", cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::Feature2D> detector;

    if (detectorType == "BRISK") {
        detector = cv::BRISK::create(70, 8);
    } else if (detectorType == "ORB") {
        detector = cv::ORB::create(1500, 1.2f, 13);
    }

    detector->detectAndCompute(img1, cv::Mat(), kp1, descriptor1, false);
    detector->detectAndCompute(img2, cv::Mat(), kp2, descriptor2, false);

    if (_RELEASE) {
        cv::Mat result;
        cv::drawKeypoints(img1, kp1, result);

        cv::namedWindow(detectorType, cv::WINDOW_AUTOSIZE);
        cv::imshow(detectorType, result);
        cv::waitKey();

        cv::drawKeypoints(img2, kp2, result);
        cv::namedWindow(detectorType, cv::WINDOW_AUTOSIZE);
        cv::imshow(detectorType, result);
        cv::waitKey();
    }
}

std::vector<cv::DMatch> mu::matching(const std::string &descriptorMatcherType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2) {

    cv::Mat img1 = cv::imread("./imagenes/Yosemite1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("./imagenes/Yosemite2.jpg", cv::IMREAD_GRAYSCALE);

    std::vector<cv::DMatch> goodMatches;

    if (descriptorMatcherType == "BruteForce+Cross") {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptor1, descriptor2, matches);
        goodMatches = mu::goodMatches(matches, 50);
        myDrawMatches(descriptorMatcherType, img1, kp1, img2, kp2, goodMatches);
    } else if (descriptorMatcherType == "FlannBased") {
        // Match between img1 and img2
        std::vector <cv::DMatch> matches;
        cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(10, 10, 2));
        matcher.match(descriptor1, descriptor2, matches);
        goodMatches = mu::goodMatches(matches, 50);
        myDrawMatches(descriptorMatcherType, img1, kp1, img2, kp2, goodMatches);
    }

    return goodMatches;
}

void mu::myDrawMatches(const std::string &descriptorMatcherType, const cv::Mat& img1,
        const std::vector<cv::KeyPoint>& kp1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch> matches) {
    if (_RELEASE) {
        cv::Mat res;
        cv::drawMatches(img1, kp1, img2, kp2, matches, res);
        cv::String name = descriptorMatcherType;
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, res);
        cv::waitKey();
    }
}

const std::vector<cv::DMatch> mu::goodMatches(const std::vector<cv::DMatch> &matches, int size) {

    // Compute what are the best matches for drawing lines point to point
    cv::Mat index;
    int n = int(matches.size());

    // Store the distance attribute of the match for sort them latter
    cv::Mat_<float> distances(n, 1);
    for (int i = 0; i < n; i++) {
        cv::DMatch dm = matches[i];
        distances(i) = dm.distance;
    }
    // Sort the distances ascending
    cv::sortIdx(distances, index, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    std::vector<cv::DMatch> bestMatches;
    // Keep only the 50 best matches to draw lines between them
    for (int i = 0; i < size; i++) {
        bestMatches.push_back(matches[index.at<int>(i, 0)]);
    }

    return bestMatches;
}

void mu::composePanorama(const std::vector<cv::Mat> &images, const std::vector<cv::DMatch> &matchs, 
        const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2) {

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    cv::Mat img1 = images.at(0);
    cv::Mat img2 = images.at(1);

    for (uint i = 0; i < matchs.size(); i++) {
        points1.push_back(kp1[ matchs[i].queryIdx ].pt);
        points2.push_back(kp2[ matchs[i].trainIdx ].pt);
    }

    cv::Mat_<double> cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix(2) = img1.rows; // Where start drawing

    cv::Mat result;
    cv::Size size = cv::Size(img1.cols + img2.cols, img1.rows);
    cv::warpPerspective(img2, result, cameraMatrix, size);

    // Find the Homography Matrix
    cv::Mat H = cv::findHomography(points1, points2, CV_RANSAC, 1);
    H = cameraMatrix * H; // Compose the two homographies
    // Use the Homography Matrix to warp the images
    cv::warpPerspective(img1, result, H, size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    // Remove black borders
    while (*(cv::sum(result.col(0))).val == 0) {
        result = result.colRange(1, result.cols);
    }
    while (*(cv::sum(result.col(result.cols - 1))).val == 0) {
        result = result.colRange(0, result.cols - 1);
    }
    
    cv::imshow("Panorama", result);
    cv::waitKey(0);
}
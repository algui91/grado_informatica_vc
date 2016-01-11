/* 
 * File:   main.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on January 4, 2016, 7:00 PM
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

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

    /***************
     * Excersise 1**
     ***************
     **/

    /********************************************************************************
     * 1.a - Estimate a finite camera matrix P from a set of random points in [0,1] *
     ********************************************************************************/
    cv::Mat P = mu::estimatePMatrix();
    std::cout << "Matrix P " << P << std::endl;
    /******************************************************************************************
     * 1.b - Let be a 3D set of points (0,x1,x2) y (x2,x1,0), for x1=0.1:0.1:1 and x2=0.1:0.1:1 *
     ******************************************************************************************/
    std::vector<cv::Mat_<double> > points3D;

    for (double k = .1; k <= 1; k += .1) {
        for (double k2 = .1; k2 <= 1; k2 += .1) {
            points3D.push_back((cv::Mat_<double>(4, 1) << 0, k, k2, 1));
            points3D.push_back((cv::Mat_<double>(4, 1) << k2, k, 0, 1));
        }
    }

    /*******************************************************************
     * 1.c - Project the world points into pixels using the computed P *
     *******************************************************************/
    std::vector<cv::Mat_<double>> points2D;
    std::cout << "Projected points using P: " << std::endl;
    for (i = 0; i < points3D.size(); i++) {
        points2D.push_back(P * points3D.at(i));
        std::cout << P * points3D.at(i) << std::endl;
    }

    /**
     * 1.d - Estimate P matrix from 3d points and projections using DLT
     */
    cv::Mat P1 = mu::dlt(points3D, points2D);
    LOG_MESSAGE("P calculada");
    LOG_MESSAGE(P);
    return 0;
    // Excercise 2 - Camera calibration using homographies
    /***************
     * Excersise 3**
     ***************
     **/

    /************************
     *  3.a estimate points *
     ************************
     **/

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
    //    std::vector<cv::DMatch> matchPoints = mu::matching(img1,
    //            img2,
    //            "FlannBased",
    //            descriptors1,
    //            descriptors2,
    //            keypoints1,
    //            keypoints2);
    std::vector<std::vector<cv::DMatch> > matches;
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(10, 10, 2), new cv::flann::SearchParams(50));
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    /**
     * ***********************************************
     *  3.b FindFundamentalMat using 8 points RANSAC *
     * ***********************************************
     **/

    std::vector<cv::DMatch> goodMatchs;
    std::vector<cv::Point2f> points1, points2;
    for (i = 0; i < matches.size(); i++) {
        cv::DMatch m, n;
        m = matches.at(i).at(0);
        n = matches.at(i).at(1);
        if (m.distance < 0.8 * n.distance) {
            goodMatchs.push_back(m);
            points2.push_back(keypoints2.at(m.trainIdx).pt);
            points1.push_back(keypoints1.at(m.queryIdx).pt);
        }
    }

    // Leave defaults params for ransac, 3 and .99
    cv::Mat mask;
    cv::Mat F = cv::findFundamentalMat(points1, points2, mask, CV_FM_8POINT + CV_FM_RANSAC);

    /***************************
     * 3.c Draw epipolar lines *
     ***************************
     **/
    cv::Mat lines1, lines2;
    cv::computeCorrespondEpilines(points2, 2, F, lines1);
    cv::Mat epipoLine1 = cv::Mat(2, lines1.size, CV_64FC1);
    std::vector<cv::Mat> imgs56 = mu::drawEpipolarLines(img1, img2, lines1, points1, points2, epipoLine1);

    cv::computeCorrespondEpilines(points1, 1, F, lines2);
    cv::Mat epipoLine2 = cv::Mat(2, lines2.size, CV_64FC1);
    std::vector<cv::Mat> imgs34 = mu::drawEpipolarLines(img2, img1, lines2, points2, points1, epipoLine2);

    std::vector<cv::Mat> img53;
    img53.push_back(imgs56.at(0));
    img53.push_back(imgs34.at(0));

    mu::pintaMI(img53);

    /****************
     * 3.d Verify F *
     ****************
     */
    //    double error = mu::checkF(epipoLine1, epipoLine2, points1, points2);
    //    std::cout << "Error in F: " << error << std::endl;

    /***********************************
     * 4 - Compute the camera movement**
     ***********************************
     **/

    /**
     * 4.a: Read data from files
     */
    std::vector<cv::Mat> K(3), radial(3), R(3), t(3);

    mu::loadFromFile("imagenes/rdimage.000.ppm.camera", K[0], radial[0],
            R[0], t[0]);
    mu::loadFromFile("imagenes/rdimage.001.ppm.camera", K[1], radial[1],
            R[1], t[1]);
    mu::loadFromFile("imagenes/rdimage.004.ppm.camera", K[2], radial[2],
            R[2], t[2]);
    /**
     * 4.b: Compute points in correspondence
     */
    cv::Mat img0 = cv::imread("./imagenes/rdimage.000.ppm", cv::IMREAD_GRAYSCALE);
    img1 = cv::imread("./imagenes/rdimage.001.ppm", cv::IMREAD_GRAYSCALE);
    img2 = cv::imread("./imagenes/rdimage.004.ppm", cv::IMREAD_GRAYSCALE);

    mu::runDetector(img0,
            img1,
            "ORB",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);
    std::vector<cv::DMatch> matchPoints = mu::matching(img0,
            img1,
            "FlannBased",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);

    points1.clear();
    points2.clear();

    for (i = 0; i < matchPoints.size() && i < 1000; i++) {
        points1.push_back(keypoints1[matchPoints.at(i).queryIdx].pt);
        points2.push_back(keypoints2[matchPoints.at(i).trainIdx].pt);
    }

    /**
     * 4.c: Compute E and motion
     */
    // First we need to compute F
    F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT + CV_FM_RANSAC);
    // Now we proceed to compute E
    //    cv::Mat E = K1.t() * F * K0;

    return 0;
}


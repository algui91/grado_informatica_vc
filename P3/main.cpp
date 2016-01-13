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
#include <opencv2/imgproc.hpp>
#include <vector>

#include "Utils.h"

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
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();
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
    std::vector<cv::Matx31d> points2DEstimated;
    std::cout << "Projected points using P: " << std::endl;
    std::cout << "Press enter to show results." << std::endl;
    std::cin.get();
    for (i = 0; i < points3D.size(); i++) {
        cv::Mat_<double> projected = P * points3D.at(i);
        projected /= projected.at<double>(2);
        points2DEstimated.push_back(projected);
        std::cout << points2DEstimated.at(i) << std::endl;
    }
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();

    /**
     * 1.d - Estimate P matrix from 3d points and projections using DLT
     */
    cv::Mat P1 = mu::dlt(points3D, points2DEstimated);

    std::cout << "New matrix P computed from the set of points using DLT." << std::endl;
    std::cout << P1 << std::endl;
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();

    std::cout << "Projected points again but using new P: " << std::endl;
    std::cout << "Press enter to show results." << std::endl;
    std::cin.get();
    std::vector<cv::Matx31d> points2DSimulated;
    for (i = 0; i < points3D.size(); i++) {
        cv::Mat_<double> projected = P1 * points3D.at(i);
        projected /= projected.at<double>(2);
        points2DSimulated.push_back(projected);
        std::cout << projected << std::endl;
    }
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();

    /****************************************************
     * 1.e - Compute error in estimation with frobenius *
     ****************************************************/
    double err = abs(mu::frobenius(P1) - mu::frobenius(P));

    std::cout << "Frobenius Error" << std::endl;
    std::cout << err << std::endl;
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();

    /*******************************************************************
     * 1.f - Show 3D points projected with P estimated and P simulated *
     *******************************************************************/
    cv::Mat image = cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 0));

    for (i = 0; i < points2DEstimated.size(); i++) {
        cv::Point pixelEstimated(points2DEstimated.at(i).val[0] * 100, points2DEstimated.at(i).val[1] * 100);
        cv::Point pixelSimulated(points2DSimulated.at(i).val[0] * 100, points2DSimulated.at(i).val[1] * 100);

        image.at<cv::Vec3b>(pixelEstimated) = cv::Vec3b(0, 255, 0);
        image.at<cv::Vec3b>(pixelSimulated) = cv::Vec3b(0, 0, 255);
    }

    mu::drawImage(image, "Image");

    /*******************************************************
     * Excercise 2 - Camera calibration using homographies *
     *******************************************************/

    /*******************************
     *  2.a Findchessboard corners *
     *******************************/

    std::vector<cv::Mat> chessBoardImages = mu::loadChessboardImages();

    cv::Size patternSize(13, 12); // better than 12,13 or 13,12. More valid images
    std::vector<std::vector<cv::Point2f> > corners;
    bool patternFound[25];

    i = 0;
    for (std::vector<cv::Mat>::iterator it = chessBoardImages.begin(); it != chessBoardImages.end(); it++, i++) {

        std::vector<cv::Point2f> corner;
        patternFound[i] = cv::findChessboardCorners(*it, patternSize, corner,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (patternFound[i]) {
            cv::cornerSubPix(*it, corner, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            cv::cvtColor(*it, *it, CV_GRAY2RGB);
            cv::drawChessboardCorners(*it, patternSize, cv::Mat(corner), patternFound[i]);
            corners.push_back(corner);
        }
    }

    std::vector<cv::Mat> goodChessBoardImages;
    for (i = 0; i < chessBoardImages.size(); i++) {
        if (patternFound[i]) {
            goodChessBoardImages.push_back(chessBoardImages.at(i));
        }
    }
    chessBoardImages.clear();
    mu::pintaMI(goodChessBoardImages);

    /**********************************************
     * 2.b Calibrate camera using the good images *
     **********************************************/
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<std::vector<cv::Point3f> > objectPoints(corners.size());

    for (i = 0; i < corners.size(); i++) {
        std::vector<cv::Point3f> p;
        for (int j = 0; j < patternSize.height; j++) {
            for (int k = 0; k < patternSize.width; k++) {
                p.push_back(cv::Point3f(float(k),
                        float(j), 0));
                LOG_MESSAGE(p.back());
            }
        }
        objectPoints.at(i) = p;
    }

    double rms = cv::calibrateCamera(objectPoints, corners, goodChessBoardImages.at(0).size(), cameraMatrix,
            distCoeffs, rvecs, tvecs, cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);

    std::cout << "Intrinsic camera parameters:" << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << "Press enter to continue" << std::endl;
    std::cin.get();
    std::cout << "Final re projection error:" << std::endl;
    std::cout << rms << std::endl;
    std::cout << "Press enter to continue" << std::endl;
    std::cin.get();
    std::cout << "Distortion Coeff:" << std::endl;
    std::cout << distCoeffs << std::endl;
    std::cout << "Press enter to continue" << std::endl;
    std::cin.get();

    for (i = 0; i < rvecs.size(); i++) {
        cv::Matx34d Rt(
            rvecs.at(i).at<double>(0, 0), rvecs.at(i).at<double>(0, 1), rvecs.at(i).at<double>(0, 2), tvecs.at(i).at<double>(0),
            rvecs.at(i).at<double>(1, 0), rvecs.at(i).at<double>(1, 1), rvecs.at(i).at<double>(1, 2), tvecs.at(i).at<double>(1),
            rvecs.at(i).at<double>(2, 0), rvecs.at(i).at<double>(2, 1), rvecs.at(i).at<double>(2, 2), tvecs.at(i).at<double>(2));
        
        std::cout << "Extrinsic Params for image : " << i << std::endl;
        std::cout << Rt << std::endl;
        std::cout << "Press enter to continue" << std::endl;
        std::cin.get();
    }

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

    std::vector<cv::DMatch> matchPoints = mu::matching(img1,
            img2,
            "FlannBased",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);

    /**
     * ***********************************************
     *  3.b FindFundamentalMat using 8 points RANSAC *
     * ***********************************************
     **/
    std::vector<cv::Point2f> points1, points2;
    for (i = 0; i < matchPoints.size(); i++) {
        points1.push_back(keypoints1.at(matchPoints.at(i).queryIdx).pt);
        points2.push_back(keypoints2.at(matchPoints.at(i).trainIdx).pt);
    }

    // Leave defaults params for ransac, 3 and .99
    cv::Mat mask;
    cv::Mat F = cv::findFundamentalMat(points1, points2, mask, CV_FM_8POINT + CV_FM_RANSAC);

    std::cout << "F Matrix" << std::endl;
    std::cout << F << std::endl;
    std::cout << "Press enter to show next exercise result" << std::endl;
    std::cin.get();

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
    matchPoints = mu::matching(img0,
            img1,
            "FlannBased",
            descriptors1,
            descriptors2,
            keypoints1,
            keypoints2);

    points1.clear();
    points2.clear();

    for (i = 0; i < matchPoints.size() && i < 200; i++) {
        points1.push_back(keypoints1[matchPoints.at(i).queryIdx].pt);
        points2.push_back(keypoints2[matchPoints.at(i).trainIdx].pt);
    }
    /**
     * 4.c: Compute E and motion
     */
    // First we need to compute F
    F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT + CV_FM_RANSAC);
    // Now we proceed to compute E
    cv::Mat E = K[1].t() * F * K[0];
    cv::Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
    cv::SVD svd(E, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    E = svd.u * cv::Mat(W) * svd.vt;
    svd(E, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // Now 4 solutions are posible, only one is feasible
    std::vector<cv::Mat> Rr, tt;
    Rr.push_back(svd.u * cv::Mat(W) * svd.vt); // U W V^T
    Rr.push_back(svd.u * cv::Mat(W).t() * svd.vt); // U W^T V^T
    tt.push_back(svd.u.col(2)); // +u3
    tt.push_back(-svd.u.col(2)); // -u3

    // Search for the feasible solution



    return 0;
}


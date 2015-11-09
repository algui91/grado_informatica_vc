#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main() {

    Mat img1 = imread("./imagenes/Tablero1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("./imagenes/Tablero2.jpg", IMREAD_GRAYSCALE);
    
    //-- Step 1: Detect the keypoints and extract descriptors using SURF
    Mat_<double> p1(10,3);

    p1(0) = 156;   p1(1) = 47;      p1(2) = 1;
    p1(3) = 532;   p1(4) = 13;      p1(5) = 1;
    p1(6) = 137;   p1(7) = 422;     p1(8) = 1;
    p1(9) = 527;   p1(10) = 465;    p1(11) = 1;
    p1(12) = 237;  p1(13) = 139;    p1(14) = 1;
    p1(15) = 416;  p1(16) = 131;    p1(17) = 1;
    p1(18) = 230;  p1(19) = 326;    p1(20) = 1;
    p1(21) = 412;  p1(22) = 335;    p1(23) = 1;
    p1(24) = 146;  p1(25) = 220;    p1(26) = 1;
    p1(27) = 533;  p1(28) = 219;    p1(29) = 1;

    cout << p1 << endl;
    
    Mat_<double> p2(10,3);
    
    p2(0) = 148;   p2(1) = 14;      p2(2)  = 1;
    p2(3) = 503;   p2(4) = 95;      p2(5)  = 1;
    p2(6) = 75;    p2(7) = 387;     p2(8)  = 1;
    p2(9) = 432;   p2(10)= 433;     p2(11) = 1;
    p2(12) = 226;  p2(13) = 134;    p2(14) = 1;
    p2(15) = 395;  p2(16) = 169;    p2(17) = 1;
    p2(18) = 192;  p2(19) = 308;    p2(20) = 1;
    p2(21) = 361;  p2(22) = 337;    p2(23) = 1;
    p2(24) = 111;  p2(25) = 190;    p2(26) = 1;
    p2(27) = 472;  p2(28) = 259;    p2(29) = 1;
    
    
//    Mat_<double> p3(2,3);
//    p3(0,0) = 10;
//    p3(0,1) = 2;
//    p3(0,2) = 1;
//    p3(1,0) = 0;
//    p3(1,1) = 2;
//    p3(1,2) = 1;

    // Normalize points
    Mat_<double> T(3,3);
    // means
    double xmean = *(mean(p1.col(0)).val);
    double ymean = *(mean(p1.col(1)).val);
    // Scaling factor
    double s = (sqrt(2) * 10);
    double d = 0;
    for (int i = 0; i < 10; i++) {
        d += pow((p1(i,0) - xmean), 2) + pow((p1(i,1) - ymean), 2);
    }
    d = sqrt(d);
    s /= d;

    T(0) = 1; 
    T(1) = 0;
    T(2) = - xmean;
    T(3) = 0;
    T(4) = 1;
    T(5) = - ymean;
    T(6) = 0;
    T(7) = 0;
    T(8) = 1/s;
    
    T = s*T;
    
    for (int i = 0; i < 10; i++) {
        p1.row(i) = p1.row(i) * T;
        cout << p1.row(i) << endl;
    }
    
    Mat_<double> T2(3,3);
    // means
    xmean = *(mean(p2.col(0)).val);
    ymean = *(mean(p2.col(1)).val);
    // Scaling factor
    s = (sqrt(2) * 10);
    d = 0;
    for (int i = 0; i < 10; i++) {
        d += pow((p2(i,0) - xmean), 2) + pow((p2(i,1) - ymean), 2);
    }
    d = sqrt(d);
    s /= d;

    T2(0) = 1; 
    T2(1) = 0;
    T2(2) = - xmean;
    T2(3) = 0;
    T2(4) = 1;
    T2(5) = - ymean;
    T2(6) = 0;
    T2(7) = 0;
    T2(8) = 1/s;
    
    T2 = s*T2;
    
    for (int i = 0; i < 10; i++) {
        p2.row(i) = p2.row(i) * T2;
        cout << p2.row(i) << endl;
    }
    
    // Compute A_i
    Mat A(2 * p1.rows, 9, CV_64F);

    double *p = A.ptr<double>(0);

    for (int i = 0; i < p1.rows; ++i) {
        // TODO: Put matrix from theory

        Point2f pp1(p1(i, 0), p1(i, 1));
        Point2f pp2(p2(i, 0), p2(i, 1));

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
    Mat Hn = svd.vt.row(8).reshape(0, 3); // H12
    warpPerspective(img1, img2, T.inv() * Hn * T2, img1.size());

    imshow("Projection", img2);
    imshow("Original", img1);
    waitKey(0);

    // If using findhomography, denormalize points
    //    Mat img3;
    //    Mat H2 = findHomography(p1, p2, RANSAC);
    ////
    //    warpPerspective(img1, img3, H2, img1.size());
    ////    
    //    imshow("Projection1", img3);
    //    imshow("Original1", img1);
    //    waitKey(0);
    //    
    ////    cout << H2 << endl;
    ////    
    //    waitKey(0);
    
    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    
////    
////    
////    
////    
//    Mat img1 = imread("./imagenes/Tablero1.jpg", IMREAD_GRAYSCALE);
//    Mat img2 = imread("./imagenes/Tablero2.jpg", IMREAD_GRAYSCALE);
//
//    //-- Step 1: Detect the keypoints and extract descriptors using SURF
//    std::vector<Point2f> points1;
//    Point2f p1(156, 47);
//    Point2f p2(532, 13);
//    Point2f p3(137, 422);
//    Point2f p4(527, 465);
//    Point2f p5(237, 139);
//    Point2f p6(416,131);
//    Point2f p7(230, 326);
//    Point2f p8(412, 335);
//    Point2f p9(146, 220);
//    Point2f p10(533, 219);
//
//
//    std::vector<Point2f> points2;
//    Point2f q1(148, 14);
//    Point2f q2(503, 95);
//    Point2f q3(75, 387);
//    Point2f q4(432, 433);
//    Point2f q5(226,134);
//    Point2f q6(395,169);
//    Point2f q7(192, 308);
//    Point2f q8(361, 337);
//    Point2f q9(111, 190);
//    Point2f q10(472,259);
//
//    points2.push_back(q1);
//    points2.push_back(q2);
//    points2.push_back(q3);
//    points2.push_back(q4);
//    points2.push_back(q5);
//    points2.push_back(q6);
//    points2.push_back(q7);
//    points2.push_back(q8);
//    points2.push_back(q9);
//    points2.push_back(q10);
//    
//    points1.push_back(p1);
//    points1.push_back(p2);
//    points1.push_back(p3);
//    points1.push_back(p4);
//    points1.push_back(p5);
//    points1.push_back(p6);
//    points1.push_back(p7);
//    points1.push_back(p8);
//    points1.push_back(p9);
//    points1.push_back(p10);
//
//    
//    // Compute A_i
//    Mat A(2 * points1.size(), 9, CV_64F);
//
//    double *p = A.ptr<double>(0);
//
//    for (unsigned int i = 0; i < points1.size(); ++i) {
//        // M12[2*i,   :] = [ p2' 0 0 0 -p1[0]*p2']
//        // M12[2*i+1, :] = [ 0 0 0 p2' -p1[1]*p2']
//
//        Point2f p1 = points1.at(i);
//        Point2f p2 = points2.at(i);
//        
//        // first row
//        *p++ = -p1.x;
//        *p++ = -p1.y;
//        *p++ = -1;
//        *p++ = 0;
//        *p++ = 0;
//        *p++ = 0;
//        *p++ = p2.x * p1.x;
//        *p++ = p2.x * p1.y;
//        *p++ = p2.x;
//
//        // second row
//        *p++ = 0;
//        *p++ = 0;
//        *p++ = 0;
//        *p++ = -p1.x;
//        *p++ = -p1.y;
//        *p++ = -1;
//        *p++ = p2.y * p1.x;
//        *p++ = p2.y * p1.y;
//        *p++ = p2.y;
//    }
//
//    
//    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//
//    Mat H = svd.vt.row(8).reshape(0, 3); // H12
//    
//    cout << H << endl;
//    
//    warpPerspective(img1, img2, H, img1.size());
//    
//    imshow("Projection", img2);
////    waitKey(0);
//    imshow("Original", img1);
//
//    //-- Step 2: Matching descriptor vectors using FLANN matcher
////    Mat img3;
////    Mat H2 = findHomography(points1, points2, RANSAC);
//
////    warpPerspective(img1, img3, H2, img1.size());
//    
////    imshow("Projection1", img3);
////    waitKey(0);
////    imshow("Original1", img1);
//    
////    cout << H2 << endl;
//    
//    waitKey(0);
//    
//    
//    
//    
//    
//    
//    
//    
//    
    
    return 0;
}

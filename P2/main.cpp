#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Utils.h"

using namespace cv;
using namespace std;

int main() {

    Mat img1 = imread("./imagenes/Tablero1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("./imagenes/Tablero2.jpg", IMREAD_GRAYSCALE);
    
    // Manually pick points in the images to stablish correspondences
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

    // Normalize points
    Mat T = mu::normalize(p1);
    Mat T2 = mu::normalize(p2);
        
    // Get a Normalized Homography
    Mat Hn = mu::dlt(p1,p2);
    // Denormalize
    Mat H = T.inv() * Hn * T2;
    
    Mat img3;
    warpPerspective(img1, img3, H, img1.size());
    
    imshow("Original", img1);
    imshow("Projection", img2);
    imshow("WarpPerspectivr", img3);
    waitKey(0);

    // Pick points again, this time very close together
    p1(0) = 139;   p1(1) = 344;     p1(2) = 1;
    p1(3) = 139;   p1(4) = 371;     p1(5) = 1;
    p1(6) = 162;   p1(7) = 346;     p1(8) = 1;
    p1(9) = 161;   p1(10) = 373;    p1(11) = 1;
    p1(12) = 138;  p1(13) = 395;    p1(14) = 1;
    p1(15) = 137;  p1(16) = 422;    p1(17) = 1;
    p1(18) = 159;  p1(19) = 397;    p1(20) = 1;
    p1(21) = 159;  p1(22) = 425;    p1(23) = 1;
    p1(24) = 180;  p1(25) = 400;    p1(26) = 1;
    p1(27) = 180;  p1(28) = 427;    p1(29) = 1;
    
    p2(0) = 88;    p2(1) = 314;     p2(2)  = 1;
    p2(3) = 83;    p2(4) = 339;     p2(5)  = 1;
    p2(6) = 114;   p2(7) = 318;     p2(8)  = 1;
    p2(9) = 109;   p2(10)= 344;     p2(11) = 1;
    p2(12) = 79;   p2(13) = 362;    p2(14) = 1;
    p2(15) = 75;   p2(16) = 388;    p2(17) = 1;
    p2(18) = 104;  p2(19) = 367;    p2(20) = 1;
    p2(21) = 102;  p2(22) = 392;    p2(23) = 1;
    p2(24) = 129;  p2(25) = 371;    p2(26) = 1;
    p2(27) = 124;  p2(28) = 396;    p2(29) = 1;
//    
    // Normalize points
    T = mu::normalize(p1);
    T2 = mu::normalize(p2);
        
    // Get a Normalized Homography
    Hn = mu::dlt(p1,p2);
    // Denormalize
    H = T.inv() * Hn * T2;
    
    warpPerspective(img1, img3, H, img1.size());

    imshow("Original", img1);
    imshow("Projection", img2);
    imshow("WarpPerspectivr", img3);
    waitKey(0);
    
    
    // If using findhomography, denormalize points
//        Mat H2 = findHomography(p1, p2, RANSAC);
//    //
//        warpPerspective(img1, img3, H2, img1.size());
//    //    
//        imshow("Projection1", img3);
//        imshow("Original1", img1);
//        waitKey(0);
    //    
    ////    cout << H2 << endl;
    ////    
    //    waitKey(0);
    
    return 0;
}

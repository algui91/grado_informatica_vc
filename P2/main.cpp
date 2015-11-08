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
    std::vector<Point2f> points1;
    Point2f p1(156, 47);
    Point2f p2(532, 13);
    Point2f p3(137, 422);
    Point2f p4(527, 465);
    Point2f p5(237, 139);
    Point2f p6(416,131);
    Point2f p7(230, 326);
    Point2f p8(412, 335);
    Point2f p9(146, 220);
    Point2f p10(533, 219);

    points1.push_back(p1);
    points1.push_back(p2);
    points1.push_back(p3);
    points1.push_back(p4);
    points1.push_back(p5);
    points1.push_back(p6);
    points1.push_back(p7);
    points1.push_back(p8);
    points1.push_back(p9);
    points1.push_back(p10);

    std::vector<Point2f> points2;
    Point2f q1(148, 14);
    Point2f q2(503, 95);
    Point2f q3(75, 387);
    Point2f q4(432, 433);
    Point2f q5(226,134);
    Point2f q6(395,169);
    Point2f q7(192, 308);
    Point2f q8(361, 337);
    Point2f q9(111, 190);
    Point2f q10(472,259);

    points2.push_back(q1);
    points2.push_back(q2);
    points2.push_back(q3);
    points2.push_back(q4);
    points2.push_back(q5);
    points2.push_back(q6);
    points2.push_back(q7);
    points2.push_back(q8);
    points2.push_back(q9);
    points2.push_back(q10);
    
    // Compute A_i
    Mat A(2 * points1.size(), 9, CV_64F);

    double *p = A.ptr<double>(0);

    for (unsigned int i = 0; i < points1.size(); ++i) {
        // M12[2*i,   :] = [ p2' 0 0 0 -p1[0]*p2']
        // M12[2*i+1, :] = [ 0 0 0 p2' -p1[1]*p2']

        Point2f p1 = points1.at(i);
        Point2f p2 = points2.at(i);
        
        // first row
        *p++ = -p1.x;
        *p++ = -p1.y;
        *p++ = -1;
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = p2.x * p1.x;
        *p++ = p2.x * p1.y;
        *p++ = p2.x;

        // second row
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = -p1.x;
        *p++ = -p1.y;
        *p++ = -1;
        *p++ = p2.y * p1.x;
        *p++ = p2.y * p1.y;
        *p++ = p2.y;
    }

    
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    Mat H = svd.vt.row(8).reshape(0, 3); // H12
    
    cout << H << endl;
    
    warpPerspective(img1, img2, H, img1.size());
    
    imshow("Projection", img2);
//    waitKey(0);
    imshow("Original", img1);

    //-- Step 2: Matching descriptor vectors using FLANN matcher
//    Mat img3;
//    Mat H2 = findHomography(points1, points2, RANSAC);

//    warpPerspective(img1, img3, H2, img1.size());
    
//    imshow("Projection1", img3);
//    waitKey(0);
//    imshow("Original1", img1);
    
//    cout << H2 << endl;
    
    waitKey(0);
    
    
    return 0;
}

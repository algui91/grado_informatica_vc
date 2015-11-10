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
    
//    imshow("Original", img1);
//    imshow("Projection", img2);
//    imshow("WarpPerspectivr", img3);
//    waitKey(0);

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
//
//    imshow("Original", img1);
//    imshow("Projection", img2);
//    imshow("WarpPerspectivr", img3);
//    waitKey(0);
    
   
    // EXERCISE 2

    img1 = imread("./imagenes/Yosemite1.jpg", IMREAD_GRAYSCALE);
    img2 = imread("./imagenes/Yosemite2.jpg", IMREAD_GRAYSCALE);
    
    Ptr<Feature2D> detector = BRISK::create();
    vector<DMatch> matches;
    vector<KeyPoint> keypoints1, keypoints2;

    detector->detect(img1, keypoints1);
    Mat descriptors1, descriptors2;
    detector->compute(img1, keypoints1, descriptors1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2, false);

    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    descriptorMatcher->match(descriptors1, descriptors2, matches, Mat());
    
    // Keep best matches only to have a nice drawing.
    // We sort distance between descriptor matches
    Mat index;
    int nbMatch = int(matches.size());
    Mat tab(nbMatch, 1, CV_32F);
    for (int i = 0; i < nbMatch; i++) {
        DMatch dm = matches[i];
        tab.at<float>(i, 0) = dm.distance;
    }
    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    vector<DMatch> bestMatches;
    for (int i = 0; i < 30; i++) {
        bestMatches.push_back(matches[index.at<int>(i, 0)]);
    }
    
    Mat result;
    drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, result);
    namedWindow("BRISK:FlannBased", WINDOW_AUTOSIZE);
    imshow("BRISK:FlannBased", result);
    waitKey();
    
//    vector<String> typeDesc;
//    vector<String> typeAlgoMatch;
//    vector<String> fileName;
//    // This descriptor are going to be detect and compute
//    typeDesc.push_back("ORB"); // see http://docs.opencv.org/trunk/de/dbf/classcv_1_1BRISK.html
//    typeDesc.push_back("BRISK"); // see http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html
//
//    // This algorithm would be used to match descriptors see http://docs.opencv.org/trunk/db/d39/classcv_1_1DescriptorMatcher.html#ab5dc5036569ecc8d47565007fa518257
//    typeAlgoMatch.push_back("BruteForce");
//    typeAlgoMatch.push_back("BruteForce-L1");
//    typeAlgoMatch.push_back("BruteForce-Hamming");
//    typeAlgoMatch.push_back("BruteForce-Hamming(2)");
//    fileName.push_back("./imagenes/Yosemite1.jpg");
//    fileName.push_back("./imagenes/Yosemite2.jpg");
//
//    img1 = imread(fileName[0], IMREAD_GRAYSCALE);
//    img2 = imread(fileName[1], IMREAD_GRAYSCALE);
//
//    if (img1.rows * img1.cols <= 0) {
//        cout << "Image " << fileName[0] << " is empty or cannot be found\n";
//        return (0);
//    }
//    if (img2.rows * img2.cols <= 0) {
//        cout << "Image " << fileName[1] << " is empty or cannot be found\n";
//        return (0);
//    }
//
//    vector<double> desMethCmp;
//    Ptr<Feature2D> b;
//
//    // Descriptor loop
//    vector<String>::iterator itDesc;
//    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++) {
//        Ptr<DescriptorMatcher> descriptorMatcher;
//        // Match between img1 and img2
//        vector<DMatch> matches;
//        // keypoint  for img1 and img2
//        vector<KeyPoint> keyImg1, keyImg2;
//        // Descriptor for img1 and img2
//        Mat descImg1, descImg2;
//        vector<String>::iterator itMatcher = typeAlgoMatch.end();
//        if (*itDesc == "ORB") {
//            b = ORB::create();
//        } else if (*itDesc == "BRISK") {
//            b = BRISK::create();
//        }
//        try {
//            // We can detect keypoint with detect method
//            b->detect(img1, keyImg1, Mat());
//            // and compute their descriptors with method  compute
//            b->compute(img1, keyImg1, descImg1);
//            // or detect and compute descriptors in one step
//            b->detectAndCompute(img2, Mat(), keyImg2, descImg2, false);
//            // Match method loop
//            for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++) {
//                descriptorMatcher = DescriptorMatcher::create(*itMatcher);
//                if ((*itMatcher == "BruteForce-Hamming" || *itMatcher == "BruteForce-Hamming(2)") && (b->descriptorType() == CV_32F || b->defaultNorm() <= NORM_L2SQR)) {
//                    cout << "**************************************************************************\n";
//                    cout << "It's strange. You should use Hamming distance only for a binary descriptor\n";
//                    cout << "**************************************************************************\n";
//                }
//                if ((*itMatcher == "BruteForce" || *itMatcher == "BruteForce-L1") && (b->defaultNorm() >= NORM_HAMMING)) {
//                    cout << "**************************************************************************\n";
//                    cout << "It's strange. You shouldn't use L1 or L2 distance for a binary descriptor\n";
//                    cout << "**************************************************************************\n";
//                }
//                try {
//                    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
//                    // Keep best matches only to have a nice drawing.
//                    // We sort distance between descriptor matches
//                    Mat index;
//                    int nbMatch = int(matches.size());
//                    Mat tab(nbMatch, 1, CV_32F);
//                    for (int i = 0; i < nbMatch; i++) {
//                        DMatch dm = matches[i];
//                        tab.at<float>(i, 0) = dm.distance;
//                    }
//                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
//                    vector<DMatch> bestMatches;
//                    for (int i = 0; i < 30; i++) {
//                        bestMatches.push_back(matches[index.at<int>(i, 0)]);
//                    }
//                    Mat result;
//                    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
//                    namedWindow(*itDesc + ": " + *itMatcher, WINDOW_AUTOSIZE);
//                    imshow(*itDesc + ": " + *itMatcher, result);
//
//                    vector<DMatch>::iterator it;
//                    cout << "**********Match results**********\n";
//                    cout << "Index \tIndex \tdistance\n";
//                    cout << "in img1\tin img2\n";
//                    // Use to compute distance between keyPoint matches and to evaluate match algorithm
//                    double cumSumDist2 = 0;
//                    for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
//                        DMatch dm = *(it);
//                        cout << dm.queryIdx << "\t" << dm.trainIdx << "\t" << dm.distance << "\n";
//                        KeyPoint s = keyImg1[dm.queryIdx];
//                        KeyPoint s2 = keyImg2[dm.trainIdx];
//                        Point2d p = s.pt - s2.pt;
//                        cumSumDist2 = p.x * p.x + p.y * p.y;
//                    }
//                    desMethCmp.push_back(cumSumDist2);
//                    waitKey();
//                } catch (Exception& e) {
//                    cout << e.msg << endl;
//                    cout << "Cumulative distance cannot be computed." << endl;
//                    desMethCmp.push_back(-1);
//                }
//            }
//        } catch (Exception& e) {
//            cout << "Feature : " << *itDesc << "\n";
//            if (itMatcher != typeAlgoMatch.end()) {
//                cout << "Matcher : " << *itMatcher << "\n";
//            }
//            cout << e.msg << endl;
//        }
//    }
//    int i = 0;
//    cout << "Cumulative distance between keypoint match for different algorithm and feature detector \n\t";
//    cout << "We cannot say which is the best but we can say results are differents! \n\t";
//    for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++) {
//        cout << *itMatcher << "\t";
//    }
//    cout << "\n";
//    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++) {
//        cout << *itDesc << "\t";
//        for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++, i++) {
//            cout << desMethCmp[i] << "\t";
//        }
//        cout << "\n";
//    }
    
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

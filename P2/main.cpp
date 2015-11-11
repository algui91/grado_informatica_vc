#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Utils.h"

using namespace cv;
using namespace std;


#define _DEBUG 1

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__ << " (" << __LINE__ << "): " << x << std::endl;
#else
#define LOG_MESSAGE(x)
#endif

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

//    img1 = imread("./imagenes/Yosemite1.jpg", IMREAD_GRAYSCALE);
//    img2 = imread("./imagenes/Yosemite2.jpg", IMREAD_GRAYSCALE);

//    
//    Ptr<Feature2D> detector = BRISK::create();
//    
//    vector<DMatch> matches;
//    vector<KeyPoint> keypoints1, keypoints2;
//
//    detector->detect(img1, keypoints1);
//    Mat descriptors1, descriptors2;
//    detector->compute(img1, keypoints1, descriptors1);
//    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2, false);
//
//    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
//    
//    descriptorMatcher->match(descriptors1, descriptors2, matches, Mat());
//    
//    // Keep best matches only to have a nice drawing.
//    // We sort distance between descriptor matches
//    Mat index;
//    int nbMatch = int(matches.size());
//    Mat tab(nbMatch, 1, CV_32F);
//    for (int i = 0; i < nbMatch; i++) {
//        DMatch dm = matches[i];
//        tab.at<float>(i, 0) = dm.distance;
//    }
//    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
//    vector<DMatch> bestMatches;
//    for (int i = 0; i < 30; i++) {
//        bestMatches.push_back(matches[index.at<int>(i, 0)]);
//    }
//    
//    Mat result;
//    drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, result);
//    namedWindow("BRISK:FlannBased", WINDOW_AUTOSIZE);
//    imshow("BRISK:FlannBased", result);
//    waitKey();

    vector<String> detectorsTypes;
    vector<int> briskThresholdParams;
    vector<int> briskOctavesParams;
    vector<int> orbNFeaturesParams;
    vector<int> orbnLevelsParams;
    
    // This descriptor are going to be detect and compute
    detectorsTypes.push_back("BRISK");
//    detectorsTypes.push_back("ORB");

    
    /**
     * Notes on params
     * With a threshold of 30, too many points are detected, we've have increased 
     * this number up to 70 an seems to give good results, with octaves between 7 and 8.
     * 
     * This detector seems to detect good match more vertically than ORB
     * @return 
     */
    briskThresholdParams.push_back(10);
    briskThresholdParams.push_back(20);
    briskThresholdParams.push_back(25);
    briskThresholdParams.push_back(30);
    briskThresholdParams.push_back(35);
    briskThresholdParams.push_back(40);
    briskThresholdParams.push_back(45);
    briskThresholdParams.push_back(50);
    briskThresholdParams.push_back(70);
    briskOctavesParams.push_back(3);
    briskOctavesParams.push_back(4);
    briskOctavesParams.push_back(5);
    briskOctavesParams.push_back(6);
    briskOctavesParams.push_back(7);
    briskOctavesParams.push_back(8);
    
    
    /**
     * Notes on params
     * ORB seems to work better with 13 octaves, and any number of nfeature, maybe with 1300 better
     * when nfeatures is 500, it is common to have failure matching. 
     * 
     * For this, we think a good param is 13 octaves, and above 1000 nfeatures
     * @return 
     */
    orbNFeaturesParams.push_back(500);
    orbNFeaturesParams.push_back(700);
    orbNFeaturesParams.push_back(900);
    orbNFeaturesParams.push_back(1100);
    orbNFeaturesParams.push_back(1300);
    orbNFeaturesParams.push_back(1500);
    orbnLevelsParams.push_back(8);
    orbnLevelsParams.push_back(9);
    orbnLevelsParams.push_back(10);
    orbnLevelsParams.push_back(11);
    orbnLevelsParams.push_back(12);
    orbnLevelsParams.push_back(13);
    // 1500 13 good

    img1 = imread("./imagenes/Yosemite1.jpg", IMREAD_GRAYSCALE);
    img2 = imread("./imagenes/Yosemite2.jpg", IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        cout << "Could not load images.\n";
        return 0;
    }

    Ptr<Feature2D> detector;

    vector<String>::iterator itDetectorsTypes;
    
    // Iterate over all detectors specified 
    for (itDetectorsTypes = detectorsTypes.begin(); itDetectorsTypes != detectorsTypes.end(); itDetectorsTypes++) {
        Ptr<DescriptorMatcher> descriptorMatcher;
        // Match between img1 and img2
        vector<DMatch> matches;
        // keypoint  for img1 and img2
        vector<KeyPoint> keyImg1, keyImg2;
        // Descriptor for img1 and img2
        Mat descImg1, descImg2;

        vector<int>::iterator itParam1;
        vector<int>::iterator itParam1end;
        vector<int>::iterator itParam2;
        vector<int>::iterator itParam2end;

        if (*itDetectorsTypes == "ORB") {
            itParam1 = orbNFeaturesParams.begin();
            itParam1end = orbNFeaturesParams.end();
            itParam2 = orbnLevelsParams.begin();
            itParam2end = orbnLevelsParams.end();
        } else if (*itDetectorsTypes == "BRISK") {
            itParam1 = briskThresholdParams.begin();
            itParam1end = briskThresholdParams.end();
            itParam2 = briskOctavesParams.begin();
            itParam2end = briskOctavesParams.end();
        }

        vector<int>::iterator reset = itParam1;

        for (; itParam2 != itParam2end; itParam2++) {
            itParam1 = reset;
            for (; itParam1 != itParam1end; itParam1++) {

                if (*itDetectorsTypes == "ORB") {
                    detector = ORB::create(*itParam1, 1.2, *itParam2);
                } else if (*itDetectorsTypes == "BRISK") {
                    detector = BRISK::create(*itParam1, *itParam2);
                }

                // Detect keypoints for images
                detector->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
                detector->detectAndCompute(img2, Mat(), keyImg2, descImg2, false);

                descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
                descriptorMatcher->match(descImg1, descImg2, matches, Mat());
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
                for (int i = 0; i < 50; i++) {
                    bestMatches.push_back(matches[index.at<int>(i, 0)]);
                }
                Mat result;

                drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
                String name = *itDetectorsTypes + ": BruteForce-Hamming(2) thr: " + to_string(*itParam1) + " o:" + to_string(*itParam2);
//                namedWindow(name, WINDOW_AUTOSIZE);
//                imshow(name, result);
                imwrite(name+".jpg", result);
                vector<DMatch>::iterator it;
                cout << "**********Match results**********\n";
                cout << "Index \tIndex \tdistance\n";
                cout << "img1 \timg2\n";
                // Use to compute distance between keyPoint matches and to evaluate match algorithm
//                for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
//                    DMatch dm = *(it);
//                    cout << dm.queryIdx << "\t" << dm.trainIdx << "\t" << dm.distance << "\n";
//                }
//                waitKey();

            }
        }
    }
    
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

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


#include "Utils.h"


#define _DEBUG 1

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

void mu::runDetector(const cv::Mat &img1, const cv::Mat &img2, const std::string &detectorType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2) {

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

    cv::Ptr<cv::Feature2D> detector;

    if (detectorType == "BRISK") {
        detector = cv::BRISK::create(70, 8);
    } else if (detectorType == "ORB") {
        detector = cv::ORB::create(1500, 1.2f, 13);
    }

    detector->detectAndCompute(img1, cv::Mat(), kp1, descriptor1, false);
    detector->detectAndCompute(img2, cv::Mat(), kp2, descriptor2, false);

    if (_DEBUG) {
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

std::vector<cv::DMatch> mu::matching(const cv::Mat &img1, const cv::Mat &img2, const std::string &descriptorMatcherType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2) {

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
    if (_DEBUG) {
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
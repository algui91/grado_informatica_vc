#include <iostream>
#include <fstream>

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
    std::vector<cv::DMatch> matches;

    if (descriptorMatcherType == "BruteForce+Cross") {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        matcher.match(descriptor1, descriptor2, matches);
        goodMatches = mu::goodMatches(matches, 50);
        myDrawMatches(descriptorMatcherType, img1, kp1, img2, kp2, goodMatches);
    } else if (descriptorMatcherType == "FlannBased") {
        // Match between img1 and img2
        //        std::vector<std::vector<cv::DMatch> > matches;
        cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(10, 10, 2), new cv::flann::SearchParams(50));
        //        matcher.knnMatch(descriptor1, descriptor2, matches, 2);
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

std::vector<cv::Mat> mu::drawEpipolarLines(cv::Mat &img1, cv::Mat &img2,
        const cv::Mat &lines, const std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
        cv::Mat &epipoleLine) {

    if (img1.type() != CV_8UC3) {
        cv::cvtColor(img1, img1, CV_GRAY2RGB);
    }
    if (img2.type() != CV_8UC3) {
        cv::cvtColor(img2, img2, CV_GRAY2RGB);
    }
    std::vector<cv::Point2f>::const_iterator it1 = p1.begin();
    std::vector<cv::Point2f>::const_iterator it2 = p2.begin();

    for (int i = 0; it2 != p2.end() && it1 != p1.end() && i < 20 && i < lines.rows; ++it1, ++it2, i++) {
        //        cv::Point2f item1 = (*it1);
        //        cv::Point2f item2 = (*it2);

        cv::Point x = cv::Point(0, -lines.at<cv::Vec3f>(i).val[2] / lines.at<cv::Vec3f>(i).val[1]);
        cv::Point y = cv::Point(img1.cols, -(lines.at<cv::Vec3f>(i).val[2] + lines.at<cv::Vec3f>(i).val[0] * img1.cols)
                / lines.at<cv::Vec3f>(i).val[1]);

        cv::RNG& rng = cv::theRNG();
        cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));

        cv::line(img1, x, y, color);

        epipoleLine.at<cv::Point>(0, i) = x;
        epipoleLine.at<cv::Point>(1, i) = y;
    }

    std::vector<cv::Mat> images;
    images.push_back(img1);
    images.push_back(img2);

    return images;
}

void mu::pintaMI(const std::vector<cv::Mat> &m) {
    if (!m.empty()) {
        int height = 0;
        int width = 0;

        // Get the size of the resulting window in which to draw the images
        // The window will be the sum of all width and the height of the greatest image
        for (std::vector<cv::Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            cv::Mat item = (*it);
            width += item.cols;
            if (item.rows > height) {
                height = item.rows;
            }
        }

        // Create a Mat to store all the images
        cv::Mat result(height, width, CV_8UC3);

        int x = 0;
        for (std::vector<cv::Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            cv::Mat item = (*it);
            // If a image is in grayscale or black and white, convert it to 3 channels 8 bit depth
            if (item.type() != CV_8UC3) {
                cv::cvtColor(item, item, CV_GRAY2RGB);
            }
            cv::Mat roi(result, cv::Rect(x, 0, item.cols, item.rows));
            item.copyTo(roi);
            x += item.cols;
        }

        cv::namedWindow("Epipolar Lines", cv::WINDOW_AUTOSIZE);
        cv::imshow("Epipolar Lines", result);
        cv::waitKey(0);
        cv::destroyWindow("Epipolar Lines");
    }
}

double mu::checkF(const cv::Mat &lines1, const cv::Mat &lines2,
        const std::vector<cv::Point2f> &p1, const std::vector<cv::Point2f> &p2) {

    double error1 = 0.0;
    for (int i = 0; i < lines1.rows; i++) {
        error1 += distance(lines1.at<cv::Point>(0, i),
                lines1.at<cv::Point>(1, i), p1[i]);
    }
    error1 /= lines1.rows;

    double error2 = 0.0;
    for (int i = 0; i < lines2.rows; i++) {
        error2 += distance(lines2.at<cv::Point>(0, i),
                lines2.at<cv::Point>(1, i), p2[i]);
    }
    error2 /= lines2.rows;

    return (error1 + error2) / 2;
}

double mu::distance(cv::Point p1, cv::Point p2, cv::Point x) {
    p2 -= p1;
    x -= p1;
    double A = x.cross(p2);

    return A / cv::norm(p2);
}

void mu::string2double(std::string string, std::vector<double> &numbers) {
    std::stringstream stream(string);

    numbers.clear();

    while (stream) {
        double n;
        stream >> n;
        numbers.push_back(n);
    }

}

bool mu::loadFromFile(const std::string file_name, cv::Mat &K, cv::Mat &radial, cv::Mat &R,
        cv::Mat &t) {

    std::string current_line;
    std::vector<double> numbers;
    std::ifstream f(file_name);

    if (f.is_open()) {
        // (3x3) camera matrix K
        K = cv::Mat(3, 3, CV_64F);

        for (int k = 0; k < 3; k++) {
            getline(f, current_line);
            string2double(current_line, numbers);

            K.at<double>(k, 0) = numbers[0];
            K.at<double>(k, 1) = numbers[1];
            K.at<double>(k, 2) = numbers[2];
        }

        // (3) radial distortion parameters
        radial = cv::Mat(1, 3, CV_64F);
        getline(f, current_line);
        string2double(current_line, numbers);

        radial.at<double>(0, 0) = numbers[0];
        radial.at<double>(0, 1) = numbers[1];
        radial.at<double>(0, 2) = numbers[2];

        // (3x3) rotation matrix R
        R = cv::Mat(3, 3, CV_64F);

        for (int k = 0; k < 3; k++) {
            getline(f, current_line);
            string2double(current_line, numbers);

            R.at<double>(k, 0) = numbers[0];
            R.at<double>(k, 1) = numbers[1];
            R.at<double>(k, 2) = numbers[2];
        }

        // (3) translation vector t
        t = cv::Mat(1, 3, CV_64F);
        getline(f, current_line);
        string2double(current_line, numbers);

        t.at<double>(0, 0) = numbers[0];
        t.at<double>(0, 1) = numbers[1];
        t.at<double>(0, 2) = numbers[2];

        f.close();

        return true;
    }

    return false;
}

cv::Mat mu::dlt(const std::vector<cv::Mat_<double> > &points3D, const std::vector<cv::Mat_<double> > &points2D) {

    cv::Mat A(2 * points3D.size(), 12, CV_64F);

    double *p = A.ptr<double>(0);

    //Compute A
    for (uint i = 0; i < points3D.size(); ++i) {
        // TODO: Put matrix from theory

        cv::Point3f p3D(points3D.at(i).at<double>(i,0), points3D.at(i).at<double>(i,1), points3D.at(i).at<double>(i,2));
        cv::Point2f p2D(points2D.at(i).at<double>(i, 0), points2D.at(i).at<double>(i, 1));

        // first row
        *p++ = p3D.x;
        *p++ = p3D.y;
        *p++ = p3D.z;
        *p++ = 1;
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = -(p2D.x * p3D.x);
        *p++ = -(p2D.x * p3D.y);
        *p++ = -(p2D.x * p3D.z);
        *p++ = -p2D.x;

        // second row
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = 0;
        *p++ = p3D.x;
        *p++ = p3D.y;
        *p++ = p3D.z;
        *p++ = 1;
        *p++ = -(p2D.y * p3D.x);
        *p++ = -(p2D.y * p3D.y);
        *p++ = -(p2D.y * p3D.z);
        *p++ = -p2D.y;
    }

    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat Hn = svd.vt.row(11).reshape(0, 3); // H12

    return Hn;
}
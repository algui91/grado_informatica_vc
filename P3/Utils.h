/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on January 4, 2016, 7:06 PM
 */

#ifndef UTILS_H
#define	UTILS_H

namespace mu {
    /**
     * Estimate a finite Camera matrix P
     * @return The Camera Matrix
     */
    cv::Mat estimatePMatrix();

    /**
     * Run the specified detector over two images
     * 
     * @param img1 Input param, image 1
     * @param img2 Input param, image 2
     * @param detectorType Type of detector to use ORB|BRISK
     * @param descriptor1 Out param, descriptors for image 1
     * @param descriptor2 Out param, descriptors for image 2
     * @param kp1 Out param, keypoints for image 1
     * @param kp2 Out para, keypoints for image 2
     */
    void runDetector(const cv::Mat &img1,
            const cv::Mat &img2,
            const std::string &detectorType,
            cv::Mat &descriptor1,
            cv::Mat &descriptor2,
            std::vector<cv::KeyPoint> &kp1,
            std::vector<cv::KeyPoint> &kp2);
    /**
     * Performs the matching between two images
     * 
     * @param img1 Input param, image 1
     * @param img2 Input param, image 2
     * @param descriptorMatcherType What kind of matcher run, BruteForce+Cross|FlannBased
     * @param descriptor1 Descriptor for image 1
     * @param descriptor2 Descriptor for image 2
     * @param kp1 Keypoints for image 1
     * @param kp2 Keypoints for image 2
     * 
     * @return A Vector of Dmatch containing the matches between the two images
     */
    std::vector<cv::DMatch> matching(const cv::Mat &img1,
            const cv::Mat &img2,
            const std::string &descriptorMatcherType,
            cv::Mat &descriptor1,
            cv::Mat &descriptor2,
            std::vector<cv::KeyPoint> &kp1,
            std::vector<cv::KeyPoint> &kp2);

    /**
     * A wrapped function of DrawMatches for ease of use.
     * 
     * @param descriptorMatcherType The name of the window
     * @param img1 Image 1
     * @param kp1 Keypoints for image 1
     * @param img2 Image 2
     * @param kp2 Keypoints for image 2
     * @param matches A vector of correspondences points
     */
    void myDrawMatches(const std::string &descriptorMatcherType,
            const cv::Mat &img1,
            const std::vector<cv::KeyPoint> &kp1,
            const cv::Mat &img2,
            const std::vector<cv::KeyPoint> &kp2,
            const std::vector<cv::DMatch> matches);

    /**
     * Computes the size best matches 
     * 
     * @param matches Vector of points in correspondence
     * @param size How many of them we want
     * 
     * @return A vector of the size best matches
     */
    const std::vector<cv::DMatch> goodMatches(const std::vector<cv::DMatch> &matches,
            int size);

    void drawEpipolarLines(std::vector<cv::Mat>& images, const std::vector<std::vector<cv::Point2f> >& points, const cv::Mat& F);

    /**
     * Shows a list of images
     * @param m list of images to show. If they are of different types, all are 
     * converted to color (CV_8UC3)
     */
    void pintaMI(const std::vector<cv::Mat> &m);

}
#endif	/* UTILS_H */


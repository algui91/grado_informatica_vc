/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on November 9, 2015, 8:24 PM
 */

#ifndef UTILS_H
#define	UTILS_H

namespace mu {

    /**
     * Normalize the points using the Matrix T:
     *  T = s   \begin{pmatrix}
     *              1 & 0 & -\bar{u}\\ 
     *              0 & 1 & -\bar{v}\\ 
     *              0 & 0 & \frac{1}{s}
     *           \end{pmatrix}
     * 
     * Where $s$ is s = \frac{\sqrt{2}n}{\sum_{i=1}^n[ (u_i - \bar{u})^2 + (v_i - \bar{v})^2]^{\frac{1}{2}} }
     * 
     * @param p1 Set of points to normalize
     * 
     * @return A normalization matrix T
     */
    cv::Mat normalize(cv::Mat_<double> &p1);

    /**
     * A Direct Linear Transformation implentation.
     * 
     * The points passed in as parameters must be normalized.
     * 
     * It computes Ah = 0 where A is:
     * 
     * \begin{pmatrix}
     *   -x & -y & -1 & 0 & 0 & 0 & ux & uy & u\\ 
     *   0 & 0 & 0 & -x & -y & -1 & vx & vy & v
     * \end{pmatrix}
     * 
     * It will be as many A_i as points
     * 
     * @param p1 Set of points of image 1
     * @param p2 Points in image2 that corresponds to the points in image 1
     * 
     * @return A normalized transformation matrix H
     */
    cv::Mat dlt(const cv::Mat_<double> &p1, const cv::Mat_<double> &p2);

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
    
    /**
     * Stich two images in one panorama
     * 
     * @param images A vector with the two images
     * @param matchs Vector of matches between the two images
     * @param kp1 Keypoints for image 1
     * @param kp2 Keypoints for image 2
     */
    void composePanorama(const std::vector<cv::Mat> &images, 
            const std::vector<cv::DMatch> &matchs, 
            const std::vector<cv::KeyPoint> &kp1, 
            const std::vector<cv::KeyPoint> &kp2);
    
    void composePanorama(const std::vector<cv::Mat> &images);
}

#endif	/* UTILS_H */
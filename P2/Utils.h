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
    cv::Mat dlt(cv::Mat_<double> &p1, cv::Mat_<double> &p2);
    
    void runDetector(const std::string &detectorType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2);
    void matching(const std::string &descriptorMatcherType, cv::Mat &descriptor1, cv::Mat &descriptor2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2);
    void myDrawMatches(const std::string &descriptorMatcherType, cv::Mat &img1, std::vector<cv::KeyPoint> &kp1, cv::Mat &img2, std::vector<cv::KeyPoint> &kp2, std::vector<cv::DMatch> matches);
    std::vector<cv::DMatch> goodMatches(std::vector<cv::DMatch> &matches, int size);
   
}

#endif	/* UTILS_H */
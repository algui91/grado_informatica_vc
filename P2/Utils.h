/* 
 * File:   Utils.h
 * Author: Alejandro Alcalde <algui91@gmail.com> (elbauldelprogramador.com)
 *
 * Created on November 9, 2015, 8:24 PM
 */

#ifndef UTILS_H
#define	UTILS_H

namespace mu {

    cv::Mat normalize(cv::Mat_<double> &p1);
    cv::Mat dlt(cv::Mat_<double> &p1, cv::Mat_<double> &p2);
}

#endif	/* UTILS_H */
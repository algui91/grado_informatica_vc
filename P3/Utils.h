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
}
#endif	/* UTILS_H */


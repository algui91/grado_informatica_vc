/* 
 * File:   p0.cpp
 * Author: Alejandro Alcalde <algui91@gmail.com>
 *
 * Created on October 5, 2015, 11:38 PM
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <strings.h>

#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    Mat a = leerImagen("lena.jpg", 0);
    modifyPoints(a);
    pintaI(a, "MiVentana");

    Mat ball = imread("ball.png", IMREAD_UNCHANGED);
    Mat ball2 = imread("ball.png", IMREAD_UNCHANGED);
    Mat lena = imread("lena.jpg", IMREAD_UNCHANGED);
    Mat lena2 = imread("lena.jpg", IMREAD_UNCHANGED);

    vector<Mat> vec;
    vec.push_back(ball);
    vec.push_back(ball2);
    vec.push_back(lena);
    vec.push_back(lena2);
    
    modifyPoints(lena);
//    modifyPoints(ball);
            
    pintaMI(vec);

    return 0;
}
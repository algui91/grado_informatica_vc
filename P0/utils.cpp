#include "utils.h"

#include <opencv2/imgcodecs.hpp>

Mat leerImagen(std::string name, int flag) {
    return imread(name, flag ? IMREAD_GRAYSCALE : IMREAD_COLOR);
}

void pintaI(Mat &m, std::string windowName){
    if (!m.empty()){
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, m);
        waitKey(0);
        destroyWindow(windowName);
    }
}

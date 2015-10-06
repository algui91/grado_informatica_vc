#include "utils.h"

#include <opencv2/imgcodecs.hpp>

Mat leerImagen(std::string name, int flag) {
    return imread(name, flag ? IMREAD_GRAYSCALE : IMREAD_COLOR);
}

void pintaI(Mat &m, std::string windowName) {
    if (!m.empty()) {
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, m);
        waitKey(0);
        destroyWindow(windowName);
    }
}

void pintaMI(const std::vector<Mat> &m) {
    if (!m.empty()) {
        int height = 0;
        int width = 0;

        // Get the size of the resulting window in which to draw the images
        // The window will be the sum of all width and the height of the greatest image
        for (std::vector<Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            Mat item = (*it);
            width += item.cols;
            if (item.rows > height) {
                height = item.rows;
            }
        }

        // Create a Mat to store all the images
        Mat result(height, width, CV_8UC3);

        int x = 0;
        for (std::vector<Mat>::const_iterator it = m.begin(); it != m.end(); ++it) {
            Mat item = (*it);
            // If a image is in grayscale or black and white, convert it to 3 channels 8 bit depth
            if (item.type() != CV_8UC3) {
                cvtColor(item, item, CV_GRAY2RGB);
            }
            Mat roi(result, Rect(x, 0, item.cols, item.rows));
            item.copyTo(roi);
            x += item.cols;
        }
        pintaI(result, "Ventana");
    }
}

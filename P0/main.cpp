#include <opencv2/opencv.hpp>

#include <opencv2/imgcodecs.hpp>
#include <strings.h>

using namespace std;
using namespace cv;
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    cout << "OpenCV detectada " << endl;

    // Read and show in a window lena
    Mat im = imread("./lena.jpg", 1);
    namedWindow("ventana", WINDOW_AUTOSIZE);
    imshow("ventana", im);
    waitKey(0);
    destroyWindow("ventana");

    cout << "Information of im:" << endl;
    cout << "\t Rows: " << im.rows << endl;
    cout << "\t Cols: " << im.cols << endl;
    cout << "\t Channels: " << im.channels() << endl;


    Mat im2 = imread("./ball.png", IMREAD_COLOR);
    namedWindow("ventana2", WINDOW_AUTOSIZE);

    cout << "Information of im:" << endl;
    cout << "\t Rows: " << im2.rows << endl;
    cout << "\t Cols: " << im2.cols << endl;
    cout << "\t Channels: " << im2.channels() << endl;

    // Create a ROI
    Mat roi(im2, Rect(10, 10, im.rows, im.cols));
    // Copy other image to the ROI
    im.copyTo(roi);

    imshow("ventana2", im2);
    waitKey(0);

    destroyWindow("ventana2");


    // Now read and show a grayscale photo
    // Convert a photo from RGB/BGR to Grayscale
    cvtColor(im, im, CV_RGB2GRAY); //CV_32FC1
    namedWindow("ventana3", WINDOW_AUTOSIZE);
    imshow("ventana3", im);
    waitKey(0);


    cout << "Information of im:" << endl;
    cout << "\t Rows: " << im.rows << endl;
    cout << "\t Cols: " << im.cols << endl;
    cout << "\t Channels: " << im.channels() << endl;


    destroyWindow("ventana3");

    return 0;
}

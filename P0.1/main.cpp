#include <opencv2/opencv.hpp>

#include <opencv2/imgcodecs.hpp>
#include <strings.h>

using namespace std;
using namespace cv;

/**
 * Show an image in the screen
 * 
 * @param m The image to show
 */
void showImage(Mat &m);
/**
 * Show information about the image, Rows, columns and channels
 * 
 * @param m The image
 */
void showInfo(Mat &m);
/**
 * Draw a cross in the points of the image.
 * 
 * @param m
 * @param points
 */
void drawCross(Mat &m, vector<Point> &points);
/**
 * Divide the image in a 4x4 grid
 * 
 * @param m The image to divide
 */
void grid16(Mat &m);
/**
 * Divide the image intro 9 equitative points
 * @param m
 */
vector<Point> points9(Mat &m);

int main(int argc, char* argv[]) {

    srand(time(NULL));
    
    // Read and show in a window lena
    Mat im = imread("./lena.jpg", 1);
    vector<Point> a = points9(im);
    drawCross(im, a);
    grid16(im);
    showImage(im);
    showInfo(im);

    Mat im2 = imread("./ball.png", IMREAD_COLOR);
    // Create a ROI
    Mat roi(im2, Rect(0, 0, im.rows, im.cols));
    // Copy other image to the ROI
    im.copyTo(roi);

    showImage(im2);
    showInfo(im2);

    // Now read and show a grayscale photo
    // Convert a photo from RGB/BGR to Grayscale
    cvtColor(im, im, CV_RGB2GRAY);

    showImage(im);
    showInfo(im);

    Mat ball = imread("./ball.png", IMREAD_COLOR);
    vector<Point> b = points9(ball);
    drawCross(ball, b);
    grid16(ball);
    showImage(ball);

    // Read Lena again, converted it to grayscale and then show the points
    Mat lenna = imread("./lena.jpg", IMREAD_GRAYSCALE);
    vector<Point> c = points9(lenna);
    drawCross(lenna, c);
    showImage(lenna);

    return 0;
}

void showImage(Mat &m) {
    if (!m.empty()) {
        namedWindow("ventana", WINDOW_AUTOSIZE);
        imshow("ventana", m);
        waitKey(0);
        destroyWindow("ventana");
    }
}

void showInfo(Mat &m) {
    if (!m.empty()) {
        cout << "Image information: " << endl;
        cout << "\t Rows: " << m.rows << endl;
        cout << "\t Cols: " << m.cols << endl;
        cout << "\t Channels: " << m.channels() << endl;
    }
}

void drawCross(Mat &m, vector<Point> &points) {
    if (!m.empty() && !points.empty()) {
        for (int i = 0; i < points.size(); i++) {
            Point a = points.at(i);
            // Make a copy of the point to draw it, the center of the point is a (x,y)
            Point cross(points.at(i).x - 2, points.at(i).y);

            // Draw line from a to a.x-2, a.y
            line(m, a, cross, CV_RGB(255, 0, 0), 1, LINE_AA);
            cross.x += 2;
            cross.y += 2;
            // Draw line from a to a.x, a.y+2
            line(m, a, cross, CV_RGB(255, 0, 0), 1, LINE_AA);
            cross.x += 2;
            cross.y -= 2;
            // Draw line from a to a.x+2, a.y
            line(m, a, cross, CV_RGB(255, 0, 0), 1, LINE_AA);
            cross.x -= 2;
            cross.y -= 2;
            // Draw line from a to a.x, a.y-2
            line(m, a, cross, CV_RGB(255, 0, 0), 1, LINE_AA);
        }
    }
}

vector<Point> points9(Mat &m) {
    vector<Point> points;

    if (!m.empty()) {
        int xdivisor = m.cols / 4;
        int ydivisor = m.rows / 4;
        int x = xdivisor;
        int y = ydivisor;

        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                Point a(x, y);
                points.push_back(a);
                x += xdivisor;
            }
            x = xdivisor;
            y += ydivisor;
        }
    }
    return points;
}

void grid16(Mat &m) {

    if (!m.empty()) {
        int splitcols = m.cols / 4;
        int splitrows = m.rows / 4;
        int initialx = 0;
        int initialy = 0;

        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 4; k++) {
                Mat rec(m, Rect(initialx, initialy, splitcols, splitrows));
                rec = rec * (rand() % 4 + 1) - (rand() % 64);
                initialx += splitcols;
            }
            initialx = 0;
            initialy += splitrows;
        }
    }
}
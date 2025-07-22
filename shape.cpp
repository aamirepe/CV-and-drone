#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {
    // Blank Image
    Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));

    circle(img, Point(256, 256), 155, Scalar(0, 69, 255), FILLED);
    rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), 3);

    imshow("Image", img);
    waitKey(0);

    return 0;
}

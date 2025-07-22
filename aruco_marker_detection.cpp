#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>

using namespace cv;
using namespace cv::aruco;
using namespace std;

int main() {
    Mat img = imread("Screenshot 2025-05-19 at 10.59.22 AM.png");
    if (img.empty()) {
        cerr << "Error: Could not load image.\n";
        return -1;
    }

    Ptr<Dictionary> dict = makePtr<Dictionary>(getPredefinedDictionary(DICT_6X6_250));

    vector<vector<Point2f>> corners;
    vector<int> ids;

    detectMarkers(img, dict, corners, ids);

    if (!ids.empty()) {
        drawDetectedMarkers(img, corners, ids);
    }

    imshow("Out", img);
    waitKey(0);

    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
// g++ -std=c++17 pose_estimation.cpp -o pose `pkg-config --cflags --libs opencv4`

int main() {
    // Load camera calibration parameters
    Mat cameraMatrix = (Mat_<double>(3, 3) << 536.07, 0, 342.37,
                                              0, 536.01, 235.53,
                                              0, 0, 1);
    Mat distCoeffs = (Mat_<double>(1, 5) << -0.265, -0.0467, 0.00183, -0.00031, 0.2523);

    // Load predefined ArUco dictionary
    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(aruco::DICT_4X4_50));


    float markerLength = 0.05f;  


    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening webcam." << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap>> frame;
        if (frame.empty()) {
            cerr << "Empty frame, skipping." << endl;
            continue;
        }

        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners;


        aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);

        if (!markerIds.empty()) {
            // Draw markers
            aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            // Estimate and draw pose
            vector<Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(markerCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
            for (size_t i = 0; i < markerIds.size(); i++) {
                drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);

                cout << "Marker ID: " << markerIds[i] << endl;
                cout << "Rvec: " << rvecs[i] << ", Tvec: " << tvecs[i] << endl;
            }
        }

        imshow("Live Pose Estimation", frame);
        if (waitKey(1) == 27) break;  // Press ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

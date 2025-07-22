#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::aruco;

// g++ traj_velo.cpp -std=c++17 -o traj_velo `pkg-config --cflags --libs opencv4`

int main() {

    // [ fx   0  cx ]
// [ 0   fy  cy ]
// [ 0    0   1 ]
// for my camera
// fx = 536.07 → focal length in x
// fy = 536.01 → focal length in y
// cx = 342.37 → optical center x
// cy = 235.53 → optical center y
    // ======== CAMERA CALIBRATION PLACEHOLDERS ========
    Mat cameraMatrix = (Mat_<double>(3, 3) << 
        536.07, 0, 342.37,
        0, 536.01, 235.53,
        0, 0, 1); // Replace with real calibration

    Mat distCoeffs = Mat::zeros(1, 5, CV_64F); // Replace if distortion exists

    // ======== ARUCO DICTIONARY & DETECTOR ========
    Ptr<Dictionary> dictionary = makePtr<Dictionary>(
        getPredefinedDictionary(DICT_4X4_50));

    Ptr<DetectorParameters> parameters = new DetectorParameters();


    string video_path = "aruco_line_det.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error opening video!" << endl;
        return -1;
    }

    float markerLength = 0.08f; // marker length in meters

    // ======== KALMAN FILTER SETUP ========
    const int stateSize = 4;        // [x, y, dx, dy]
    const int measSize = 2;         // [x, y]
    KalmanFilter kf(stateSize, measSize, 0);

    kf.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf.measurementMatrix = (Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    kf.processNoiseCov = Mat::eye(stateSize, stateSize, CV_32F) * 0.03f;
    kf.measurementNoiseCov = Mat::eye(measSize, measSize, CV_32F) * 0.5f;
    kf.errorCovPost = Mat::eye(stateSize, stateSize, CV_32F);
    randn(kf.statePost, 0, 0.1);

    // ======== FOR VELOCITY MEASUREMENT ========
    double prev_time = static_cast<double>(getTickCount()) / getTickFrequency();
    Vec3d prev_tvec(0, 0, 0);
    bool first_measurement = true;

    Mat frame;
    while (cap.read(frame)) {
        // Time delta
        double curr_time = static_cast<double>(getTickCount()) / getTickFrequency();
        double dt = curr_time - prev_time;
        prev_time = curr_time;

        // Kalman prediction
        Mat prediction = kf.predict();
        int pred_x = static_cast<int>(prediction.at<float>(0));
        int pred_y = static_cast<int>(prediction.at<float>(1));

        vector<int> ids;
        vector<vector<Point2f>> corners;
        vector<Vec3d> rvecs, tvecs;

        detectMarkers(frame, dictionary, corners, ids, parameters);

        if (!ids.empty()) {
            estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
            vector<Point2f> marker_corners = corners[0];

            // Image center of marker
            float center_x = 0.0f, center_y = 0.0f;
            for (const auto& pt : marker_corners) {
                center_x += pt.x;
                center_y += pt.y;
            }
            center_x /= 4.0f;
            center_y /= 4.0f;

            // Kalman update
            Mat measurement = (Mat_<float>(2, 1) << center_x, center_y);
            kf.correct(measurement);

            // Draw actual center
            circle(frame, Point(center_x, center_y), 6, Scalar(0, 255, 0), 2); // Green

            // Real-world velocity
            Vec3d tvec = tvecs[0];
            if (!first_measurement) {
                double dist = norm(tvec - prev_tvec);      // in meters
                double velocity_mps = dist / dt;           // m/s

                // Display
                string vel_text = "v = " + to_string(velocity_mps).substr(0, 5) + " m/s";
                putText(frame, vel_text, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 4);
            }
            prev_tvec = tvec;
            first_measurement = false;
        }

        // Draw predicted position
        circle(frame, Point(pred_x, pred_y), 8, Scalar(0, 0, 255), 2); // Red

        // Show frame
        imshow("Kalman ArUco Tracking", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

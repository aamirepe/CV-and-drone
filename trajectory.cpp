#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::aruco;

// g++ trajectory.cpp -std=c++17 -o traj `pkg-config --cflags --libs opencv4`

int main() {

    Ptr<Dictionary> dictionary = makePtr<Dictionary>(
        getPredefinedDictionary(DICT_4X4_50));


    Ptr<DetectorParameters> parameters = new DetectorParameters();


    string video_path = "aruco_line.mp4";
    VideoCapture cap(video_path);

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


    vector<Point> kalmanPath;
    vector<Point> measuredPath;

    Mat frame;
    while (cap.read(frame)) {

        Mat prediction = kf.predict();
        int pred_x = static_cast<int>(prediction.at<float>(0));
        int pred_y = static_cast<int>(prediction.at<float>(1));
        kalmanPath.push_back(Point(pred_x, pred_y));


        vector<int> ids;
        vector<vector<Point2f>> corners;
        detectMarkers(frame, dictionary, corners, ids, parameters);


        if (!ids.empty()) {
            vector<Point2f> marker_corners = corners[0];

            float center_x = 0.0f, center_y = 0.0f;
            for (const auto& pt : marker_corners) {
                center_x += pt.x;
                center_y += pt.y;
            }
            center_x /= 4.0f;
            center_y /= 4.0f;

            Mat measurement = (Mat_<float>(2, 1) << center_x, center_y);
            kf.correct(measurement);
            measuredPath.push_back(Point(static_cast<int>(center_x), static_cast<int>(center_y)));

            circle(frame, Point(center_x, center_y), 6, Scalar(0, 255, 0), 2);
        }

        circle(frame, Point(pred_x, pred_y), 8, Scalar(0, 0, 255), 2);

        for (size_t i = 1; i < measuredPath.size(); ++i) {
            line(frame, measuredPath[i - 1], measuredPath[i], Scalar(0, 255, 0), 2);
        }

        for (size_t i = 1; i < kalmanPath.size(); ++i) {
            line(frame, kalmanPath[i - 1], kalmanPath[i], Scalar(0, 0, 255), 2);
        }

        imshow("Kalman ArUco Tracking", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

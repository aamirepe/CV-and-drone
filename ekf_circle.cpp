#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;
using namespace Eigen;


// g++ -std=c++17 ekf_circle.cpp -o ekf -I/opt/homebrew/include/eigen3 `pkg-config --cflags --libs opencv4`
//--- Configuration --------------------------------------------------------
const int    NUM_POINTS    = 200;
const double RADIUS        = 100.0;
const Point  CENTER        = Point(250, 250);
const double NOISE_STDDEV  = 8.0;
const double DT            = 1.0;  // time step

//--- Extended Kalman Filter for circular motion ---------------------------
struct EKF_Circle {
    Vector2d x;   // [theta; omega]
    Matrix2d P, Q, R;

    EKF_Circle()
    {
        // initial guess: theta=0, small angular velocity
        x << 0.0, 2 * M_PI / NUM_POINTS;  

        P = Matrix2d::Identity() * 1.0;       // initial covariance
        Q = (Matrix2d() << 1e-4, 0, 0, 1e-4).finished();  // process noise
        R = Matrix2d::Identity() * (NOISE_STDDEV * NOISE_STDDEV);
    }

    // Predict step: θₖ₊₁ = θₖ + ω·Δt, ω constant
    void predict() {
        double theta = x(0), omega = x(1);
        // State prediction
        x(0) = theta + omega * DT;
        // Jacobian F = d f / d x
        Matrix2d F;
        F << 1.0, DT,
             0.0, 1.0;
        // Covariance update
        P = F * P * F.transpose() + Q;
    }

    // Update step using measurement z = [x_meas, y_meas]
    void update(const Vector2d& z) {
        double theta = x(0);
        // h(x) = [cx + r cos θ; cy + r sin θ]
        Vector2d h;
        h << CENTER.x + RADIUS * cos(theta),
             CENTER.y + RADIUS * sin(theta);

        // Jacobian H = dh/dx
        Matrix<double,2,2> H;
        H << -RADIUS * sin(theta),  0,
              RADIUS * cos(theta),  0;

        // Innovation
        Vector2d y = z - h;
        Matrix2d S = H * P * H.transpose() + R;
        Matrix2d K = P * H.transpose() * S.inverse();

        // State & covariance update
        x = x + K * y;
        P = (Matrix2d::Identity() - K * H) * P;
    }

    // Convert current θ→(x,y)
    Point2d estimatedPosition() const {
        double theta = x(0);
        return Point2d(
            CENTER.x + RADIUS * cos(theta),
            CENTER.y + RADIUS * sin(theta)
        );
    }
};

//--- Helper: generate noisy circle points --------------------------------
vector<Point2d> generateNoisyCircle(int n, double noiseStd) {
    vector<Point2d> pts;
    default_random_engine gen((unsigned)123);
    normal_distribution<double> noise(0.0, noiseStd);

    for (int i = 0; i < n; ++i) {
        double angle = 2*M_PI * i / n;
        double x = CENTER.x + RADIUS * cos(angle) + noise(gen);
        double y = CENTER.y + RADIUS * sin(angle) + noise(gen);
        pts.emplace_back(x, y);
    }
    return pts;
}

//--- Main -----------------------------------------------------------------
int main() {
    // 1) Generate measurements
    vector<Point2d> meas = generateNoisyCircle(NUM_POINTS, NOISE_STDDEV);

    // 2) Run EKF
    EKF_Circle ekf;
    vector<Point2d> track;
    track.reserve(NUM_POINTS);

    for (int i = 0; i < NUM_POINTS; ++i) {
        ekf.predict();
        Vector2d z(meas[i].x, meas[i].y);
        ekf.update(z);
        track.push_back(ekf.estimatedPosition());
    }

    // 3) Draw everything
    Mat img(500, 500, CV_8UC3, Scalar::all(255));

    // a) True circle (yellow)
    circle(img, CENTER, (int)RADIUS, Scalar(0,0,0), 1);

    // b) Noisy measurements (red)
    for (int i = 1; i < NUM_POINTS; ++i)
        line(img, meas[i-1], meas[i], Scalar(0,0,255), 1);

    // c) EKF track (blue)
    for (int i = 1; i < NUM_POINTS; ++i)
        line(img, track[i-1], track[i], Scalar(255,0,0), 1);

    // d) Mark first few estimates (green)
    for (int i = 0; i < 5; ++i)
        circle(img, track[i], 3, Scalar(0,255,0), -1);

    // Show
    imshow("EKF Circular Tracking", img);
    waitKey();
    return 0;
}

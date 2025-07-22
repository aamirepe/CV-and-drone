#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

class Filter {
public:
    virtual void predict() = 0;
    virtual void update(const Vector2d& measurement) = 0;
    virtual Vector2d get_point() = 0;

    Vector2d predict_point(const Vector2d& measurement) {
        update(measurement);
        predict();
        return get_point();
    }
};

class KalmanFilter : public Filter {
public:
    VectorXd x;
    MatrixXd f, p, h, r, i;

    KalmanFilter(double sigma) {
        x = VectorXd::Zero(6);
        f = MatrixXd(6, 6);
        f << 1, 0, 1, 0, 0, 0,
             0, 1, 0, 1, 0, 0,
             0, 0, 1, 0, 1, 0,
             0, 0, 0, 1, 0, 1,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1;

        p = MatrixXd::Identity(6, 6) * 999;
        h = MatrixXd::Zero(2, 6);
        h(0, 0) = 1;
        h(1, 1) = 1;

        r = MatrixXd::Identity(2, 2) * sigma;
        i = MatrixXd::Identity(6, 6);
    }

    void predict() override {
        x = f * x;
        p = f * p * f.transpose();
    }

    void update(const Vector2d& measurement) override {
        Vector2d z = measurement;
        Vector2d y = z - h * x;
        Matrix2d s = h * p * h.transpose() + r;
        MatrixXd k = p * h.transpose() * s.inverse();
        x = x + k * y;
        p = (i - k * h) * p;
    }

    Vector2d get_point() override {
        return x.head<2>();
    }
};

vector<Vector2d> create_artificial_circle_data(double step_size, double radius, int N) {
    vector<Vector2d> measurements;
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        for (double x = -radius; x < radius; x += step_size) {
            double y = sqrt(radius * radius - x * x) + noise(gen);
            measurements.emplace_back(x, y);
        }
        for (double x = radius; x > -radius; x -= step_size) {
            double y = -sqrt(radius * radius - x * x) + noise(gen);
            measurements.emplace_back(x, y);
        }
    }
    return measurements;
}

double get_dist(const Vector2d& a, const Vector2d& b) {
    return (a - b).norm();
}

pair<vector<Vector2d>, vector<double>> action(Filter& filter, const vector<Vector2d>& measurements) {
    vector<Vector2d> result;
    vector<double> err_list;

    for (const auto& m : measurements) {
        Vector2d pred = filter.predict_point(m);
        double err = get_dist(m, pred);
        cout << "error: " << err << endl;
        result.push_back(pred);
        err_list.push_back(err);
    }

    return {result, err_list};
}

void plot(const vector<Vector2d>& measurements, const vector<Vector2d>& result, const vector<double>& err_list, double radius) {
    int width = 800, height = 800;
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    double scale = width / (2 * radius);

    for (const auto& p : measurements) {
        cv::circle(image, cv::Point(scale * (p(0) + radius), scale * (radius - p(1))), 1, cv::Scalar(0, 0, 255), -1);
    }

    for (const auto& p : result) {
        cv::circle(image, cv::Point(scale * (p(0) + radius), scale * (radius - p(1))), 1, cv::Scalar(255, 0, 0), -1);
    }

    cv::imshow("Kalman Filter Tracking", image);
    cv::waitKey(0);

    cv::Mat errorPlot(err_list.size(), 400, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 1; i < err_list.size(); ++i) {
        cv::line(errorPlot,
                 cv::Point(i - 1, 400 - int(err_list[i - 1] * 10)),
                 cv::Point(i, 400 - int(err_list[i] * 10)),
                 cv::Scalar(0, 0, 0), 1);
    }

    cv::imshow("Error Plot", errorPlot);
    cv::waitKey(0);
}

int main() {
    KalmanFilter kf(0.01);
    vector<Vector2d> measurements = create_artificial_circle_data(0.1, 400, 10);
    auto [result, err_list] = action(kf, measurements);
    plot(measurements, result, err_list, 800);

    return 0;
}


// g++ -std=c++11 kalman_circle_object.cpp -o track -I/opt/homebrew/include/eigen3 `pkg-config --cflags --libs opencv4`

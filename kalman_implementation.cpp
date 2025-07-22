#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;
using namespace Eigen;

#define DEG2RAD(x) ((x) * M_PI / 180.0)

class KalmanFilter {
public:
    KalmanFilter(double dt, double INIT_POS_STD, double INIT_VEL_STD, double ACCEL_STD, double GPS_POS_STD) {
        this->dt = dt;
        x = Vector4d::Zero();

        P = Matrix4d::Zero();
        P(0, 0) = INIT_POS_STD * INIT_POS_STD;
        P(1, 1) = INIT_POS_STD * INIT_POS_STD;
        P(2, 2) = INIT_VEL_STD * INIT_VEL_STD;
        P(3, 3) = INIT_VEL_STD * INIT_VEL_STD;

        F << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

        H << 1, 0, 0, 0,
             0, 1, 0, 0;

        Matrix2d q = Matrix2d::Zero();
        q(0, 0) = ACCEL_STD * ACCEL_STD;
        q(1, 1) = ACCEL_STD * ACCEL_STD;

        L << 0.5 * dt * dt, 0,
             0, 0.5 * dt * dt,
             dt, 0,
             0, dt;

        Q = L * q * L.transpose();

        R = Matrix2d::Zero();
        R(0, 0) = GPS_POS_STD * GPS_POS_STD;
        R(1, 1) = GPS_POS_STD * GPS_POS_STD;

        I = Matrix4d::Identity();
    }

    Vector2d predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
        return Vector2d(x(0), x(1));
    }

    Vector2d update(const Vector2d& z) {
        Vector2d z_hat = H * x;
        Vector2d y = z - z_hat;
        Matrix2d S = H * P * H.transpose() + R;
        Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();
        x = x + K * y;
        P = (I - K * H) * P;
        return Vector2d(x(0), x(1));
    }

    Vector2d getVelocity() const {
        return Vector2d(x(2), x(3));
    }

private:
    double dt;
    Vector4d x;
    Matrix4d P, F, Q, I;
    Matrix<double, 4, 2> L;
    Matrix<double, 2, 4> H;
    Matrix2d R;
};

int main() {
    const int num_measurements = 100;
    Vector2d initial_position(0.0, 0.0);
    double speed = 5.0;
    double heading_deg = 45.0;
    double heading_rad = DEG2RAD(heading_deg);

    default_random_engine generator;
    normal_distribution<double> noise(0.0, 5.0);

    vector<Vector2d> true_positions;
    vector<Vector2d> measurements;

    Vector2d pos = initial_position;
    for (int i = 0; i < num_measurements; ++i) {
        true_positions.push_back(pos);

        double x_measured = pos(0) + noise(generator);
        double y_measured = pos(1) + noise(generator);
        measurements.push_back(Vector2d(x_measured, y_measured));

        pos(0) += speed * cos(heading_rad);
        pos(1) += speed * sin(heading_rad);
    }

    double dt = 1.0;
    KalmanFilter kf(dt, 10.0, 10.0, 5.0, 3.0);

    vector<Vector2d> filtered_positions;
    vector<Vector2d> filtered_velocities;
    for (const auto& z : measurements) {
        kf.predict();
        Vector2d updated_pos = kf.update(z);
        filtered_positions.push_back(updated_pos);
        filtered_velocities.push_back(kf.getVelocity());
    }

    for (size_t i = 0; i < num_measurements; ++i) {
        cout << "True Pos: [" << true_positions[i].transpose() << "] "
             << "Measured Pos: [" << measurements[i].transpose() << "] "
             << "Filtered Pos: [" << filtered_positions[i].transpose() << "] "
             << "Filtered Vel: [" << filtered_velocities[i].transpose() << "]\n";
    }

    return 0;
}

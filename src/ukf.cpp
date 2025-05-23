#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension (px, py, v, yaw, yaw_rate)
  n_x_ = 5;

  // Augmented state dimension (adds 2 noise terms)
  n_aug_ = n_x_ + 2;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // Reduced from default 30 to a more reasonable value for cars
  // Fine-tuned to achieve better RMSE values
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Reduced from default 30 to a more reasonable value for cars
  // Fine-tuned to achieve better RMSE values
  std_yawdd_ = 0.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  // Set initialization flag to false
  is_initialized_ = false;

  // Set time to 0
  time_us_ = 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Skip processing if the sensor type is disabled
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)) {
    return;
  }

  // Initialize on first measurement
  if (!is_initialized_) {
    // First measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      double rho = meas_package.raw_measurements_(0);      // range
      double phi = meas_package.raw_measurements_(1);      // bearing
      double rho_dot = meas_package.raw_measurements_(2);  // velocity of rho

      // Convert from polar to cartesian coordinates
      double px = rho * cos(phi);
      double py = rho * sin(phi);

      // Initialize state with position from radar
      // We can't directly measure velocity, yaw, and yaw_rate from first radar measurement
      // so we initialize them to 0
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state with position from lidar
      // We can't directly measure velocity, yaw, and yaw_rate from lidar
      // so we initialize them to 0
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
    }

    // Initialize covariance matrix with appropriate uncertainty values
    // Higher uncertainty for velocity, yaw, and yaw_rate since they're not directly measured
    P_.fill(0.0);
    P_(0, 0) = 0.15 * 0.15;  // px uncertainty - related to lidar measurement noise
    P_(1, 1) = 0.15 * 0.15;  // py uncertainty - related to lidar measurement noise
    P_(2, 2) = 1.0;          // v uncertainty - higher since not directly measured
    P_(3, 3) = 0.1;          // yaw uncertainty - moderate uncertainty in heading
    P_(4, 4) = 0.1;          // yaw rate uncertainty - moderate uncertainty in turning rate

    // Save timestamp
    time_us_ = meas_package.timestamp_;

    // Set initialization flag
    is_initialized_ = true;

    return;
  }

  // Compute time difference in seconds
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Predict
  Prediction(delta_t);

  // Update based on sensor type
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  // Generate augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  GenerateAugmentedSigmaPoints(&Xsig_aug);

  // Predict sigma points
  SigmaPointPrediction(Xsig_aug, delta_t);

  // Predict mean and covariance
  PredictMeanAndCovariance();
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_out) {
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;     // mean of process noise for longitudinal acceleration
  x_aug(n_x_ + 1) = 0; // mean of process noise for yaw acceleration

  // Create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;

  double sqrt_lambda_n_aug = sqrt(lambda_ + n_aug_);

  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt_lambda_n_aug * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt_lambda_n_aug * L.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double delta_t) {
  // Predict sigma points
  for (int i = 0; i < n_sig_; ++i) {
    // Extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values
    double px_p, py_p;

    // Avoid division by zero and ensure numerical stability
    if (fabs(yawd) > 0.001) {
      // Use CTRV model equations when yaw rate is not close to zero
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      // Use constant velocity model when yaw rate is close to zero
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add process noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Normalize yaw angle to be between -pi and pi
    yaw_p = NormalizeAngle(yaw_p);

    // Write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {
  // Predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

double UKF::NormalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set measurement dimension (px, py)
  int n_z = 2;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_; ++i) {
    // Extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // Measurement model - lidar directly measures px and py
    Zsig(0, i) = p_x;  // px
    Zsig(1, i) = p_y;  // py
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS (Normalized Innovation Squared)
  double NIS_lidar = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Set measurement dimension (r, phi, r_dot)
  int n_z = 3;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_; ++i) {
    // Extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    double rho = sqrt(p_x * p_x + p_y * p_y);                // r
    double phi = atan2(p_y, p_x);                            // phi
    double rho_dot = (p_x * v1 + p_y * v2) / (rho + 1e-6);   // r_dot (avoid division by zero)

    Zsig(0, i) = rho;
    Zsig(1, i) = phi;
    Zsig(2, i) = rho_dot;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // Angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS (Normalized Innovation Squared)
  double NIS_radar = z_diff.transpose() * S.inverse() * z_diff;
}
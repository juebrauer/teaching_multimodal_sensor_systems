/// file: kalman_filter_ndim_demo.cpp
///
/// This is a simple 2D Kalman filter demo using
/// the n-dimensional Kalman filter implementation
/// encapsulated in the class kalman_filter_ndim.
///
/// A 2D car drives around in a 2D plane. We want
/// to estimate its 2D position, but we only have
/// information about a 2D steering vector u and
/// a noisy 2D position sensor.
///
/// The 2D KF can help! It combines the information
/// from several sources:
///
/// - a (linear) motion model encapsulated in the transition matrix F,
///
/// - information about how noisy the motions are
///   (process noise covariance matrix Q)
///
/// - a (linear) model how the steering signal u changes the
///   state (encapsulated in the control matrix B)
///
/// - the steering signal / vector u itself
///
/// - a (linear) measurement model encapsulated in the measurement matrix
///   H that explains how states are mapped to sensor values by our
///   sensors
///
/// - information about how noisy this measurement process is
///   (measurement noise covariance matrix R)
///
/// - the measurements z (encapsulated in a vector)
///
/// By fusing all this pieces of information we can become
/// better than just believing our noisy 2D position sensor!
///
/// See it and believe it!
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org

#include "opencv2/opencv.hpp"

#include <iostream>
#include <conio.h>
#include <random>                 // for random numbers & 1D normal distribution
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI

#include "kalman_filter_ndim.h"
#include "mvnrnd.h"               // for random number generator following a n-dimensional normal distribution

#define IMG_WIDTH  800
#define IMG_HEIGHT 800
#define LINE_WIDTH 2
#define COL_GT_POS         CV_RGB(255,255,255)   // WHITE
#define COL_NAIVE_EST_POS  CV_RGB(255,255,0)     // YELLOW
#define COL_MEASUREMENT    CV_RGB(255,0,0)       // RED
#define COL_KF_EST_POS     CV_RGB(0,255,0)       // GREEN
#define COL_KF_UNCERTAINTY CV_RGB(0,255,255)     // CYAN


using namespace cv;


Mat image;

// Macro for mapping a 2D matrix/vector to a 2D point
#define m2p(M) Point((int)M.at<float>(0,0), (int)M.at<float>(1,0))


void show_ground_truth_pos_vs_estimated_pos(Mat       gt_pos,
                                            Mat       naive_est_pos,
                                            Mat       measurement,
                                            Mat       kf_est_pos,
                                            Mat       kf_uncertainty,
                                            double    error_measurement,
                                            double    error_naive,
                                            double    error_kf)
{
  const int radius = 5;

  // 1. clear visualization image
  image = 0;

  // 2. show info what we display with which circle
  char txt[100];  
  double text_size = 0.5;
  Point offsetvec1 = Point(-100,  +50); // left bottom
  Point offsetvec2 = Point( +75,  -50); // right top
  Point offsetvec3 = Point(-100,  -50); // left top
  sprintf_s(txt, "Ground truth pos: (%d,%d)", (int)gt_pos.at<float>(0,0), (int)gt_pos.at<float>(1,0));
  putText(image, txt, m2p(gt_pos) + offsetvec1, FONT_HERSHEY_SIMPLEX, text_size, COL_GT_POS, 1);
  line(image, m2p(gt_pos) + offsetvec1, m2p(gt_pos), COL_GT_POS, 1);

  sprintf_s(txt, "Measured pos: error = %.2f", error_measurement);
  putText(image, txt, m2p(measurement) + offsetvec2, FONT_HERSHEY_SIMPLEX, text_size, COL_MEASUREMENT, 1);
  line(image, m2p(measurement) + offsetvec2, m2p(measurement), COL_MEASUREMENT, 1);

  sprintf_s(txt, "Kalman filtered pos: error = %.2f", error_kf);
  putText(image, txt, m2p(kf_est_pos) + offsetvec3, FONT_HERSHEY_SIMPLEX, text_size, COL_KF_EST_POS, 1);
  line(image, m2p(kf_est_pos) + offsetvec3, m2p(kf_est_pos), COL_KF_EST_POS, 1);

  sprintf_s(txt, "Naive estimated pos: error = %.2f", error_naive);
  putText(image, txt, m2p(naive_est_pos) + offsetvec2, FONT_HERSHEY_SIMPLEX, text_size, COL_NAIVE_EST_POS, 1);
  line(image, m2p(naive_est_pos) + offsetvec2, m2p(naive_est_pos), COL_NAIVE_EST_POS, 1);


  // 3. draw ground truth position
  circle(image, m2p(gt_pos), radius*2, COL_GT_POS, 1);
  
  // 4. draw measurement
  circle(image, m2p(measurement), radius, COL_MEASUREMENT, LINE_WIDTH);
      
  // 5.1 draw Kalman filter estimated position
  circle(image, m2p(kf_est_pos), radius, COL_KF_EST_POS, LINE_WIDTH);

  // 5.2 draw Kalman filter uncertainty as an ellipse
  //     corresponding to the current uncertainty covariance matrix
  if (false)
  {
    // compute Eigenvectors + Eigenvalues of the uncertainty covariance matrix  
    Mat eigenvalues;
    Mat eigenvectors;
    eigen(kf_uncertainty, eigenvalues, eigenvectors);
    std::cout << kf_uncertainty << std::endl;
    std::cout << eigenvalues << std::endl;
    std::cout << eigenvectors << std::endl;

    // compute ellipse axis
    double eigenvec1_len = eigenvalues.at<double>(0, 0);
    double len1 = sqrt(eigenvec1_len) * 3;
    double eigenvec1_x = eigenvectors.at<double>(0, 0) * len1;
    double eigenvec1_y = eigenvectors.at<double>(1, 0) * len1;
    double eigenvec2_len = eigenvalues.at<double>(1, 0);
    double len2 = sqrt(eigenvec2_len) * 3;
    double eigenvec2_x = eigenvectors.at<double>(0, 1) * len2;
    double eigenvec2_y = eigenvectors.at<double>(1, 1) * len2;

    // compute rotated bounding box for drawing the ellipse
    double dx = eigenvec1_x;
    double dy = eigenvec1_y;
    double angle_rad = atan2(dy, dx);
    double angle_deg = angle_rad * (180.0 / M_PI); // convert radians (0,2PI) to degree (0°,360°)
    cv::RotatedRect myRotatedRect(m2p(kf_est_pos), Size((int)len1, (int)len2), (float)angle_deg);
    ellipse(image, myRotatedRect, COL_KF_EST_POS);
  }
  
  // 6. draw naive estimated position
  circle(image, m2p(naive_est_pos), radius, COL_NAIVE_EST_POS, LINE_WIDTH);
  
  // 7. show visualization image
  imshow("Kalman Filter 2D Demo : {Ground truth | measured | naive estimated | Kalman filtered} position of a 2D car", image);

} // show_ground_truth_pos_vs_estimated_pos



int main()
{
  // 1. create visualization image
  image = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
  

  // 2. initialize vectors and covariance matrices

  // 2.1 state vector µ: the 2D car is in middle of image at start
  cv::Mat µ = (Mat_<float>(2, 1) << IMG_WIDTH/2, IMG_HEIGHT/2);

  // 2.2 covariance matrix P: initial uncertainty about the state
  cv::Mat P = (Mat_<float>(2, 2) << 10.0,  0.0,
                                     0.0, 10.0);

  // 2.3 state transition matrix F: describes the system dynamics
  cv::Mat F_true = (Mat_<float>(2, 2) << 1.0,  0.0,
                                         0.0,  1.0);
  cv::Mat F_model = F_true * 1.0;

  // 2.4 control matrix B: describes how control signals affect the new state
  //     the desired control signal u = "change in (x,y) coords" will
  //     be mapped by the system differently than desired,
  //     namely by a change vector of (1.02*x, 0.98*y)
  cv::Mat B_true = (Mat_<float>(2, 2) << 1.02, 0.0,
                                          0.0, 0.98);
  cv::Mat B_model = B_true * 1.01;


  // 2.5 process noise covariance matrix: describes randomness of state
  //     transitions per dimension
  //     here noise in x-coordinate transitions is smaller than
  //     noise in y-coordinate state transitions
  cv::Mat Q_true = (Mat_<float>(2, 2) <<  50.0,   0.0,
                                           0.0,  25.0);
  cv::Mat Q_model = Q_true * 0.98;
  
  // 2.6 measurement matrix: describes how states are mapped by our sensors
  //     to measurements
  cv::Mat H_true = (Mat_<float>(2, 2) << 0.98, 0.0,
                                         0.0, 0.98);
  cv::Mat H_model = H_true * 1.03;

  // 2.7 measurement noise covariance matrix: describes randomness of
  //     measurements per dimension
  cv::Mat R_true = (Mat_<float>(2, 2) << 500.0,  0.0,
                                         0.0,  700.0);
  cv::Mat R_model = R_true * 0.97;
  //cv::Mat R_model = R_true * 0.5;


  // 3. prepare a n-dimensional Kalman filter for estimating the µ=(x,y)
  //    position of the 2D car 
  kalman_filter_ndim myFilter(µ, P, F_model, B_model, Q_model, H_model, R_model);


  // 4. prepare random number generators for simulating process & measurement noise
  

  // prepare random number generator for process noise
  cv::Mat mean_vec = (Mat_<float>(2, 1) << 0,0 );
  mvnrnd rnd_generator_process_noise = mvnrnd(mean_vec, Q_true);

  // prepare random number generator for measurement noise
  mvnrnd rnd_generator_measurement_noise = mvnrnd(mean_vec, R_true);


  // 5. define start control vector u
  cv::Mat u = (Mat_<float>(2, 1) << 2, 2);

  // 6. initialize naive state estimate, where
  //    we just integrate the control vectors
  cv::Mat µ_naive_est = µ.clone();
  
  int simulation_step = 0;  
  double error_measurement, error_naive, error_kf;
  error_measurement = error_naive = error_kf = 0.0;
  while (true)
  {
    // 1. clear screen
    system("cls");
    printf("Simulation step : %d\n", simulation_step);

    // 2. generate new random control signal
    //    from time to time
    if (rand() % 4 == 0)
    {
      float rnd_dux = (float)(-1 + rand() % 3);
      float rnd_duy = (float)(-1 + rand() % 3);
      u = u + (Mat_<float>(2, 1) << rnd_dux, rnd_duy);
    }

    float car_x = µ.at<float>(0,0);
    float car_y = µ.at<float>(1,0);
    
    if (car_x < IMG_WIDTH/4)                // near to left border?
      u.at<float>(0, 0) = 1;                // --> drive to the right
    if (car_x > IMG_WIDTH - IMG_WIDTH / 4)  // near to right border?
      u.at<float>(0, 0) = -1;               // --> drive to the left 

    if (car_y < IMG_HEIGHT / 4)             // near to upper border?
      u.at<float>(1, 0) = 1;                // --> drive to the bottom
    if (car_y > IMG_HEIGHT - IMG_HEIGHT/4)  // near to lower border?
      u.at<float>(1, 0) = -1;               // --> drive to the top



    // 3. update naive state estimate
    µ_naive_est = µ_naive_est + u;

    // 4. simulate new (ground truth) state / vector
    Mat rnd_vec1 = rnd_generator_process_noise.get_next_random_vector();
    µ = F_true*µ + B_true*u + rnd_vec1;

    // 5. simulate sensor data / vector
    Mat rnd_vec2 = rnd_generator_measurement_noise.get_next_random_vector();
    Mat z = H_true*µ + rnd_vec2;

    // 6. predict new state (1st step of a Kalman filter)
    myFilter.predict( u );

    // 7. correct new estimated state by help of sensor data (2nd step of a Kalman filter)
    myFilter.correct_by_measurement( z );

    // 8. update moving average of ground truth pos vs.
    //    measured, naive estimated, Kalman filtered 1D pos estimate
    double N = (double)simulation_step;
    error_measurement = (error_measurement*N + norm(z - µ))                                     / (N+1.0);
    error_naive       = (error_naive      *N + norm(µ_naive_est - µ))                           / (N+1.0);
    error_kf          = (error_kf         *N + norm(myFilter.get_current_state_estimate() - µ)) / (N+1.0);

    // 9. visualize ground truth vs. estimated state
    show_ground_truth_pos_vs_estimated_pos( µ,
                                            µ_naive_est,
                                            z,
                                            myFilter.get_current_state_estimate(),
                                            myFilter.get_current_uncertainty(),
                                            error_measurement,
                                            error_naive,
                                            error_kf
                                           );

    // 10. wait for user input
    printf("Press any key to go next simulation step!\n");
    int c = waitKey();
    if (c == 29) // ESC pressed?
      break;

    // 11. time goes by...
    simulation_step++;

  } // while


} // main
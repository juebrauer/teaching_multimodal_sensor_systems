/// file: extended_kalman_filter_demo.cpp
///
/// In this demo a 3D state is simulated.
/// The 3D state evolves according to a non-linear
/// state transition function f.
/// Further, a noisy 3D measurement sensor of this
/// 3D state is available. The measurement process
/// is described by a non-linear 3D function h.
///
/// The task of the Extended Kalman Filter (EKF)
/// here is to estimate the 3D state vector.
/// For this, the EKF approximates the non-linear
/// functions f and h in the update formulas using
/// a Taylor series approximation of the first order,
/// i.e. using a linear approximation. The Taylor
/// series approximation for high-dimensional functions
/// corresponds to the Jacobi-matrix.
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
///


#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <conio.h>
#include <random>                 // for random numbers & 1D normal distribution
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI

#include "mvnrnd.h"               // for random number generator following a n-dimensional normal distribution

#include "params.h"


using namespace cv;
using namespace std;


vector<Point2d> traj_gt;               // ground truth trajectory
vector<Point2d> traj_est_predonly;     // trajectory based on prediction only
vector<Point2d> traj_est_EKF;          // trajectory based on EKF estimate

Mat image;

// Macro for mapping a 3D matrix/vector to a 2D point
#define m2p(M) Point((int)M.at<float>(1,0), (int)M.at<float>(2,0))


// some non-linear function f that maps 3D states µ to new 3D states µ'
Mat f(Mat µ)
{ 
  float µ1 = µ.at<float>(0, 0);
  float µ2 = µ.at<float>(1, 0);
  float µ3 = µ.at<float>(2, 0);
  Mat µ_prime =
    (Mat_<float>(3, 1) <<
       µ1 + 1, 400 + 200 * cos(µ1 / 40.0), 400 + 200 * sin(µ1 / 20.0));
  
  return µ_prime;
}


// some non-linear function h that maps 3D states µ to 3D measurements z
Mat h(Mat µ)
{
  float µ1 = µ.at<float>(0, 0);
  float µ2 = µ.at<float>(1, 0);
  float µ3 = µ.at<float>(2, 0);

  
  Mat measurement_vec =
    (Mat_<float>(3, 1) << µ1*µ1, µ2*µ2, µ3*µ3);
  

  /*
  Mat measurement_vec =
     (Mat_<float>(3, 1) << µ1+µ2, cos(µ2), sin(µ3));
  */

  return measurement_vec;
}



int main()
{
  // 1. create visualization image
  image = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
  

  // 2. initialize vectors and covariance matrices

  // 2.1 state vector µ: the 2D car is in middle of image at start
  cv::Mat µ_gt = (Mat_<float>(3, 1) << 0, IMG_WIDTH/2, IMG_HEIGHT/2);

  // 2.2 covariance matrix P: initial uncertainty about the state
  cv::Mat P = (Mat_<float>(3, 3) << 10.0,  0.0, 0.0,
                                     0.0, 10.0, 0.0,
                                     0.0,  0.0, 10.0);

  // 2.3 process noise covariance matrix: describes randomness of state
  //     transitions per dimension
  cv::Mat Q = (Mat_<float>(3, 3) <<  5.01,   0.0,   0.0,
                                     0.0,    50.01,  0.0,
                                     0.0,    0.0,   50.01);

  // 2.4 measurement noise covariance matrix: describes randomness of
  //     measurements per dimension
  cv::Mat R = (Mat_<float>(3, 3) <<  5e8,   0.0,   0.0,
                                     0.0,   5e8,   0.0,
                                     0.0,    0.0,  5e8);
  

  // 3. prepare random number generators for simulating process & measurement noise
  
  // prepare random number generator for process noise
  cv::Mat mean_vec = (Mat_<float>(3, 1) << 0,0,0 );
  mvnrnd rnd_generator_process_noise = mvnrnd(mean_vec, Q);

  // prepare random number generator for measurement noise
  cv::Mat mean_vec2 = (Mat_<float>(3, 1) << 0,0,0);
  mvnrnd rnd_generator_measurement_noise = mvnrnd(mean_vec2, R);

  
  

  // initialize estimated state vectors
  // with estimation just by prediction and
  // with estimation based on EKF
  Mat µ_est_predonly = µ_gt.clone();
  Mat µ_est          = µ_gt.clone();  

  float avg_error_µ_est_predonly = 0.0;  
  float avg_error_µ_est = 0.0;
  int simulation_step = 0;

  while (true)
  {
    // 1. clear screen & visualization image
    system("cls");
    printf("Simulation step : %d\n", simulation_step);
    image = 0;


    // 2. simulate new ground truth state
    //    that is computed by some non-linear function
    Mat rnd_vec1 = rnd_generator_process_noise.get_next_random_vector();
    if (SHOW_DEBUG_INFO)
    {
      printf("rnd_vec1 = (%.2f,%.2f,%.2f)\n",
        rnd_vec1.at<float>(0, 0), rnd_vec1.at<float>(1, 0), rnd_vec1.at<float>(2, 0));
    }
    µ_gt = f(µ_gt) + rnd_vec1;
    

    
    // 3. simulate sensor data / vector    
    Mat rnd_vec2 = rnd_generator_measurement_noise.get_next_random_vector();    
    Mat gt_measurement_vec = h(µ_gt);
    Mat z = gt_measurement_vec + rnd_vec2;
    if (SHOW_DEBUG_INFO)
    {
      printf("rnd_vec2 that will be added to measurement vec = (%.2f,%.2f,%.2f)\n",
        rnd_vec2.at<float>(0, 0), rnd_vec2.at<float>(1, 0), rnd_vec2.at<float>(2, 0) );
      printf("gt_measurement_vec = (%.2f,%.2f,%.2f)\n",
        gt_measurement_vec.at<float>(0, 0), gt_measurement_vec.at<float>(1, 0), gt_measurement_vec.at<float>(2, 0));
      printf("z = (%.2f,%.2f,%.2f)\n",
        z.at<float>(0, 0), z.at<float>(1, 0), z.at<float>(2, 0));
    }

    
    // 4. update state estimate just based on prediction
    µ_est_predonly = f(µ_est_predonly);
    



    
    // 5.1 predict new state using EKF & update error covariance matrix    
    µ_est = f(µ_est);
    
    // compute Jacobi matrix of f at location a=µ_est
    float µ1 = µ_est.at<float>(0, 0);    
    float µ2 = µ_est.at<float>(1, 0);
    float µ3 = µ_est.at<float>(2, 0);
    Mat F = (Mat_<float>(3, 3) <<                              1.0, 0.0, 0.0,
                                  -200.0 * sin(µ1/40.0) * 1.0/40.0, 0.0, 0.0,
                                   200.0 * cos(µ1/20.0) * 1.0/20.0, 0.0, 0.0);
    //F = cv::Mat::eye(3,3,CV_32FC1);
    P = F*P*F.t() + Q;


    
    // 5.2 correct predicted state & covariance matrix using measurement    
    cv::Mat y = z - h(µ_est);

    // compute Jacobi matrix of h at location a=µ_est    
    Mat H = (Mat_<float>(3, 3) << 2*µ1,  0.0, 0.0,
                                  0.0,  2*µ2, 0.0,
                                  0.0,   0.0, 2*µ3);
    
    /*
    Mat H = (Mat_<float>(3, 3) << 1,         1,      0,
                                  0,  -sin(µ2),      0,
                                  0,         0, cos(µ3));
    */
    //H = cv::Mat::eye(3, 3, CV_32FC1);

    // compute residual covariance matrix S
    cv::Mat S = H*P*H.t() + R;

    // compute Kalman gain matrix
    // the Kalman gain matrix tells us how strongly to correct
    // each dimension of the predicted state vector by the help
    // of the measurement
    cv::Mat K = P * H.t() * S.inv();

    // correct previously predicted new state vector
    µ_est = µ_est + K*y;

    // update uncertainty covariance matrix
    P = P - K*H*P;


    // 6. show ground truth & estimated trajectories

    // 6.1 update trajectories
    traj_gt.push_back          ( m2p(µ_gt)           );
    traj_est_predonly.push_back( m2p(µ_est_predonly) );
    traj_est_EKF.push_back     ( m2p(µ_est)          );
    const int MAX_NR_TRAJECTORY_POINTS = 100;
    if (traj_gt.size() > MAX_NR_TRAJECTORY_POINTS)
    {
      traj_gt          .erase(traj_gt.begin()          );
      traj_est_predonly.erase(traj_est_predonly.begin());
      traj_est_EKF     .erase(traj_est_EKF.begin()     );
    }

    // 6.2 draw trajectories
    for (unsigned int i = 1; i < traj_gt.size(); i++)
    {
      line(image, traj_gt.at(i - 1), traj_gt.at(i), COL_GT, 2);

      if (SHOW_NAIVE_ESTIMATION)
         line(image, traj_est_predonly.at(i - 1), traj_est_predonly.at(i), COL_NAIVE, 2);

      if (SHOW_EKF_ESTIMATION)
        line(image, traj_est_EKF.at(i - 1), traj_est_EKF.at(i), CV_RGB(255, 0, 0), 2);
    }

    // 6.3 draw current ground truth state & state estimates
    const int line_width = 2;
    const int circle_size = 10;
    circle(image, traj_gt.back(), circle_size, COL_GT, line_width);
    if (SHOW_NAIVE_ESTIMATION)
      circle(image, traj_est_predonly.back(), circle_size, COL_NAIVE, line_width);
    if (SHOW_EKF_ESTIMATION)
       circle(image, traj_est_EKF.back(), circle_size, COL_EKF, line_width);

    // 6.4 show current measurement?
    //     No, it does not make sense!
    //     If state is (x1,x2,x3) sensor value is about (x1*x1,x2*x2,x3*x3)!
    //circle(image, Point((int)z.at<float>(1, 0), (int)z.at<float>(2, 0)), circle_size, COL_MEASUREMENT, line_width);

    // 7. show visualization image
    imshow("Extended Kalman Filter Demo", image);

    // 8. display gt state vector and estimated state vectors
    if (SHOW_DEBUG_INFO)
    {
       printf("gt = (%.1f,%.1f,%.1f)\n",
          µ_gt.at<float>(0, 0), µ_gt.at<float>(1, 0), µ_gt.at<float>(2, 0));
       printf("est_predonly = (%.1f,%.1f,%.1f)\n",
          µ_est_predonly.at<float>(0, 0), µ_est_predonly.at<float>(1, 0), µ_est_predonly.at<float>(2, 0));
       printf("est = (%.1f,%.1f,%.1f)\n",
          µ_est.at<float>(0, 0), µ_est.at<float>(1, 0), µ_est.at<float>(2, 0));
    }

    // 9. update error estimates
    float N = (float)simulation_step;
    avg_error_µ_est_predonly = (avg_error_µ_est_predonly*N + (float)norm(µ_est_predonly - µ_gt)) / (N + 1.0f);
    avg_error_µ_est          = (avg_error_µ_est         *N + (float)norm(µ_est          - µ_gt)) / (N + 1.0f);
    printf("\n");
    printf("avg error est predonly = %.3f\n", avg_error_µ_est_predonly);
    printf("avg error EKF est      = %.3f\n", avg_error_µ_est);
    

    // 10. wait for user input
    printf("Press any key to go next simulation step!\n");
    int c = waitKey();
    if (c == 29) // ESC pressed?
      break;

    // 11. time goes by...
    simulation_step++;

  } // while


} // main
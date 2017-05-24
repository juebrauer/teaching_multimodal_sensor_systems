/// file: kalman_filter_projectile_demo.cpp
///
/// 4D state to be estimated here is the 2D position
/// and 2D velocity of a projectile: s = (x,vx,y,vy)
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

using namespace cv;

const int WIDTH = 800;
const int HEIGHT = 800;
const double g = 9.81;
const double dt = 0.1; // simulation step size in sec
const int cs = 3; // circle size (for projectile visualization)
const int fill = -1; // -1==fill 1=do not fill

#define drawPoint(x,y) Point((int)x,(int)(HEIGHT / 2 - y))



int main()
{
   // 1. initialize random number generator
   srand((unsigned int)time(NULL));
   

   // 2. prepare visualization of projectile's position
   Mat world(HEIGHT,WIDTH,CV_8UC3);
   world = Scalar(255, 255, 255);
   

   // 3. setup Kalman filter matrices
   Mat µ, P, F, B, Q, H, R;

   // initial guess of the state
   µ = (Mat_<float>(4, 1) <<  0.0, 0.0, 0.0, 0.0);

   P = (Mat_<float>(4, 4) <<  1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0);

   F = (Mat_<float>(4, 4) << 1.0, dt, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, dt,
                             0.0, 0.0, 0.0, 1.0);

   B = (Mat_<float>(4, 4) << 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0);

   Q = (Mat_<float>(4, 4) << 1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0);

   H = (Mat_<float>(4, 4) << 1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0);

   R = (Mat_<float>(4, 4) << 150.0, 0.0, 0.0, 0.0,
                             0.0, 50.0, 0.0, 0.0,
                             0.0, 0.0, 500.0, 0.0,
                             0.0, 0.0, 0.0, 50.0);

   kalman_filter_ndim myFilter(µ, P, F, B, Q, H, R);

   // 4. prepare random number generator for measurement noise
   cv::Mat mean_vec = (Mat_<float>(4, 1) << 0, 0, 0, 0);
   mvnrnd rnd_generator_measurement_noise = mvnrnd(mean_vec, R);


   // 5. prepare control vector
   cv::Mat u = (Mat_<float>(4, 1) << 0, 0, -0.5*g*dt*dt, -g*dt);


   // 6. prepare ground truth state vector s
   cv::Mat s = (Mat_<float>(4, 1) << 0,
                                     cos(M_PI / 4)*75.0,
                                     0,
                                     sin(M_PI / 4)*75.0);

   // 7. simulate some time steps of the projectile's trajectory
   for (double t = dt; t <= 100.0; t += dt)
   {
      printf("simulation time=%.2f\n", t);

      // 7.1 simulate new projectile's state
      // x  = x + vx*dt;
      // vx = vx;      
      // y  = y + vy*dt - (g/2.0)*(dt*dt);
      // vy = vy - g*dt;
      s = F*s + B*u;

      // 7.2 get projectile position (px,py) from s
      double px = s.at<float>(0, 0);
      double py = s.at<float>(2, 0);
      

      // 7.3 if the projectile touches the ground, we stop the simulation
      if (py <= 0.0)
      {
         printf("Touch down!\n");
         _getch();
         break;
      }


      // 7.4 simulate noisy sensor
      Mat noise_measurement = rnd_generator_measurement_noise.get_next_random_vector();
      Mat z = H*s + noise_measurement;
      

      // 7.5 Kalman filter our state estimate
      myFilter.predict( u );
      myFilter.correct_by_measurement( z );
      
      
      // 7.6 visualize
      //      projectile's position
      //      measured position (noisy)
      //      KF estimated position      
      circle(world, drawPoint(px, py), cs, CV_RGB(0,0,0), fill);

      double sx = z.at<float>(0, 0);
      double sy = z.at<float>(2, 0);
      circle(world, drawPoint(sx, sy), cs, CV_RGB(255,0,0), fill);

      Mat est_s = myFilter.get_current_state_estimate();
      double est_x = est_s.at<float>(0, 0);
      double est_y = est_s.at<float>(2, 0);
      circle(world, drawPoint(est_x, est_y), cs, CV_RGB(0, 0, 255), fill);

      imshow("Ballistic trajectory of a projectile", world);

      // 7.7 continue with next simulation step if user presses a key
      waitKey(0);    

   }

   printf("Simulation finished. Press a key to exit.\n");
   _getch();

} // main
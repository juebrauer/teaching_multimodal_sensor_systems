/// 2D Robot class implementation file
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "Robot.h"

#include <conio.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "mvnrnd.h"   // for random number generator following a n-dimensional normal distribution


// compute a random double value in the interval [min,max]
double get_rand_val_from_interval(double min, double max)
{
  // compute random int in [0,100]
  int rndval = rand() % 101; // rndval is in [0,100]

  // convert to double value in [0,1]
  double rndvald = (double)rndval / 100.0;

  // map to position in interval [min,max]
  double rndintervalpos = min + rndvald * (max - min);

  return rndintervalpos;
} // get_rand_val_from_interval



Robot::Robot( string             name,
              int                radius,
              Point2d            start_pos,
              double             start_orientation,
              vector<double>     sensor_angles,
              vector<double>     sensor_distances,
              bool               noisy_distance_sensors,
              int                world_height,
              int                world_width
            )
{
  // 1. store robot attribute information
  this->name                            = name;
  this->radius                          = radius;  
  this->sensor_angles                   = sensor_angles;
  this->sensor_distances                = sensor_distances;
  this->noisy_distance_sensors          = noisy_distance_sensors;
  this->nr_sensors                      = (int) sensor_angles.size();
  this->world_height                    = world_height;
  this->world_width                     = world_width;

  // 2. store robot (x,y) position and orientation angle as 3D vector
  this->robot_pos_and_orient_real              = (Mat_<double>(3, 1) << start_pos.x, start_pos.y, start_orientation);
  this->robot_pos_and_orient_naively_estimated = this->robot_pos_and_orient_real.clone();

  // 3. create occupancy grids
  occupancy_grid* o1 = new occupancy_grid("Grid #1: naive est pos/orient + noisy sensors",          world_width, world_height);
  occupancy_grid* o2 = new occupancy_grid("Grid #2: KF est pos/orient + KF filtered noisy sensors", world_width, world_height);
  my_occupancy_grids.push_back( o1 );
  my_occupancy_grids.push_back( o2 );

  // 4. prepare normal distributions for simulating sensor noise
  double µ            = 0.0;
  double sensor_noise = 5.0;
  my_normal_distribution_for_sensor_noise = std::normal_distribution<double>(µ, sensor_noise);

  // 5. create a 1D Kalman filter for each distance sensor
  for (int sensor_nr = 0; sensor_nr < nr_sensors; sensor_nr++)
  {
    double init_µ            = sensor_distances[sensor_nr];  // start estimate for the sensor value
    double init_sigma        = 1.0;                          // start uncertainty about this sensor value
    double process_noise     = 0.1;                          // how much randomness is there in the distance to walls?
    double measurement_noise = sensor_noise;                 // how much randomness is there in the sensor data?
    kalman_filter_1d* ptr = new kalman_filter_1d( init_µ, init_sigma, process_noise, measurement_noise );
    kalman_filter_per_sensor.push_back( ptr );
  }
  
  // 6. initialize error variables for tracking the average error
  //    between ground truth sensor values and
  //    ... noisy sensor values
  //    ... Kalman filtered sensor values
  moving_avg_error_update_counter                = 0;
  moving_avg_error_noisy_sensors                 = 0.0;
  moving_avg_error_Kalman_filtered_noisy_sensors = 0.0;
    
  // 7. prepare all the matrices we need for Kalman filtering position and orientation
  setup_real_and_Kalman_filter_model_matrices();

} // constructor of class Robot





vector<Point2d> Robot::get_trajectory_ground_truth()
{
  return trajectory_ground_truth;
}



vector<Point2d> Robot::get_trajectory_naively_estimated()
{
  return trajectory_naively_estimated;
}


vector<Point2d> Robot::get_trajectory_KF_estimated()
{
  return trajectory_KF_estimated;
}


vector<occupancy_grid*> Robot::get_occupancy_grids()
{
  return my_occupancy_grids;
}


double Robot::get_radius()
{
  return radius;
}



Point2d Robot::get_position_real()
{
  return Point2d(robot_pos_and_orient_real.at<double>(0, 0), robot_pos_and_orient_real.at<double>(1, 0) );
}


double Robot::get_orientation_real()
{
  return robot_pos_and_orient_real.at<double>(2, 0);
}


Point2d Robot::get_position_naively_estimated()
{
  return Point2d(robot_pos_and_orient_naively_estimated.at<double>(0, 0), robot_pos_and_orient_naively_estimated.at<double>(1, 0));
}


double Robot::get_orientation_naively_estimated()
{
  return robot_pos_and_orient_naively_estimated.at<double>(2, 0);
}


Point2d Robot::get_position_KF_estimated()
{
  cv::Mat µ_est = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
  return Point2d(µ_est.at<double>(0, 0), µ_est.at<double>(1, 0));
}


double Robot::get_orientation_KF_estimated()
{
  cv::Mat µ_est = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
  return µ_est.at<double>(2, 0);
}



int Robot::get_nr_sensors()
{
  return nr_sensors;
}



vector<double> Robot::get_sensor_values()
{
  return sensor_values;
}



vector<double> Robot::get_sensor_angles()
{
  return sensor_angles;
}



vector<double> Robot::get_sensor_distances()
{
  return sensor_distances;
}


/// return the direction of the <sensor_nr>th sensor ray relative
/// to the robot's orientation
Point2d Robot::get_sensor_ray_direction_vector(double robot_orientation, int sensor_nr)
{
  // get sensor orientation relative to robots orientation
  double sensor_angle = sensor_angles[sensor_nr];

  // map robot angle + sensor_angle to a direction vector
  double sensor_dx = cos(robot_orientation + sensor_angle);
  double sensor_dy = sin(robot_orientation + sensor_angle);

  return Point2d(sensor_dx, sensor_dy);
}



void Robot::compute_sensor_values(Mat world)
{
  // 1. clear old distance sensor values / object detection results
  sensor_values.clear();
  sensor_object_detected.clear();
  sensor_values_Kalman_filtered.clear();
  sensor_object_detected_Kalman_filtered.clear();

  // 2. for all distance sensors the robot has ...
  for (int sensor_nr = 0; sensor_nr < nr_sensors; sensor_nr++)
  {
    // 2.1 get (x,y) coords of current robot position
    double x = robot_pos_and_orient_real.at<double>(0,0);
    double y = robot_pos_and_orient_real.at<double>(1,0);

    // 2.2 get sensor orientation relative to robots orientation
    double sensor_angle = sensor_angles[sensor_nr];

    // 2.3 map robot angle + sensor angle to a direction vector
    double sensor_dx = cos(robot_pos_and_orient_real.at<double>(2,0) + sensor_angle);
    double sensor_dy = sin(robot_pos_and_orient_real.at<double>(2,0) + sensor_angle);

    // 2.4 compute sensor start position
    double sensor_startx = x + sensor_dx * radius;
    double sensor_starty = y + sensor_dy * radius;

    // 2.5 now move from sensor start position into sensor direction
    //     till we reach the maximum distance or hit an obstacle in the world (white pixel)
    // get maximum sensor distance
    double sensor_max_dist = sensor_distances[sensor_nr];
    int step;
    for (step = 0; step < sensor_max_dist; step++)
    {
      double sx = sensor_startx + step*sensor_dx;
      double sy = sensor_starty + step*sensor_dy;

      // get value of world pixel at the sensor ray position (sx,sy)
      Vec3b pixel_color = world.at<Vec3b>((int)sy, (int)sx);

      // is it black or white?
      if ((pixel_color.val[0] == 0) && (pixel_color.val[1] == 0) && (pixel_color.val[2] == 0))
      {
        // black pixel, so continue
        continue;
      }
      else
      {        
        break;
      }

    } // for (move along sensor line)


    int ground_truth_sensor_value = step;
    int noisy_sensor_value        = ground_truth_sensor_value;

        
    // 2.6 simulate noise distance sensors?
    if (noisy_distance_sensors)
    {
      // compute some random noise value using an uniform distribution
      //int rnd_int_val = (rand() % 40 + 1) - 20;

      // compute some random noise value using the prepared Gaussian sensor noise distribution
      double rnd_val = my_normal_distribution_for_sensor_noise( my_rnd_generator );

      // add noise to sensor value
      noisy_sensor_value += (int) round(rnd_val);
    }


    //////////////////////////////////////////////////////////
    // 2.7 Update Kalman filter estimate for this sensor value

    
    // get the Kalman filter object for this sensor
    kalman_filter_1d* kf_for_current_sensor = kalman_filter_per_sensor[ sensor_nr ];

    // control signal will be:
    // assume we have no information how the distance sensor signal will change
    double u = 0;

    // measurement signal wil be:
    double z = noisy_sensor_value;

    // predict step:
    // if use u=0, we assume, that the distance value does not change
    // this brings us some latency
    kf_for_current_sensor->predict(u);

    // correct-by-measurement step:
    // the input for the Kalman filter correct step is the noisy sensor value    
    kf_for_current_sensor->correct_by_measurement(z);

    // now get the Kalman filter estimate for the noisy sensor value
    double sensor_value_kf = kf_for_current_sensor->get_current_state_estimate();

    //
    //////////////////////////////////////////////////////////



    // 2.8 object detected according to this noisy sensor ray?
    bool object_detected;
    if (noisy_sensor_value < sensor_max_dist)
      object_detected = true;
    else
      object_detected = false;

    // 2.9 object detected according to this Kalman filtered sensor ray?
    bool object_detected_Kalman_filtered;
    if (sensor_value_kf < sensor_max_dist)
      object_detected_Kalman_filtered = true;
    else
      object_detected_Kalman_filtered = false;

    // 2.10 store noisy sensor values & information whether we detected an object (<=> ray stops before its max distance)
    //     according to the noisy sensors
    sensor_values.push_back(noisy_sensor_value);
    sensor_object_detected.push_back(object_detected);

    // 2.11 store Kalman filtered sensor values & information whether we detected an object (<=> ray stops before its max distance)
    //     according to the Kalman filtered noisy sensor values
    sensor_values_Kalman_filtered.push_back(sensor_value_kf);
    sensor_object_detected_Kalman_filtered.push_back(object_detected_Kalman_filtered);


    // 2.12 update moving average errors of {noisy sensor values | Kalman filtered noisy sensor values} vs. ground truth sensor value
    double N = (double)moving_avg_error_update_counter;
    double diff;

    diff = abs(noisy_sensor_value - ground_truth_sensor_value);    
    moving_avg_error_noisy_sensors = (diff + N*moving_avg_error_noisy_sensors) / (N+1.0);

    diff = abs(sensor_value_kf - ground_truth_sensor_value);
    moving_avg_error_Kalman_filtered_noisy_sensors = (diff + N*moving_avg_error_Kalman_filtered_noisy_sensors) / (N + 1.0);

    moving_avg_error_update_counter++;

  } // for (sensor_nr)
  
} // compute_sensor_values


void Robot::show_state_information()
{
  std::cout << "   avg error of noisy sensors          : " << moving_avg_error_noisy_sensors                 << std::endl;
  std::cout << "   avg error of Kalman filtered sensors: " << moving_avg_error_Kalman_filtered_noisy_sensors << std::endl;
}


void Robot::update(Mat world)
{
  // 1. compute new distance sensor values
  //    note: motion encoder / orientation sensor values
  //    are computed in move() and turn()
  compute_sensor_values(world);
    

  // 2. update occupancy grids by information of all sensors
  for (int sensor_nr = 0; sensor_nr <= 1; sensor_nr++)
  {
    Point2d sensor_ray_dir_vec;
    
    // update occupancy grid where we use the naively estimated pos/orientation + noisy sensor value
    cv::Mat µ_naive = robot_pos_and_orient_naively_estimated;
    double orientation_naively_estimated = µ_naive.at<double>(2, 0);

    sensor_ray_dir_vec = get_sensor_ray_direction_vector(orientation_naively_estimated, sensor_nr);
    my_occupancy_grids[0]->update(
      get_position_naively_estimated(),
      sensor_ray_dir_vec,
      radius + sensor_values[sensor_nr],
      sensor_object_detected[sensor_nr]
      );

    // update occupancy grid where we use the Kalman filtered pos/orientation + Kalman filtered noisy sensor value
    cv::Mat µ = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
    double orientation_due_to_KF = µ.at<double>(2, 0);

    sensor_ray_dir_vec = get_sensor_ray_direction_vector(orientation_due_to_KF, sensor_nr);
    my_occupancy_grids[1]->update(
      get_position_KF_estimated(),
      sensor_ray_dir_vec,
      radius + sensor_values_Kalman_filtered[sensor_nr],
      sensor_object_detected_Kalman_filtered[sensor_nr]
      );
  }
  

  // 3. compute driving behavior:
  //    i.e. set variables cmd_pixel_to_drive & cmd_turn_angle
  double sensor_val0 = sensor_values[0];
  double sensor_val1 = sensor_values[1];  
  double rnd_val = get_rand_val_from_interval(-M_PI/100, +M_PI/100);
  double cmd_turn_angle = 0.0;
  double cmd_pixel_to_drive = 0.0;
  if ((sensor_val0 < 10) || (sensor_val1 < 10))
  { 
    // turn left or right?
    if (sensor_val0 < sensor_val1)
    {
      cmd_turn_angle = M_PI / 16 + rnd_val;
    }
    else
    {
      cmd_turn_angle = -M_PI / 16 + rnd_val;
    }
  }
  else
  {
    // move forward in current direction
     cmd_pixel_to_drive = 1.0;
  }


  // 4. simulate state change of
  //    robot and noisy state measurement.
  //    I.e. turn and move robot
  //    according to desired control
  //    signal <cmd_pixel_to_drive, cmd_turn_angle>
  //    but also add process noise!

  if (false)
  {
    double x = robot_pos_and_orient_real.at<double>(0, 0);
    double y = robot_pos_and_orient_real.at<double>(1, 0);
    double orientation = robot_pos_and_orient_real.at<double>(2, 0);
    printf("robot state before is (x,y,orientation)=(%.1f,%.1f,%.1f)\n", x, y, orientation);
    _getch();
  }

  // compute real orientation as direction vector
  double dir_x = cos(robot_pos_and_orient_real.at<double>(2,0));
  double dir_y = sin(robot_pos_and_orient_real.at<double>(2,0));

  // compute desired coordinate change vector
  double move_x = dir_x * cmd_pixel_to_drive;
  double move_y = dir_y * cmd_pixel_to_drive;

  // compute true state change vector = desired state change vector + 3D random vector
  Mat desired_state_change_vec = (Mat_<double>(3, 1) << move_x, move_y, cmd_turn_angle);
  Mat rnd_vec = rnd_generator_process_noise->get_next_random_vector();
  Mat state_change_vec = desired_state_change_vec + rnd_vec;
  
  // update real state
  robot_pos_and_orient_real += state_change_vec;
  
  // show new robot state (debug)
  if (false)
  {    
    printf("desired_state_change_vec is (%.1f,%.1f,%.1f)\n", desired_state_change_vec.at<double>(0, 0), desired_state_change_vec.at<double>(1, 0), desired_state_change_vec.at<double>(2, 0));
    printf("rnd_vec for process noise is (%.1f,%.1f,%.1f)\n", rnd_vec.at<double>(0,0), rnd_vec.at<double>(1,0), rnd_vec.at<double>(2,0) );
    printf("final state_change_vec is (%.1f,%.1f,%.1f)\n", state_change_vec.at<double>(0, 0), state_change_vec.at<double>(1, 0), state_change_vec.at<double>(2, 0));
    printf("robot state after is (x,y,orientation)=(%.1f,%.1f,%.1f)\n",
      robot_pos_and_orient_real.at<double>(0, 0), robot_pos_and_orient_real.at<double>(1, 0), robot_pos_and_orient_real.at<double>(2, 0));
    _getch();
  }

  // simulate noisy measurement of real position & orientation
  cv::Mat rnd_vec_measurement = rnd_generator_measurement_noise->get_next_random_vector();
  cv::Mat z = robot_pos_and_orient_real + rnd_vec_measurement;

  if (false)
  {
    printf("robot state is (x,y,orientation)=(%.2f,%.2f,%.2f)\n",
      robot_pos_and_orient_real.at<double>(0, 0), robot_pos_and_orient_real.at<double>(1, 0), robot_pos_and_orient_real.at<double>(2, 0));
    printf("rnd_vec_measurement=(%.2f,%.2f,%.2f)\n",
      rnd_vec_measurement.at<double>(0, 0), rnd_vec_measurement.at<double>(1, 0), rnd_vec_measurement.at<double>(2, 0));
    printf("noisy measurement vector of state is (x,y,orientation)=(%.2f,%.2f,%.2f)\n",
      z.at<double>(0, 0), z.at<double>(1, 0), z.at<double>(2, 0));
    _getch();
  }



  
  // 5. update naive state estimate
  //    that uses just the
  //    noisy state sensor measurement
     
  // we totally believe the measurement! :) --> :(
  robot_pos_and_orient_naively_estimated = z;
  
  if (false)
  {
    printf("robot state is (x,y,orientation)=(%.2f,%.2f,%.2f)\n",
      robot_pos_and_orient_real.at<double>(0, 0), robot_pos_and_orient_real.at<double>(1, 0), robot_pos_and_orient_real.at<double>(2, 0));
    printf("robot_pos_and_orient_naively_estimated=(%.2f,%.2f,%.2f)\n",
      robot_pos_and_orient_naively_estimated.at<double>(0, 0), robot_pos_and_orient_naively_estimated.at<double>(1, 0), robot_pos_and_orient_naively_estimated.at<double>(2, 0));
    _getch();
  }


  // 6. update Kalman filter estimate
  //    for 2D position and orientation = 3D vector

  // 6.1 combine control signal in a 3D vector u
  double current_KF_estimated_orientation =
    kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate().at<double>(2, 0); // get 3rd coordinate
  double cmd_move_x = cmd_pixel_to_drive * cos(current_KF_estimated_orientation);
  double cmd_move_y = cmd_pixel_to_drive * sin(current_KF_estimated_orientation);
  cv::Mat u = (Mat_<double>(3, 1) << cmd_move_x, cmd_move_y, cmd_turn_angle);

  if (false)
  {
    cv::Mat µ_KF = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
    printf("\nBefore prediction:\n");
    printf("\tKF state=(%.2f,%.2f,%.2f)\n", µ_KF.at<double>(0, 0), µ_KF.at<double>(1, 0), µ_KF.at<double>(2, 0));
    printf("\tcontrol vector u=(%.2f,%.2f,%.2f)\n", u.at<double>(0, 0), u.at<double>(1, 0), u.at<double>(2, 0));
    _getch();
  }

 
  // 6.2 predict next position and orientation of robot using only control signal u
  kalman_filter_for_pos_and_orientation_estimation->predict( u );

  if (false)
  {
    printf("\nAfter prediction:\n");
    cv::Mat µ_KF = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
    printf("\tKF state=(%.2f,%.2f,%.2f)\n", µ_KF.at<double>(0, 0), µ_KF.at<double>(1, 0), µ_KF.at<double>(2, 0));
    _getch();
  }

  
  // 6.3 correct position and orientation estimate using noisy measurement vector
  kalman_filter_for_pos_and_orientation_estimation->correct_by_measurement( z );
  
  if (false)
  {
    printf("\nAfter correction:\n");
    printf("\tz=(%.2f,%.2f,%.2f)\n", z.at<double>(0, 0), z.at<double>(1, 0), z.at<double>(2, 0));
    cv::Mat µ_KF = kalman_filter_for_pos_and_orientation_estimation->get_current_state_estimate();
    printf("\ttrue state=(%.2f,%.2f,%.2f)\n", robot_pos_and_orient_real.at<double>(0, 0), robot_pos_and_orient_real.at<double>(1, 0), robot_pos_and_orient_real.at<double>(2, 0));
    printf("\tKF state=(%.2f,%.2f,%.2f)\n", µ_KF.at<double>(0, 0), µ_KF.at<double>(1, 0), µ_KF.at<double>(2, 0));
    _getch();
  }



  // 7. trajectories updates

  // 7.1 update ground truth (real) 2D trajectory
  trajectory_ground_truth.push_back( get_position_real() );

  // 7.2 update naively estimated 2D trajectory - estimated based on just control signals
  trajectory_naively_estimated.push_back( get_position_naively_estimated() );

  // 7.3 update KF estimated 2D trajectory - estimated based on control signals & measurements
  trajectory_KF_estimated.push_back( get_position_KF_estimated() );

  // 7.4 store maximally the last N trajectory points
  const int N = 200;
  if (trajectory_ground_truth.size() > N)
  {
    trajectory_ground_truth.erase(trajectory_ground_truth.begin());
    trajectory_naively_estimated.erase(trajectory_naively_estimated.begin());
    trajectory_KF_estimated.erase(trajectory_KF_estimated.begin());
  }


} // update






void Robot::setup_real_and_Kalman_filter_model_matrices()
{

  // 1. initialize vectors and covariance matrices

  // 1.1 initialize 3D state vector µ:
  //     we assume, we know the start pos & orientation of our 2D robot
  cv::Mat µ = robot_pos_and_orient_real.clone();

  // 1.2 covariance 3x3 matrix P: initial uncertainty about the state
  //     we are quite sure about the start state, for this we
  //     initialize P with small variance values about each state dimension
  cv::Mat P = (Mat_<double>(3, 3) << 0.8, 0.0, 0.0, 
                                     0.0, 0.8, 0.0,
                                     0.0, 0.0, 0.8);

  // 1.3 state transition 3x3 matrix F: describes the system dynamics
  //     the robot does not change its position or orientation
  //     if there is no control signal
  cv::Mat F_true = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);
  
  // 1.4 control matrix B: describes how control signals affect the new state
  //     the control signal is here u=(delta_x,delta_y,delta_orientation)
  cv::Mat B_true = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);


  // 1.5 process noise covariance matrix: describes randomness of state
  //     transitions per dimension
  cv::Mat Q_true = (Mat_<double>(3, 3) <<   0.01, 0.0,  0.0,
                                            0.0,  0.01, 0.0,
                                            0.0,  0.0,  0.01);


  // 1.6 measurement matrix: describes how states are mapped by our sensors
  //     to measurements
  cv::Mat H_true = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);


  // 1.7 measurement noise covariance matrix: describes randomness of
  //     measurements per dimension
  cv::Mat R_true = (Mat_<double>(3, 3) << 500.0,   0.0, 0.0,
                                            0.0, 500.0, 0.0,
                                            0.0,   0.0, 0.01);


  // 2. prepare random number generators for simulating
  //    process & measurement noise  
  cv::Mat mean_vec = (Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

  // 2.1 prepare random number generator for process noise
  rnd_generator_process_noise = new mvnrnd(mean_vec, Q_true);

  // 2.2 prepare random number generator for measurement noise
  rnd_generator_measurement_noise = new mvnrnd(mean_vec, R_true);



    
  // 3. For experimenting with different assumed model matrices,
  //    change the multiplication factors here:  
  
  // try 1.2, 0.8, then 1.01:
  //          small wrong error in state transition matrix
  //          makes a big difference!
  //          Reason: arguments in state vector change
  //                  exponentially fast
  //          --> F has a central role in the Kalman Filter!
  cv::Mat F_model = F_true * 1.0;

  // try 5.0: assumed control signal influence is too high
  //cv::Mat B_model = B_true * 1.02;
  cv::Mat B_model = B_true * 1.0;

  // try 1000.0: assumed process noise is too high -->
  //             we believe more the sensor data
  //cv::Mat Q_model = Q_true * 2.0;
  cv::Mat Q_model = Q_true * 1.0;

  // try 1.25: assumed measurement matrix gives too large sensor values -->
  //           estimated state coordinates are too small
  // try 0.75: assumed measurement matrix gives too small sensor values -->
  //           estimated state coordinates are too large
  //cv::Mat H_model = H_true * 1.02;
  cv::Mat H_model = H_true * 1.0;

  // try 1000.0: assumed measurement noise is too high -->
  //             we believe more the predicted state
  //            (as R_model becomes too large, state estimation
  //             becomes more and more similar to the
  //             naive approach)
  //cv::Mat R_model = R_true * 0.7;
  cv::Mat R_model = R_true * 1.0;
  
    
  // 4. prepare a n-dimensional Kalman filter for estimating the µ=(x,y,theta)
  //    position and orientation of the 2D robot  
  kalman_filter_for_pos_and_orientation_estimation =
    new kalman_filter_ndim(µ, P, F_model, B_model, Q_model, H_model, R_model);


} // setup_real_and_Kalman_filter_model_matrices

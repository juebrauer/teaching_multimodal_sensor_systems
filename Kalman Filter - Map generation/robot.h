/// 2D Robot class header file
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#pragma once

#include <random> // for Gaussian random number generator

#include "opencv2/core.hpp"

#include "occupancy_grid.h"
#include "kalman_filter_1d.h"
#include "kalman_filter_ndim.h"
#include "mvnrnd.h"

using namespace std;
using namespace cv;

class Robot
{
  private:

    string                            name;
    double                            radius;
    cv::Mat                           robot_pos_and_orient_real;
    cv::Mat                           robot_pos_and_orient_naively_estimated;
    Point2d                           pos_estimated;
    double                            orientation_estimated;
    vector<double>                    sensor_angles;
    vector<bool>                      sensor_object_detected;
    vector<double>                    sensor_distances;
    int                               nr_sensors;
    vector<double>                    sensor_values;    
    Point2d                           sensor_val1_move;
    Point2d                           sensor_val2_move;
    double                            sensor_val_turn;
    int                               world_height;
    int                               world_width;
    vector<occupancy_grid*>           my_occupancy_grids;
    bool                              noisy_distance_sensors;
    vector<Point2d>                   trajectory_ground_truth;
    vector<Point2d>                   trajectory_naively_estimated;
    vector<Point2d>                   trajectory_KF_estimated;
    std::default_random_engine        my_rnd_generator;
    std::normal_distribution<double>  my_normal_distribution_for_sensor_noise;

    // Variables needed for Kalman filtering sensor values:
    vector<double>                    sensor_values_Kalman_filtered;
    vector<kalman_filter_1d*>         kalman_filter_per_sensor;
    vector<bool>                      sensor_object_detected_Kalman_filtered;
    double                            moving_avg_error_noisy_sensors;
    double                            moving_avg_error_Kalman_filtered_noisy_sensors;
    int                               moving_avg_error_update_counter;

    // Variables needed for Kalman filtering noisy position and orientation measurements
    void                              setup_real_and_Kalman_filter_model_matrices();    
    mvnrnd*                           rnd_generator_process_noise;
    mvnrnd*                           rnd_generator_measurement_noise;
    kalman_filter_ndim*               kalman_filter_for_pos_and_orientation_estimation;


  public:

                                Robot::Robot(string             name,
                                             int                radius,
                                             Point2d            start_pos,
                                             double             start_orientation,
                                             vector<double>     sensor_angles,
                                             vector<double>     sensor_distances,
                                             bool               noisy_distance_sensors,
                                             int                world_height,
                                             int                world_width
                                            );

    void                        compute_sensor_values(Mat world);
    void                        update(Mat world);
    void                        show_state_information();

    Point2d                     get_position_real();
    Point2d                     get_position_naively_estimated();
    Point2d                     get_position_KF_estimated();
    double                      get_orientation_real();
    double                      get_orientation_naively_estimated();
    double                      get_orientation_KF_estimated();
    double                      get_radius();
    int                         get_nr_sensors();
    vector<double>              get_sensor_values();
    vector<double>              get_sensor_angles();
    vector<double>              get_sensor_distances();
    vector<Point2d>             get_trajectory_ground_truth();
    vector<Point2d>             get_trajectory_naively_estimated();
    vector<Point2d>             get_trajectory_KF_estimated();
    Point2d                     get_sensor_ray_direction_vector(double robot_orientation, int sensor_nr);
    vector<occupancy_grid*>     get_occupancy_grids();
    
}; // class Robot

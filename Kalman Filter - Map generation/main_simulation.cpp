/// Robot simulation
///
/// Simulation loop for a simulated 2D world
/// where a 2D robot instance of class Robot
/// can drive through this world.
///
/// Note: the world can be painted with any
///       drawing program as a RGB picture.
///       Black pixels (R=0,G=0,B=0) will
///       be considered as free driveable spaces.
///       All other pixel colors will be
///       considered as obstacles / walls. 
///       See Robot::compute_sensor_values()
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <conio.h>
#include <time.h>

#include "params.h"
#include "Robot.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;


double compute_avg_gridcell_error(occupancy_grid* o1, occupancy_grid* o2)
{
  double sum_errors = 0.0;
  double** g1 = o1->get_the_grid();
  double** g2 = o2->get_the_grid();
  for (int y = 0; y < o1->get_height(); y++)
  {
    for (int x = 0; x < o1->get_width(); x++)
    {
      // compute current error at (x,y)
      double curr_error = abs(g1[y][x] - g2[y][x]);

      // sum up all errors
      sum_errors += curr_error;
    }
  }

  // compute average error
  double avg_error = sum_errors / (double)(o1->get_height() * o1->get_width());

  return avg_error;

} // compute_avg_gridcell_error




int main()
{
  // 1. start each simulation with another initializiation
  //    of the pseudo random number generator?
  bool use_time_as_startpoint_for_pseudo_number_generator = false;
  if (use_time_as_startpoint_for_pseudo_number_generator)
    srand( (int) time(NULL) );


  // 2. load image of world: change this to the folder where you store the world image!
  string img_filename = "world4.png";
  Mat world = imread(img_filename);


  // 3. image load error?
  if ((world.cols == 0) && (world.rows == 0))
  {
    cout << "Error! Could not read the image file: " << img_filename << endl;
    _getch();
    return -1;
  }


  // 4. get world image dimensions
  int WORLD_WIDTH  = world.cols;
  int WORLD_HEIGHT = world.rows;


  // 5. compute ground truth occupancy grid from ground truth world image
  occupancy_grid* occupancy_grid_ground_truth = new occupancy_grid("ground truth grid", WORLD_WIDTH, WORLD_HEIGHT);
  double** gt_grid = occupancy_grid_ground_truth->get_the_grid();
  for (int y = 0; y < WORLD_HEIGHT; y++)
  {
    for (int x = 0; x < WORLD_WIDTH; x++)
    {
      // get value of pixel at (x,y)
      Vec3b pixel_color = world.at<Vec3b>(y,x);

      // map color to grid value
      double prob = 1.0;
      if ((pixel_color.val[0] == 0) && (pixel_color.val[1] == 0) && (pixel_color.val[2] == 0))
        prob = 0.0;

      // store value in ground truth grid
      gt_grid[y][x] = prob;

    } // for (x)
  } // for (y)


  // 6. prepare an image (where we will draw the robot and the world)
  Mat image(WORLD_HEIGHT, WORLD_WIDTH, CV_8UC3);


  // 7. create a robot
  vector<double> sensor_angles, sensor_distances;
  sensor_angles.push_back(-M_PI/4);
  sensor_angles.push_back(+M_PI/4);
  sensor_distances.push_back(50);
  sensor_distances.push_back(50);
  string  robot_name             = "R2D2";
  int     robot_radius           = 10;                                        // in pixel
  Point2d robot_startpos         = Point(WORLD_HEIGHT / 2, WORLD_WIDTH / 2);  // in pixel
  double  robot_startorientation = M_PI / 4;                                  // in radians
  bool    noisy_sensors          = true;
  Robot r1(robot_name,
           robot_radius,
           robot_startpos,
           robot_startorientation,
           sensor_angles,
           sensor_distances,
           noisy_sensors,
           WORLD_HEIGHT,
           WORLD_WIDTH);      


  // 8. simulation loop
  bool exit = false;
  int simulation_step = 0;
  double pos_error_naively_estimated = 0.0;
  double pos_error_KF_estimated      = 0.0;
  while (!exit)
  {
    // 8.1 show simulation step nr on console
     if ((VISUALIZATION) || (simulation_step % 100 == 0))
     {
        system("cls");
        printf("\n\nsimulation step : %d\n\n", simulation_step);
     }

    // 8.2 move the robot
    r1.update(world);

    // 8.3 update position error as a moving average,
    // see https://en.wikipedia.org/wiki/Moving_average#Cumulative_moving_average
    double n = (double)simulation_step;

    double curr_pos_error_naively_estimated = norm(r1.get_position_naively_estimated() - r1.get_position_real());
    pos_error_naively_estimated = (curr_pos_error_naively_estimated + n*pos_error_naively_estimated) / (n + 1);
    double curr_pos_error_KF_estimated = norm(r1.get_position_KF_estimated() - r1.get_position_real());
    pos_error_KF_estimated = (curr_pos_error_KF_estimated + n*pos_error_KF_estimated) / (n + 1);

    // 8.4 initialize image with world image
    world.copyTo(image);
                
    // 8.5 draw robot's real & estimated trajectory driven so far
    vector<Point2d> traj_gt        = r1.get_trajectory_ground_truth();
    vector<Point2d> traj_est_naive = r1.get_trajectory_naively_estimated();
    vector<Point2d> traj_est_KF    = r1.get_trajectory_KF_estimated();
    for (unsigned int i = 1; i < traj_gt.size(); i++)
    {
      line(image, traj_gt.at(i - 1),        traj_gt.at(i),        CV_RGB(255, 255, 255), 1);
      //line(image, traj_est_naive.at(i - 1), traj_est_naive.at(i), CV_RGB(128, 128, 128), 1);
      //line(image, traj_est_KF.at(i - 1),    traj_est_KF.at(i),    CV_RGB(  0, 255,   0), 1);
    }

    // 8.6 show robot's position as a circle
    double  r = r1.get_radius();
    Point2d pos = r1.get_position_real();
    circle(image, pos, (int)r, CV_RGB(255, 0, 0), 1);

    // 8.7 show robot's real orientation by a line
    double  ori   = r1.get_orientation_real();
    double  dx    = cos(ori);
    double  dy    = sin(ori);    
    line(image, pos, pos + Point2d(dx*r, dy*r), CV_RGB(255, 0, 0), 1);
    
    // 8.8 show robot's KF orientation by a line
    double  ori2 = r1.get_orientation_KF_estimated();
    double  dx2 = cos(ori2);
    double  dy2 = sin(ori2);
    line(image, pos, pos + Point2d(dx2*r, dy2*r), CV_RGB(0, 255, 0), 1);
            
    // 8.9 get all the sensor values
    vector<double> sensor_values = r1.get_sensor_values();

    // 8.10 draw all the sensor rays
    for (int sensor_nr = 0; sensor_nr < r1.get_nr_sensors(); sensor_nr++)
    {
      double sensor_value = sensor_values[sensor_nr];

      // get (x,y) coords of current robot position
      double x = pos.x;
      double y = pos.y;

      // get sensor orientation relative to robots orientation
      double sensor_angle = sensor_angles[sensor_nr];

      // map robot angle + sensor_angle to a direction vector
      double sensor_dx = cos(r1.get_orientation_real() + sensor_angle);
      double sensor_dy = sin(r1.get_orientation_real() + sensor_angle);

      // compute sensor start position
      double sensor_startx = x + sensor_dx * r1.get_radius();
      double sensor_starty = y + sensor_dy * r1.get_radius();

      // compute sensor ray end position
      double sensor_endx = sensor_startx + sensor_dx * sensor_value;
      double sensor_endy = sensor_starty + sensor_dy * sensor_value;

      // draw sensor ray line
      line(image,
           Point((int)sensor_startx, (int)sensor_starty),
           Point((int)sensor_endx, (int)sensor_endy),
           CV_RGB(255, 255, 0), 1);

    } // for (draw all sensor rays)

    // 8.11 show simulation step nr & position error in image
    char txt[100];
    sprintf_s(txt, "simulation step %d", simulation_step);
    putText(image,
      txt,
      Point(20, 50),
      FONT_HERSHEY_SIMPLEX, 0.7, // font face and scale
      CV_RGB(255,0,0), // white
      1); // line thickness and type

    // 8.12 show current state of world
    if ((VISUALIZATION) || (simulation_step % 100 == 0))
    {
       imshow("Robot Simulator", image);

       // 8.13 let the robot also ouptut some state information (whatever it wants)
       r1.show_state_information();

       if (SAVE_VISUALIZATION_IMAGES)
       {
          char fname[500];
          sprintf_s(fname, "V:\\tmp\\world\\world_%05d.png", simulation_step);
          imwrite(fname, image);
       }
    }

    // 8.14 show all occupancy grids estimated by robot r1
    int nr_of_occupancy_grids_robot_r1 = (int)r1.get_occupancy_grids().size();        
    if ((VISUALIZATION) || (simulation_step % 100 == 0))
    {
       printf("\n");
       printf("   avg error naive estimated pos : %.5f\n", pos_error_naively_estimated);
       printf("   avg error KF estimated pos : %.5f\n\n", pos_error_KF_estimated);
       for (int i = 0; i < nr_of_occupancy_grids_robot_r1; i++)
       {
          // get a pointer to next occpancy grid
          occupancy_grid* o = r1.get_occupancy_grids()[i];

          // compute error of this occpancy grid
          double avg_error = compute_avg_gridcell_error(o, occupancy_grid_ground_truth);

          // print average error on console
          cout << "   avg error grid '" << o->get_name() << "' --> " << avg_error << "\n";

          // show occpancy grid as image
          imshow(o->get_name(), o->get_grid_as_image());


          if (SAVE_VISUALIZATION_IMAGES)
          {
             char fname[500];
             sprintf_s(fname, "V:\\tmp\\%d\\%05d.png", i, simulation_step);
             imwrite(fname, o->get_grid_as_image());
          }
       }
    }

    // 8.15 one step simulated more
    simulation_step++;

    // 8.16 wait for a key
    char c = (char)waitKey(1);

    // 8.17 ESC pressed? --> exit simulation loop
    if (c == 27)
      exit = true;

  } // while (simulation shall continue)

} // main
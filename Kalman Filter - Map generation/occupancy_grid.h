/// 2D occupancy grid class header file
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org

#pragma once

#include <string>

#include "opencv2/core.hpp" // for Point2d e.g.


using namespace cv;

/// represents an array of m x n cells
/// where we store for each cell a probability p in [0,1]
/// that this cell is occupied by an object, i.e., is no
/// driveable space
class occupancy_grid
{
  public:

                            occupancy_grid(std::string name, int width, int height);

                            ~occupancy_grid();

        void                reset_grid();

        void                update(Point2d robot_pos,
                                   Point2d sensor_ray_dir_vec,
                                   double distance,
                                   bool object_detected);

        Mat&                get_grid_as_image();

        std::string         get_name();

        double**            get_the_grid();

        int                 get_width();
        int                 get_height();

                

  private:

        void                update_cell(int x, int y, double info);

        void                update_visualization_of_cell(int x, int y);

        void                generate_image_from_grid_values();

        double**            the_grid;

        int**               the_grid_counter;
          
        std::string         name;
        int                 height;
        int                 width;

        Mat                 grid_as_an_image;

}; // end class occupancy_grid

/// 2D occupancy grid class implementation file
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "occupancy_grid.h"


occupancy_grid::occupancy_grid(std::string name, int width, int height)
{
  this->name   = name;
  this->height = height;
  this->width  = width;

  // allocate memory for height x width cells
  the_grid = new double*[height];
  for (int y = 0; y < height; y++)
    the_grid[y] = new double[width];

  the_grid_counter = new int*[height];
  for (int y = 0; y < height; y++)
    the_grid_counter[y] = new int[width];

  reset_grid();

  grid_as_an_image = Mat(height, width, CV_8UC3);
  generate_image_from_grid_values();
}


void occupancy_grid::generate_image_from_grid_values()
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int grayness = (int)(the_grid[y][x] * 255.0);
      Vec3b color(grayness, grayness, grayness);
      grid_as_an_image.at<Vec3b>(y, x) = color;
    } // for (x)
  } // for (y)
}


double** occupancy_grid::get_the_grid()
{
  return the_grid;
}


int occupancy_grid::get_width()
{
  return width;
}


int occupancy_grid::get_height()
{
  return height;
}


std::string occupancy_grid::get_name()
{
  return this->name;
}


void occupancy_grid::reset_grid()
{
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      the_grid[y][x] = 0.5;

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      the_grid_counter[y][x] = 0;
}



occupancy_grid::~occupancy_grid()
{
  // free memory allocated for each row
  for (int y = 0; y < this->height; y++)
    delete[] the_grid[y];

  // free memory allocated for the row pointer array
  delete[] the_grid;


  // free memory allocated for each counter row
  for (int y = 0; y < this->height; y++)
    delete[] the_grid_counter[y];

  // free memory allocated for the row pointer array
  delete[] the_grid_counter;
}



void occupancy_grid::update_cell(int x, int y, double info)
{
  const int USE_METHOD = 2;

  if (USE_METHOD == 0)
  {
    if (info == 0.0)
      the_grid[y][x] = 0; // cell is assumed to be free
    else
      the_grid[y][x] = 1; // cell is assumed to be occupied
  }

  if (USE_METHOD == 1)
  {
    if (info == 0.0)
      the_grid[y][x] -= 0.1; // cell is probably free
    else
      the_grid[y][x] += 0.1; // cell is probably occupied
  }

  if (USE_METHOD == 2)
  {
    // moving average, see https://en.wikipedia.org/wiki/Moving_average
    // avg = (new_value + n*avg) / (n+1)
    //

    double n = (double)the_grid_counter[y][x];
    the_grid[y][x] = (info + n * the_grid[y][x]) / (n + 1.0);

    the_grid_counter[y][x]++;
  }



  // make sure, the_grid[y][x] can be always interpreted
  // as a probability, i.e., value has to be in [0,1]
  if (the_grid[y][x] < 0)
    the_grid[y][x] = 0.0;

  if (the_grid[y][x] > 1)
    the_grid[y][x] = 1.0;
}


void occupancy_grid::update_visualization_of_cell(int x, int y)
{
  // get probability that this cell is occupied
  double p = the_grid[y][x];

  // compute color intensity value
  int i = (int) (p*255.0);
  
  Vec3b col;
  if (p <= 0.5)
    col = Vec3b(i, i, i); // gray to black
  else
    col = Vec3b(0, 0, i); // redness
  
  grid_as_an_image.at<Vec3b>(y, x) = col;
}


void occupancy_grid::update(Point2d robot_pos, Point2d sensor_ray_dir_vec, double distance, bool object_detected)
{
  // mark free space in occupancy grid
  // this is all the space starting from the position of the robot
  for (double pos_on_ray = 0.0; pos_on_ray < distance; pos_on_ray += 1.0)
  {
    // compute next position on sensor ray
    double x = robot_pos.x + sensor_ray_dir_vec.x * pos_on_ray;
    double y = robot_pos.y + sensor_ray_dir_vec.y * pos_on_ray;

    // map floating point position to discrete grid cell position
    int ix = (int)round(x);
    int iy = (int)round(y);

    // cell coordinate valid?
    // note: may be invalid, e.g., due to wrong estimated robot position!
    if ((iy < 0) || (ix < 0) || (ix >= width) || (iy >= height))
      continue;

    // that position in the world is assumed to be free
    update_cell(ix, iy, 0.0);

    // update visualization image of occupancy grid
    update_visualization_of_cell(ix, iy);
  }
    
  // did we detect an object at the end of the sensor ray?
  // if yes, we know that that position is not free driveable space
  if (object_detected)
  {
    int ix = (int)round(robot_pos.x + sensor_ray_dir_vec.x * distance);
    int iy = (int)round(robot_pos.y + sensor_ray_dir_vec.y * distance);

    // cell coordinate valid?
    // note: may be invalid, e.g., due to wrong estimated robot position!
    if (!((iy < 0) || (ix < 0) || (ix >= width) || (iy >= height)))
    {
      // that position in the world is assumed to be free
      update_cell(ix, iy, 1.0);

      // update visualization image of occupancy grid
      update_visualization_of_cell(ix, iy);
    }
  }

} // update



Mat& occupancy_grid::get_grid_as_image()
{
  return grid_as_an_image;

} // get_grid_as_image


/// Particle Filter Demo
///
/// In this demo an alien invasion is
/// simulated.
///
/// Unfortunately, the aliens have the
/// ability to split their spaceships
/// into several parts and move these
/// parts independently.
///
/// So we have to be able to represent the
/// location of the spaceship by some
/// *multi-modal representation*!
///
/// For this, a Kalman filter is not appropriate.
///
/// Instead, we try to track their
/// spaceships with the help of a particle filter,
/// which has the ability to represent multi-modal
/// probability distributions. Fortunately!
///
/// ---
/// by Prof. Dr. J�rgen Brauer, www.juergenbrauer.org


#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <conio.h>
#include <random>                 // for random numbers & 1D normal distribution
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI
#include <time.h>

#include "params.h"
#include "spaceship.h"
#include "mvnrnd.h"
#include "particle_filter.h"




using namespace cv;
using namespace std;


vector<Mat> measurements;
float       max_euclidean_distance;

// Macro for mapping a 2D matrix to a 2D point
#define m2p(M) Point((int)M.at<float>(0,0), (int)M.at<float>(1,0))


///
/// here we make use of some "knowledge" about possible movements
/// of the spaceship in order to
/// predict the next particle location in state space
///
class update_by_prediction_model : particle_filter_update_model
{
  void update_particle(particle* p)
  {
    // make a prediction where the object will be next

    // get (x,y,vx,vy) state of this particle
    float x  = p->state[0];
    float y  = p->state[1];
    float vx = p->state[2];
    float vy = p->state[3];

    // prediction step
    x += vx;
    y += vy;

    if (USE_NOISE_IN_PREDICTION_STEP)
    {
       x  += get_rnd_from_interval(-5.0f, 5.0f);
       y  += get_rnd_from_interval(-5.0f, 5.0f);
       vx += get_rnd_from_interval(-0.1f, 0.1f);
       vy += get_rnd_from_interval(-0.1f, 0.1f);
    }

    // add noise to predicted particle position &
    // store new particle position in state space
    p->state[0] = x;
    p->state[1] = y;
    p->state[2] = vx;
    p->state[3] = vy;

  } // update_particle()

}; // end my_prediction_model


///
/// here we make use of the measurement data
/// to compute new particle positions in state space
///
class update_by_measurement_model : particle_filter_update_model
{
  void update_particle(particle* p)
  {
    // get current position of particle
    float x  = p->state[0];
    float y  = p->state[1];
    float vx = p->state[2];
    float vy = p->state[3];
    
    // find out what the nearest measurement to
    // this particle is
    float min_d;
    int nearest_measurement;
    for (unsigned int i = 0; i < measurements.size(); i++)
    {
      float m_x  = measurements[i].at<float>(0,0);
      float m_y  = measurements[i].at<float>(1,0);
      float m_vx = measurements[i].at<float>(2,0);
      float m_vy = measurements[i].at<float>(3,0);
      
      float diff_x  = x  - m_x;
      float diff_y  = y  - m_y;
      float diff_vx = vx - m_vx;
      float diff_vy = vy - m_vy;
      float d = sqrt(diff_x*diff_x + diff_y*diff_y + diff_vx*diff_vx + diff_vy*diff_vy);

      if ((i==0) || (d < min_d))
      {
        min_d = d;
        nearest_measurement = i;
      }
    }

    // move particle "a little bit" into the direction of the
    // nearest measurement
    float fac = MOVE_TO_MEASUREMENT_SPEED;
    x = x   + fac * (measurements[nearest_measurement].at<float>(0, 0) - x);
    y = y   + fac * (measurements[nearest_measurement].at<float>(1, 0) - y);
    vx = vx + fac * (measurements[nearest_measurement].at<float>(2, 0) - vx);
    vy = vy + fac * (measurements[nearest_measurement].at<float>(3, 0) - vy);

    if (RESAMPLING)
    {      
      // compute new weight of particle
      p->weight = max_euclidean_distance - min_d;
    }

    // store new particle position in state space
    p->state[0] = x;
    p->state[1] = y;
    p->state[2] = vx;
    p->state[3] = vy;

  } // update_particle()

}; // end my_measurement_model




// maps a value in [0,1] to a color
// using a heat map coding:
// blue --> white --> yellow --> green --> red
Vec3b map_prob_to_color(float prob)
{
   // 1. define heat map with 6 colors
   //    from cold to hot:
   //    white -> blue -> cyan -> green -> yellow -> red
   const int NUM_Colors = 6;
   const float HeatMapColors[NUM_Colors][3] =
   { { 255,255,255 },{ 0,0,255 },{ 0,255,255 },{ 0,255,0 },{ 255,255,0 },{ 255,0,0 } };

   // 2. make sure, prob is not larger than 1.0
   if (prob > 1.0)
      prob = 1.0;

   // 3. compute bottom and top color in heat map
   prob *= (NUM_Colors - 1);
   int   idx1 = (int)floor(prob);
   int   idx2 = idx1 + 1;
   float fract = prob - (float)idx1; // where are we between the two colors?

   // 4. compute some intermediate color between HeatMapColors[idx2] and HeatMapColors[idx1]
   float R = (HeatMapColors[idx2][0] - HeatMapColors[idx1][0])*fract + HeatMapColors[idx1][0];
   float G = (HeatMapColors[idx2][1] - HeatMapColors[idx1][1])*fract + HeatMapColors[idx1][1];
   float B = (HeatMapColors[idx2][2] - HeatMapColors[idx1][2])*fract + HeatMapColors[idx1][2];

   // 5. set (R,G,B) values in a color / Vec3b object
   Vec3b col;
   col.val[0] = (int)B;
   col.val[1] = (int)G;
   col.val[2] = (int)R;

   // 6. your heat map color is ready!
   return col;

} // map_prob_to_color



Mat get_image_of_continuous_probability_distribution(particle_filter* pf)
{

   clock_t start_time = clock();

   const double sigma = 10.0; // 1,10,20
   const double normalization_fac = 1.0 / (sqrt(2 * M_PI) * sigma);


   // 1. prepare visualization image
   int w = (int)pf->max_values[0];
   int h = (int)pf->max_values[1];
   int N = pf->population_size;


   // 2. prepare a LUT for exp() function values?
   float* LUT_EXP = nullptr;
   if (USE_LUT_FOR_EXP)
   {
       LUT_EXP = new float[2000];
       int len_of_diagonal = (int)sqrt(w*w + h*h);
       int i = 0;
       for (int d = 0; d < len_of_diagonal; d++)
       {
           LUT_EXP[d] = (float)(normalization_fac * exp(-(double)(d*d) / (2.0*sigma*sigma)));
       }       
   }

   printf("generating image of continuous probability distribution:\n");

   
   // 3. prepare 2D array for storing probabilities
   float** probs = new float*[h];
   for (int y = 0; y<h; y++)
      probs[y] = new float[w];

   // 4. compute sum of all particle weights
   float W = 0.0f;
   for (int particle_nr = 0; particle_nr < N; particle_nr++)
   {
      W += pf->all_particles[particle_nr]->weight;
   } // for (particle_nr)

   // 4. compute probability for each (x,y) position
   printf("computing probability for each (x,y) position");   
   float max_prob = 0.0;
   for (int y = 0; y < h; y+= SPEED_UP_KDE_STEPSIZE)
   {
      printf(".");
      for (int x = 0; x < w; x+= SPEED_UP_KDE_STEPSIZE)
      {
         // 4.1 compute probability P(x,y)
         float prob = 0.0f;
         for (int particle_nr = 0; particle_nr < N; particle_nr++)
         {
            // get the next particle
            particle* p = pf->all_particles[particle_nr];

            // compute Eculidean distance between particle pos & (x,y) pos
            float dx = p->state[0] - (float)x;
            float dy = p->state[1] - (float)y;
            float d = sqrt(dx*dx + dy*dy);

            // get Kernel value K(d)   
            float kernel_value;
            if (!USE_LUT_FOR_EXP)
                kernel_value = (float)(normalization_fac * exp(-d*d / (2.0*sigma*sigma)));
            else
            {
                kernel_value = LUT_EXP[(int)d];
            }

            // update prob
            prob += p->weight * kernel_value;

         } // for (particle_nr)

         // 4.2 later we need to know the max probability found
         //     in order to normalize the values for drawing a heat map
         if (prob>max_prob)
            max_prob = prob;

         // 4.3 store computed probability for that (x,y) position
         probs[y][x] = prob;

      } // for (x)
   } // for (y)
   printf("\n");
   //printf("max_prob = %.5f\n", max_prob);

   // 5. normalize all values to be in [0,1]
   for (int y = 0; y < h; y+= SPEED_UP_KDE_STEPSIZE)
   {
      for (int x = 0; x < w; x+= SPEED_UP_KDE_STEPSIZE)
      {
         probs[y][x] /= max_prob;

      } // for (x)
   } // for (y)

   // 6. generate image from 2D probability array
   Mat image_prob_distri(h / SPEED_UP_KDE_STEPSIZE,
                         w / SPEED_UP_KDE_STEPSIZE,
                         CV_8UC3);
   image_prob_distri = 0;
   for (int y = 0; y < h/SPEED_UP_KDE_STEPSIZE; y++)
   {
      for (int x = 0; x < w/SPEED_UP_KDE_STEPSIZE; x++)
      {
         // 6.1 map that probability to a color
         Vec3b col = map_prob_to_color(probs[y*SPEED_UP_KDE_STEPSIZE][x*SPEED_UP_KDE_STEPSIZE]);

         // 6.2 set color in image
         image_prob_distri.at<Vec3b>(y, x) = col;

      } // for (x)
   } // for (y)

   double elpased_time= (clock()-start_time) / CLOCKS_PER_SEC;
   printf("Time elpased for computing continuous density: %.2f sec", elpased_time);

   // 7. free memory
   delete LUT_EXP;

   // 8. image is ready! return it to caller
   return image_prob_distri;

} // get_image_of_continuous_probability_distribution


int main()
{
  srand(28011979);

  // 1. load background image
  //    image license is CC0 (public domain).
  //    Download location:
  //    https://pixabay.com/de/gewitter-wolken-horizont-brazilien-548951/
  string background_img_filename =
    "pics\\earth_orbit_seen_from_space_pixabay_lowres.jpg";
  Mat background_image = imread(background_img_filename);
  

  // 2. image load error?
  if ((background_image.cols == 0) || (background_image.rows == 0))
  {
    cout << "Error! Could not read the image file: " << background_img_filename << endl;
    _getch();
    return -1;
  }
  max_euclidean_distance = (float)
    sqrt(background_image.cols * background_image.cols + background_image.rows * background_image.rows);


  // 3. make background image gray, but let it be a 3 color channel RGB image
  //    so that we can draw colored information on it
  Mat background_image_gray;
  cvtColor(background_image, background_image_gray, CV_RGB2GRAY);
  cvtColor(background_image_gray, background_image, CV_GRAY2RGB);


  // 4. create an alien spaceship with NR_SPACESHIP_PARTS parts
  //    and tell the spaceship how big our 2D world is
  spaceship alien_spaceship(NR_SPACESHIP_PARTS, background_image.size());



  // 5. define the "noisy-ness" of our measurements &
  //    prepare random generator according to the measurement
  //    noise covariance matrix
  cv::Mat R = (Mat_<float>(4, 4) <<  200.0,    0.0, 0.0, 0.0,
                                       0.0,  200.0, 0.0, 0.0,
                                       0.0,    0.0, 0.1, 0.0,
                                       0.0,    0.0, 0.0, 0.1);
  cv::Mat mean_vec = (Mat_<float>(4, 1) << 0.0, 0.0);
  mvnrnd* rnd_generator_measurement_noise = new mvnrnd(mean_vec, R);


  // 6. prepare particle filter object  
  particle_filter* my_pf = NULL;

  update_by_prediction_model*   my_update_by_prediction_model  = 
     new update_by_prediction_model();

  update_by_measurement_model*  my_update_by_measurement_model = 
     new update_by_measurement_model();
  

  // 7. the simulation loop:
  int simulation_step = 0;
  while (true)
  {
    // 7.1 clear screen & visualization image
    //system("cls");
    //printf("Simulation step : %d\n", simulation_step);


    // 7.2 copy background image into new visualization image
    //Mat image = background_image.clone();
    Mat image;
    background_image.copyTo(image);


    // 7.3 move the alien spaceship
    alien_spaceship.move();


    // 7.4 draw alien spaceship into image
    alien_spaceship.draw_yourself_into_this_image( image );


    // 7.5 simulate noisy measurements

    // 7.5.1 clear measurement vector
    measurements.clear();


    // 7.5.2 add a noisy measurement near to each part
    vector<spaceship::part_info*> part_infos = alien_spaceship.get_part_info_vector();    
    for (unsigned int part_nr = 0; part_nr < part_infos.size(); part_nr++)
    {
      // number of measurements per part is randomly
      // note: the particle filter can live with a varying length
      //       of measurements! That's another advantage!
      unsigned int rnd_nr_of_measurements_per_part = rand() % 3 + 1;

      for (unsigned int i = 0; i < NR_MEASUREMENTS_PER_SPACESHIP_PART; i++)
      {
        // prepare noise vector
        Mat noise_4D_vec = rnd_generator_measurement_noise->get_next_random_vector();
        
        // generate new measurement near to ground truth location & movement vector of that part
        Mat measurement_4D_vec = (Mat_<float>(4, 1) <<
          part_infos[part_nr]->location.x + noise_4D_vec.at<float>(0, 0),
          part_infos[part_nr]->location.y + noise_4D_vec.at<float>(1, 0),
          part_infos[part_nr]->move_vec.x + noise_4D_vec.at<float>(2, 0),
          part_infos[part_nr]->move_vec.y + noise_4D_vec.at<float>(3, 0));
          

        // add new measurement to list of measurements
        measurements.push_back( measurement_4D_vec );

      } // for (i)
    } // for (part_nr)


    // 7.5.3 add some wrong measurements from time to time
    if (SIMULATE_COMPLETELY_WRONG_MEASUREMENTS)
    {
      bool always_generate_wrong_measurements = true;
      if ((always_generate_wrong_measurements) || (rand() % 1 == 0)) // from time to time ...
      {
        
        for (int nr=0; nr<NR_OF_COMPLETELY_WRONG_MEASUREMENTS; nr++)
        {
          int x = rand() % background_image.cols;
          int y = rand() % background_image.rows;
          float vx = get_rnd_from_interval(-1, 1);
          float vy = get_rnd_from_interval(-1, 1);
          Mat measurement_4D_vec = (Mat_<float>(4, 1) << x, y, vx, vy);
          measurements.push_back(measurement_4D_vec);
        }
      }
    } // if (add some wrong measurement)


    // 7.5.4 test relocation speed of particles?
    // for testing the relocation of particles
    // using the RESAMPLING method
    // after measurements do not show up
    // any longer in some area of the state space
    
    if (TEST_RELOCATION_OF_PARTICLES && (simulation_step<150))
    {
       for (int i = 0; i < 20; i++)
       {
          int x = rand() % (background_image.cols / 4);
          int y = rand() % (background_image.rows / 4);
          float vx = get_rnd_from_interval(-1, 1);
          float vy = get_rnd_from_interval(-1, 1);
          Mat measurement_4D_vec = (Mat_<float>(4, 1) << x, y, vx, vy);
          measurements.push_back(measurement_4D_vec);
       }
    }


    // 8. visualize all measurements by yellow circles
    for (unsigned int i = 0; i < measurements.size(); i++)
    {
      Point p((int) measurements[i].at<float>(0,0), (int) measurements[i].at<float>(1,0));
      circle(image, p, 5, CV_RGB(255,255,0), -1);
    }


    // 9. wait for user input
    int c;
    if (STEP_BY_STEP)
      c = waitKey(0);
    else
      c = waitKey(1);
    if (c == 29) // ESC pressed?
      break;
        

    // 10. do one particle filter update step
    if (my_pf != NULL)
      my_pf->update();


    // 11. if the user wants to track the spaceship,
    //     we initialize a particle filter
    if (c == 't')
    {
      std::cout << "Starting tracking spaceship using particle filter..." << std::endl;

      // 11.1 if there is already a particle filter object,
      //      delete it
      if (my_pf != NULL)
        delete my_pf;

      // 11.2 generate new particle filter object
      my_pf = new particle_filter(POPULATION_SIZE, 4);

      // 11.3 initialize particle filter:

      // 11.4 set state space dimensions
      my_pf->set_param_ranges(0, 0.0f, (float)image.cols);
      my_pf->set_param_ranges(1, 0.0f, (float)image.rows);
      my_pf->set_param_ranges(2, -1.0f, 1.0f);
      my_pf->set_param_ranges(3, -1.0f, 1.0f);

      // 11.5 set motion & measurement model			
      my_pf->set_prediction_model((particle_filter_update_model*)my_update_by_prediction_model);
      my_pf->set_perception_model((particle_filter_update_model*)my_update_by_measurement_model);

      // 11.6 start with random positions in state space and uniform weight distribution
      my_pf->reset_positions_and_weights();
    }


    // 12. visualize all particle locations
    if (my_pf != NULL)
    {
      for (int i = 0; i < my_pf->population_size; i++)
      {
        // get the i-th particle
        particle* p = my_pf->all_particles[i];

        // get (x,y) position of particle
        int x = (int)p->state[0];
        int y = (int)p->state[1];

        // visualize particle position by a red circle
        circle(image, Point(x, y), 3, CV_RGB(255, 0, 0), 1);

      } // for (all particles)
    } // if (we currently track using the particle filter)

    
    // 13. does the user want to generate a continous probability image based
    //     on the discrete particle positions?
    if ((c == 'p') && (my_pf != NULL))
    {
       imshow("Continuous probability based on discrete particle positions",
              get_image_of_continuous_probability_distribution(my_pf));
    }

    // 13. show visualization image
    char txt[500];
    sprintf_s(txt, "%04d", simulation_step);
    putText(image, txt, Point(image.cols-45, image.rows-10),
       FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,255,0), 1);
    imshow("Tracking an alien spaceship with a particle filter!", image);

    // 14. time goes by...
    simulation_step++;

  } // while


  printf("Particle filter demo finished! Press a key to exit.\n");
  _getch();

} // main
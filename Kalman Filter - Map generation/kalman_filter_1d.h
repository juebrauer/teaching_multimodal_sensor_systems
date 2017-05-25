/// n-dimensional (linear) Kalman filter header file
///
/// for a good explanation of (a special case) of the
/// 1D Kalman filter case, see Sebastian Thrun's
/// videos:
///
/// https://www.youtube.com/watch?v=X7YggdDnLaw 
/// (prediction step)
/// and
/// https://www.youtube.com/watch?v=d8UrbKKlGxI 
/// (correction-by-measurement step)
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#pragma once

class kalman_filter_1d
{
  public:

               kalman_filter_1d(double init_µ,
                                double init_sigma,
                                double process_noise,
                                double measurement_noise);

      void     predict(double u);

      void     correct_by_measurement(double z);

      double   get_current_state_estimate();

      double   get_current_uncertainty();


  private:

      double   µ;                 // current estimated state with highest probability
      double   sigma;             // uncertainty about the estimated state
                                  
      double   process_noise;     // how much (Gaussian) randomness is there during a state transition?
      double   measurement_noise; // how much (Gaussian) randomness is there during a measurement?

};

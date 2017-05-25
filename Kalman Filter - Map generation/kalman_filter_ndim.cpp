/// n-dimensional (linear) Kalman filter implementation file
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "kalman_filter_ndim.h"




kalman_filter_ndim::kalman_filter_ndim(
  cv::Mat init_µ,           // initial state vector:
                            //   what we know about the (probable) start state
  cv::Mat init_P,           // initial (uncertainty) covariance matrix:
                            //   how sure are we about the start state - in each dimension?
  cv::Mat F,                // state transition matrix:
                            //   for predicting the next state from the current state
  cv::Mat B,                // control matrix:
                            //   for predicting the next state from the current state using control signals
  cv::Mat Q,                // process noise covariance matrix:
                            //   describes the (Gaussian) randomness of state transitions in each dimension
  cv::Mat H,                // measurement matrix:
                            //   describes how we think that our sensors map states to measurements z
  cv::Mat R                 // measurement noise covariance matrix:
                            //   describes the (Gaussian) randomness of measurements per dimension
  )
{
  this->µ = init_µ.clone();
  this->P = init_P.clone();
  this->F = F.clone();
  this->B = B.clone();
  this->Q = Q.clone();
  this->H = H.clone();
  this->R = R.clone();

} // constructor kalman_filter_ndim




// predicts the new state vector µ using the transition matrix F
// and the specified control vector u
// and updates the uncertainty covariance matrix
// F embodies our knowledge about the system dynamics
void kalman_filter_ndim::predict(cv::Mat u)
{
  // predict new state
  µ = F*µ + B*u;

  // update uncertainty covariance matrix
  P = F*P*F.t() + Q;

} // predict
                                                      
   

                                                      
// corrects the state vector µ and uncertaininty covariance matrix P
// predicted by predict() using the new measurement vector z
void kalman_filter_ndim::correct_by_measurement(cv::Mat z)
{
  // compute innovation y
  cv::Mat y = z - H*µ;

  // compute residual covariance matrix S
  cv::Mat S = H*P*H.t() + R;

  // compute Kalman gain matrix
  // the Kalman gain matrix tells us how strongly to correct
  // each dimension of the predicted state vector by the help
  // of the measurement
  cv::Mat K = P * H.t() * S.inv();
 

  // correct previously predicted new state vector
  µ = µ + K*y;

  // update uncertainty covariance matrix
  P = P - K*H*P;  // (I - K*H)*P

} // correct_by_measurement
  



// returns the current estimated state vector µ,
// may it be after the predict() or after the
// correct_by_measurement() step
cv::Mat kalman_filter_ndim::get_current_state_estimate()
{
  return µ;

} // get_current_state_estimate
                                                      
   


// returns the current estimated uncertainty covariance matrix P
// may it be after the predict() or after the
// correct_by_measurement() step
// the covariance matrix describes the variance of each
// state vector argument = uncertainty about this argument
// of the state vector
cv::Mat kalman_filter_ndim::get_current_uncertainty()
{
  return P;

} // get_current_uncertainty
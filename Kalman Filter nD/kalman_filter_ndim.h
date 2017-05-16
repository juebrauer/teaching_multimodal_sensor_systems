/// file: kalman_filter_ndim.h
///
/// by Prof. Dr. Jürgen Brauer

#pragma once

#include "opencv2/core.hpp" // get definition of cv::Mat

/// kalman_filter_ndim:
/// 
/// An implementation of the n-dimensional Kalman filter
///
/// Kalman filters better estimate states then just predicting
/// new states or just believing measurements
///
/// Their power comes from the fact that we combine
/// our system knowledge about state transitions with
/// measurements.
///
/// Think of them as a weighted average of predicted and
/// measured states, where the weighting is changed
/// steadily.
///
class kalman_filter_ndim
{
  public:

               kalman_filter_ndim(cv::Mat init_µ,           // initial state vector:
                                                            //   what we know about the (probable) start state
                                  cv::Mat init_P,           // initial (uncertainty) covariance matrix:
                                                            //   how sure are we about the start state - in each dimension?
                                  cv::Mat F,                // state transition matrix:
                                                            //   for predicting the next state from the current state using system dynamics
                                  cv::Mat B,                // control matrix:
                                                            //   for predicting the next state modification using control signals
                                  cv::Mat Q,                // process noise covariance matrix:
                                                            //   describes the (Gaussian) randomness of state transitions in each dimension
                                  cv::Mat H,                // measurement matrix:
                                                            //   describes how we think that our sensors map states to measurements z
                                  cv::Mat R                 // measurement noise covariance matrix:
                                                            //   describes the (Gaussian) randomness of measurements per dimension
                                );

      void     predict(cv::Mat u);                          // predicts the new state vector µ using the transition matrix F
                                                            // and the specified control vector u
                                                            // and updates the uncertainty covariance matrix
                                                            // F embodies our knowledge about the system dynamics

      void     correct_by_measurement(cv::Mat z);           // corrects the state vector µ and uncertaininty covariance matrix P
                                                            // (as predicted by predict()) using the new measurement vector z

      cv::Mat  get_current_state_estimate();                // returns the current estimated state vector µ,
                                                            // may it be after the predict() or after the
                                                            // correct_by_measurement() step - whenever you call it

      cv::Mat  get_current_uncertainty();                   // returns the current estimated uncertainty covariance matrix P
                                                            // may it be after the predict() or after the
                                                            // correct_by_measurement() step - whenever you call it
                                                            // the covariance matrix describes the variance of each
                                                            // state vector argument = uncertainty about this argument
                                                            // of the state vector

  private:

      cv::Mat   µ;                        // current n-dimensional state vector
      cv::Mat   P;                        // uncertainty covariance matrix

      cv::Mat   F;                        // state transition matrix
      cv::Mat   B;                        // control matrix
      cv::Mat   Q;                        // Gaussian state transition noise = process noise

      cv::Mat   H;                        // measurement matrix
      cv::Mat   R;                        // Gaussian measurement noise = how (Gaussian) noisy are our sensors
      
}; // class kalman_filter_ndim

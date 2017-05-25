/// file: mvnrnd.cpp
///
/// Implementation for a class
/// that represents a random number generator
/// that can generate n-dimensional
/// random vectors y that follow
/// a specified multivariate normal distribution, i.e.,
/// y ~ N(µ,S)
///    where µ is a n-dimensional mean vector
///          S is a n x n covariance matrix
///
/// by Prof. Dr. Jürgen Brauer

#include <iostream>

#include "mvnrnd.h"

#include "math_tools.h" // for computing the Cholesky decomposition C of S, i.e., such that CC^T = S

mvnrnd::mvnrnd(Mat µ, Mat S)
{
  this->µ = µ.clone();
  this->S = S.clone();

  this->C = cholesky_decomposition(S).clone();

  if (false)
  {
    printf("Cholesky decomposition of \n");
    std::cout << S << std::endl;
    printf("is\n");
    std::cout << C << std::endl;
  }
}



/// returns an n-dimensional random vector
/// distributed according to a multivariate normal distribution
/// with mean µ and covariance matrix S
///
/// See
/// https://www.quora.com/How-can-I-generate-two-dependent-random-variables-follow-the-standard-normal-distribution-which-has-the-correlation-value-0-5-in-C++-11
Mat mvnrnd::get_next_random_vector()
{
  int n = µ.rows;

  // 1. generate a random vector Z of uncorrelated variables
  Mat Z = Mat(n, 1, CV_64F);
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (int i = 1; i <= n; i++)
  {
    double rnd_number = distribution(generator);
    Z.at<double>(i - 1, 0) = rnd_number;
  }

  // 2. now map that vector of uncorrelated variables to one
  //    of correlated variables
  Mat mult_result = C*Z;
  Mat Y = µ + mult_result;

  // 3. return that random vector
  return Y;

} // get_next_random_vector

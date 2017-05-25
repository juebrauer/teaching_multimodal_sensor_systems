#pragma once

#include "opencv2/core.hpp"  // for cv::Mat


/// Cholesky decomposition of a matrix A
///
/// Each symmetric (!) positive definitive (!, i.e. x^TAx>0) matrix A can be decomposed
/// to A = LL^T, where A is a lower triangular matrix
///
/// formulas for Cholesky decomposition can be found, e.g., at:
/// https://de.wikipedia.org/wiki/Cholesky-Zerlegung#Berechnung
///
/// my final code is similar to:
/// http://blog.csdn.net/naruto0001/article/details/9151159
///

// macro for mapping
// normal matrix indices 1,2,3,...
// to our computer scientist 0,1,2,...
// indices ;-)
#define m(x) x-1

cv::Mat cholesky_decomposition(const cv::Mat&A)
{
  // 1. generate a matrix from size of A
  //    initialized with zeros
  //    where we want to store double values
  cv::Mat G = cv::Mat::zeros(A.size(), CV_64F);

  // 2. how many rows are there?
  int n = A.rows;

  // 3. for all rows of matrix G to compute ...
  for (int i = 1; i <= n; i++)
  {
    // compute elements G_ij before diagonal element G_ii
    for (int j = 1; j < i; j++)
    {
      double sum = 0;
      for (int k = 1; k <= j - 1; k++)
      {
        sum += G.at<double>(m(i), m(k)) * G.at<double>(m(j), m(k));
      }
      G.at<double>(m(i), m(j)) = (A.at<double>(m(i), m(j)) - sum) / G.at<double>(m(j), m(j));

    } // for (all column elements j)

      // compute element G_jj = G_ii
    double sum = 0;
    for (int k = 1; k <= i - 1; k++)
    {
      sum += G.at<double>(m(i), m(k)) * G.at<double>(m(i), m(k));
    }
    G.at<double>(m(i), m(i)) = sqrt(A.at<double>(m(i), m(i)) - sum);

  } // for (all rows i)

    // 4. return cholesky decomposition result, i.e.,
    //    matrix G, such that G*G^T = A
  return G;

} // cholesky_decomposition
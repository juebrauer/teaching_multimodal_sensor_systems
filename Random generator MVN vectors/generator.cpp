/// file: generator.cpp
///
/// A random vector generator for random vectors
/// that follow a multi-variate normal distribution.
/// I.e., individual variables of the vector can be
/// correlated and correlation is described by a
/// covariance matrix.
///
/// Note: you can plot the data with the help of gnuplot:
///       set grid
///       gnuplot "random_vecs.txt"
///
///
/// ---
/// by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "opencv2/opencv.hpp"

#include <conio.h>        // for _getch()
#include <iostream>       // for cout
#include <random>         // for 1D Gaussian random number generator std::normal_distribution<double> 
#include <fstream>        // for writing random vectors to a file

using namespace cv;

std::default_random_engine generator;


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
// normal matrix indices to our computer scientist indices ;-)
#define m(x) x-1

Mat cholesky_decomposition(const Mat&A)
{
  // 1. generate a matrix from size of A
  //    initialized with zeros
  //    where we want to store float values
  Mat G = Mat::zeros(A.size(), CV_32F);

  // 2. how many rows are there?
  int n = A.rows;

  // 3. for all rows of matrix G to compute ...
  for (int i = 1; i <= n; i++)
  {
    // compute elements G_ij before diagonal element G_ii
    for (int j = 1; j < i; j++)
    {
      float sum = 0;
      for (int k = 1; k <= j-1; k++)
      {
        sum += G.at<float>( m(i), m(k) ) * G.at<float>( m(j), m(k) );
      }
      G.at<float>(m(i),m(j)) = (A.at<float>(m(i), m(j)) - sum) / G.at<float>(m(j), m(j));

    } // for (all column elements j)
    
    // compute element G_jj = G_ii
    float sum = 0;
    for (int k = 1; k <= i-1; k++)
    {
      sum += G.at<float>(m(i), m(k)) * G.at<float>(m(i), m(k));
    }
    G.at<float>(m(i), m(i)) = sqrt(A.at<float>(m(i), m(i)) - sum);

  } // for (all rows i)

  // 4. return cholesky decomposition result, i.e.,
  //    matrix G, such that G*G^T = A
  return G;

} // cholesky_decomposition


/// returns an n-dimensional random vector
/// distributed according to a multivariate normal distribution
/// with mean µ and covariance matrix S
///
/// See
/// https://www.quora.com/How-can-I-generate-two-dependent-random-variables-follow-the-standard-normal-distribution-which-has-the-correlation-value-0-5-in-C++-11

Mat get_ndim_random_vector_from_multivariate_normal_distribution(Mat µ, Mat S)
{
  int n = µ.rows;

  // 1. generate a random vector Z of uncorrelated variables
  Mat Z = Mat(n,1, CV_32F);
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i = 1; i<=n; i++)
  {
    float rnd_number = distribution(generator);
    Z.at<float>(i - 1, 0) = rnd_number;
  }

  // 2. get Cholesky decomposition of S
  Mat G = cholesky_decomposition(S);

  // 3. now map that vector of uncorrelated variables to one
  //    of correlated variables
  Mat Y = µ + G*Z;

  // 4. return that random vector
  return Y;

} // get_ndim_random_vector_from_multivariate_normal_distribution



int main()
{
   srand((unsigned int)time(NULL));

  // 1. define and output a test matrix A
  float f[4][4] = {
    { 11.0,  4.0,  4.0,  4.0 },
    {  4.0,  6.0,  4.0,  2.0 },
    {  4.0,  4.0,  6.0,  2.0 },
    {  4.0,  2.0,  2.0,  4.0 },
  };
  Mat A;
  A = Mat(4, 4, CV_32F, f);
  std::cout << "A = " << A;

  // 2. compute Cholesky decomposition of A,
  //    i.e., a matrix L, such that L*L^T = A
  Mat G = cholesky_decomposition(A);

  // 3. output matrix G
  std::cout << "\n\nG = " << G;

  // 4. output matrix G*G^T --> should be A if Cholesky decomposition works!
  Mat G_transpose;
  transpose(G, G_transpose);
  std::cout << "\n\nGG^T = " << G*G_transpose;

  

  // 5. Now let us compute random numbers
  //    generated from a multivariate normal distribution
  //    For doing so, see the hint at
  //    https://www.quora.com/How-can-I-generate-two-dependent-random-variables-follow-the-standard-normal-distribution-which-has-the-correlation-value-0-5-in-C++-11
  Mat µ = (Mat_<float>(2, 1) << 5.0, 2.0);
  Mat S = (Mat_<float>(2, 2) << 10.0, 2.1,
                                 2.1, 1.0);
  Mat G2 = cholesky_decomposition(S);
  Mat G2_transpose;
  transpose(G2, G2_transpose);
  std::cout << "\n\nS = " << S;
  std::cout << "\n\nG2G2^T = " << G2*G2_transpose; 
  
  
  std::ofstream my_file("V:\\tmp\\random_vecs.txt");
  for (int vec_nr = 0; vec_nr < 1000; vec_nr++)
  {
    Mat x = get_ndim_random_vector_from_multivariate_normal_distribution(µ, S);
    for (int idx = 0; idx < x.rows; idx++)
    {
      my_file << x.at<float>(idx, 0) << " ";
    }
    my_file << std::endl;
  } // for (all random vectors to generate and write to file)
  

  /*
  const int H = 800;
  const int W = 800;
  Mat plot(H,W,CV_8UC3);
  plot = 0;
  int scale = 10;
  for (int vec_nr = 0; vec_nr < 1000; vec_nr++)
  {
     Mat x = get_ndim_random_vector_from_multivariate_normal_distribution(µ, S);
     int xcoord = (int) x.at<float>(0, 0);
     int ycoord = (int) x.at<float>(0, 1);     
     circle(plot, Point(xcoord*scale+W/2, ycoord*scale+H/2), 1, CV_RGB(0, 255, 0), 1);
  }
  imshow("plot", plot);
  waitKey(0);
  */


  // 6. wait for user to press a key
  _getch();

} // main
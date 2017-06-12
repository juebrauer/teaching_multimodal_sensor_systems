#pragma once

#define IMG_WIDTH  800
#define IMG_HEIGHT 800


// if set to true, naive estimated state will be shown
#define SHOW_NAIVE_ESTIMATION true

// if set to true, EKF estimated state will be shown
#define SHOW_EKF_ESTIMATION true

// if set to true, EKF debug information will be displayed
// on the console
#define SHOW_DEBUG_INFO false


#define COL_GT             CV_RGB(255, 255, 255)
#define COL_NAIVE          CV_RGB(  0, 255,   0)
#define COL_MEASUREMENT    CV_RGB(  0, 255, 255)
#define COL_EKF            CV_RGB(255,   0,   0)


// how many last points of the trajectory shall be
// saved and displayed?
#define MAX_NR_TRAJECTORY_POINTS 200
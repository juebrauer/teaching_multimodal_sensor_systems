/// State of the art tracker test
///
/// The contribu library contains several state of the art
/// tracker implementations.
/// 
/// This program allows to experiment with them.
///
/// Note! There are temporay problems with OpenCV master version
/// and ffmpeg, see https://github.com/opencv/opencv/issues/8097
///
/// For this, we read in the images and not use VideoCapture.
///
/// ---
/// Prof. Dr. Jürgen Brauer, www.juergenbrauer.org

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking.hpp"

#include <conio.h>

using namespace cv;
using namespace std;


void smaller(Mat& img)
{
   resize(img, img, Size(img.cols / 2, img.rows / 2));
}

#define NR_OF_TRACKERS 4
char tracker_name[NR_OF_TRACKERS][100] =
      { "MEDIANFLOW", "MIL", "BOOSTING", "KCF" };


int main(int argc, char **argv)
{  
   // 1. let user choose a region to track
   Mat frame =
      imread("V:\\01_job\\12_datasets\\traffic_scenes\\yosemite\\0001.png");
   smaller(frame);
   Rect2d bbox_start = selectROI(frame, false);

   int frame_save_counter = 0;
   for (int tracker_nr = 0; tracker_nr < NR_OF_TRACKERS; tracker_nr++)
   {
      // 2. Set up tracker
      //    Instead of MIL, you can also use 
      //    BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN  
      Ptr<Tracker> tracker =
         Tracker::create(tracker_name[tracker_nr]);

      // 3. initialize tracker with first frame and bounding box   
      Rect2d bbox = bbox_start;
      tracker->init(frame, bbox);

      // 4. now track region from frame to frame
      for (int frame_nr = 2; frame_nr < 800; frame_nr++)
      {
         // 4.1 read in next video frame
         char fname[500];
         sprintf_s(fname,
            "V:\\01_job\\12_datasets\\traffic_scenes\\yosemite\\%04d.png", frame_nr);
         frame = imread(fname);

         // 4.2 is it valid?
         if (frame.empty())
            break;

         // 4.3 update tracking results
         smaller(frame);
         tracker->update(frame, bbox);

         // 4.4 draw bounding box
         rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

         // 4.5 draw information text
         char txt[100];
         sprintf_s(txt, "Testing tracker %s - Frame: %04d",
            tracker_name[tracker_nr], frame_nr);
         putText(frame,
            txt,
            Point(10,15),
            FONT_HERSHEY_SIMPLEX, 0.5, // font face and scale
            CV_RGB(0, 0, 255),
            1); // line thickness and type


         // 4.6 display result
         imshow("Tracking result", frame);
         int k = waitKey(1);
         if (k == 27) break;

         // 4.7 save result image
         if (1)
         {
            char fname[500];
            sprintf_s(fname, "V:\\tmp\\%04d.png", frame_save_counter++);
            imwrite(fname, frame);
         }

         // 4.8 compute next frame number
         frame_nr++;
      }

   } // for all tracker types
   
   printf("Tracking finished.");
   _getch();
   return 0;
}
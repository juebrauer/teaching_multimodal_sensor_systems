/// Example of tracking image parts
///
/// Tracking is an important method to bridge
/// situations in which an object detector fails.
///
/// ---
/// Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "opencv2/opencv.hpp"
#include <conio.h>

using namespace cv;
using namespace std;

int select_state;
Point mouse_pos;
Point select_tl;
Point select_br;
Rect  selection;

#define VIDEO_FILENAME "V:\\01_job\\12_datasets\\tracking_tests\\sports_ground_part1.mp4";




#define MODEL_LEN 3
struct model
{
   double data[MODEL_LEN];
};


void mouse_callback_func(int event, int x, int y, int flags, void* userdata)
{
   mouse_pos = Point2i(x, y);

   if ((select_state == 0) && (event == EVENT_LBUTTONDOWN)) {
      select_tl = mouse_pos;
      cout << "Left mouse button clicked at " << select_tl << endl;
      select_state = 1;
      return;
   }

   if ((select_state == 1) && (event == EVENT_LBUTTONDOWN)) {
      select_br = mouse_pos;
      cout << "Right mouse button clicked at " << select_br << endl;
      select_state = 2;
      return;
   }
}

void smaller(Mat& img)
{
   resize(img, img, Size(img.cols / 3, img.rows / 3));
}


void compute_model(Mat img, Rect r, model* m, bool print_info = false)
{  
}


double compute_model_distance(model* m1, model* m2)
{
   double sum_dist = 0.0;
   for (int i = 0; i < MODEL_LEN; i++)
   {
      sum_dist += abs(m1->data[i] - m2->data[i]);
   }
   return sum_dist;
}


Rect track_and_update_model(model* search_model, Rect r, Mat frame)
{  
   return r;

} // track_and_update_model


int main()
{
   // 1. define a video filename
   string videofilename = VIDEO_FILENAME;



   // 2. try to open the video file
   VideoCapture cap(videofilename);
   if (!cap.isOpened()) {
      printf("Error! Could not open video file %s\n", videofilename.c_str());
      _getch();
      return -1;
   }


   // 3. get a first image from the video
   Mat first_frame;
   cap >> first_frame;
   smaller(first_frame);


   // 4. Register a mouse callback function   
   namedWindow("Region select", 1);
   setMouseCallback("Region select", mouse_callback_func, nullptr);


   // 5. let the user select some area
   select_state = 0;
   while (select_state != 2)
   {
      Mat img = first_frame.clone();

      if (select_state == 1)
         rectangle(img, Rect(select_tl, mouse_pos), CV_RGB(0, 255, 0), 2);

      if (select_state == 2)
         rectangle(img, Rect(select_tl, select_br), CV_RGB(0, 255, 0), 2);

      imshow("Region select", img);
      waitKey(50);
   }
   selection = Rect(select_tl, select_br);


   // 6. build up a representation model
   //    for the selected image region
   model* my_model = new model;
   compute_model(first_frame, selection, my_model, true);


   // 7. read in frames from video and process them   
   int frame_counter = 0;
   Rect tracking_rect = selection;
   while (true)
   {
      // 6.1 show video frame nr
      //printf("frame: %d\n", frame_counter);

      // 6.2 get next video frame
      Mat frame;
      cap >> frame;

      // 6.3 is it a valid video frame?
      if (frame.empty())
         break; // no! so quit this loop!

                // 6.4 resize frame
      smaller(frame);

      // 6.5 track the region!
      tracking_rect = track_and_update_model(my_model, tracking_rect, frame);

      // 6.6 prepare a visualization image
      //     where we show the tracked region
      Mat visu = frame.clone();
      rectangle(visu, tracking_rect, CV_RGB(0, 255, 0), 2);
      char txt[100];
      sprintf_s(txt, "%04d: d0:%.2f, d1:%.2f, d2:%.2f",
         frame_counter, my_model->data[0], my_model->data[1], my_model->data[2]);
      putText(visu,
         txt,
         Point(tracking_rect.tl().x, tracking_rect.tl().y - 15),
         FONT_HERSHEY_SIMPLEX, 0.5, // font face and scale
         CV_RGB(255, 255, 0),
         1); // line thickness and type

             // 6.5 show frame
      imshow("Tracking result", visu);
      waitKey(1);

      // 6.6 next frame
      frame_counter++;
   }

   printf("Tracking finished. Press a key to exit!\n");
   _getch();
}
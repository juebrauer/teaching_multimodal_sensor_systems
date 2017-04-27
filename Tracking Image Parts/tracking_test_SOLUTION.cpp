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

#define VIDEO_FILENAME "V:\\01_job\\12_datasets\\tracking_tests\\sports_ground_part2.mp4";

// tracking algorithm parameters:
#define UPDATE_MODEL false
#define SEARCH_DIAMETER 40
#define SCALE_MIN 0.7
#define SCALE_MAX 1.3
#define SCALE_STEP 0.02
#define MIN_PROPOSAL_REGION_SIZE 20
#define DESCRIPTOR_METHOD 2


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
   // 1. sum up R,G,B values
   int R, G, B;
   R = G = B = 0;
   int nr_pixels = 0;
   for (int y = r.tl().y; y < r.br().y; y++)
   {
      for (int x = r.tl().x; x < r.br().x; x++)
      {
         Vec3b v = img.at<Vec3b>(y, x);
         B += v[0];
         G += v[1];
         R += v[2];
         nr_pixels++;
      }
   }
   double NR = (double)R / (double)nr_pixels;
   double NG = (double)G / (double)nr_pixels;
   double NB = (double)B / (double)nr_pixels;

   // 2. compute & save model data
   if (DESCRIPTOR_METHOD == 1)
   {
      m->data[0] = NR;
      m->data[1] = NG;
      m->data[2] = NB;
   }
   if (DESCRIPTOR_METHOD == 2)
   {
      m->data[0] = NR / NG;
      m->data[1] = NG / NB;
      m->data[2] = NR / NB;
   }

   // 3. print user readable information?
   if (print_info)
   {
      printf("model info: d[0]=%.2f, d[1]=%.2f, d[2]=%.2f\n",
         m->data[0], m->data[1], m->data[2]);
   }
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
   // 1. get middle point of region to search
   int swidth = r.width;
   int sheight = r.height;
   int smx = r.tl().x + r.width / 2;
   int smy = r.tl().y + r.height / 2;


   // 2. generate region proposals and corresponding models
   //    and compare them with the search model   
   model* proposal_model = new model;
   double min_dist = -1.0;
   Rect best_rect_found;
   model* best_model_found = new model;
   for (double scale = SCALE_MIN; scale <= SCALE_MAX; scale += SCALE_STEP)
   {
      for (int dy = -SEARCH_DIAMETER; dy < SEARCH_DIAMETER; dy += 2)
      {
         for (int dx = -SEARCH_DIAMETER; dx < SEARCH_DIAMETER; dx += 2)
         {
            // compute compare region middle point and size
            int cmx = smx + dx;
            int cmy = smy + dy;
            int scaled_width = (int)((double)selection.width *scale);
            int scaled_height = (int)((double)selection.height*scale);

            // make sure, scaled proposal region does not become
            // smaller than N pixels in width and height
            if ((scaled_width<MIN_PROPOSAL_REGION_SIZE) ||
               (scaled_height<MIN_PROPOSAL_REGION_SIZE))
               continue;

            // compute proposal rectangle coordinates
            int cx1 = cmx - scaled_width / 2;
            int cy1 = cmy - scaled_height / 2;
            int cx2 = cmx + scaled_width / 2;
            int cy2 = cmy + scaled_height / 2;

            // is the compare region fully within the image?
            if (!((cx1 >= 0) && (cy1 >= 0) && (cx2 < frame.cols) && (cy2 < frame.rows)))
               continue; // no! so go to next region proposal

            // so what is the proposal region finally?
            Rect proposal_region = Rect(Point(cx1, cy1), Point(cx2, cy2));

            // compute model for the proposal region
            compute_model(frame, proposal_region, proposal_model);

            // compute distance
            // between model of region to track
            // and model of proposal region
            double dist = compute_model_distance(search_model, proposal_model);

            // found a better region that matches to the search region?
            if ((min_dist == -1.0) || (dist < min_dist))
            {
               min_dist = dist;
               best_rect_found = proposal_region;
               for (int i = 0; i < MODEL_LEN; i++)
                  best_model_found->data[i] = proposal_model->data[i];
            }

         } // for (x)
      } // for (y)
   } // for (scale)

     // update search model
   if (UPDATE_MODEL)
   {
      for (int i = 0; i < MODEL_LEN; i++)
         search_model->data[i] = best_model_found->data[i];
   }

   // release memory
   delete proposal_model;
   delete best_model_found;

   // return the best region we could find
   return best_rect_found;

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

      if (1)
      {
         char filename[500];
         sprintf_s(filename, "V:\\tmp\\img%04d.png", frame_counter);
         imwrite(filename, visu);
      }

      // 6.6 next frame
      frame_counter++;
   }

   printf("Tracking finished. Press a key to exit!\n");
   _getch();
}
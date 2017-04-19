/// Example of a very simple person tracker
/// that uses detections to track persons
/// (Tracking-By-Detection approach)
///
/// Example video source:
/// https://videos.pexels.com/videos/musicians-and-dancers-on-the-street-2078
/// (CC0 license)
///
/// ---
/// Prof. Dr. Jürgen Brauer, www.juergenbrauer.org


#include "opencv2/opencv.hpp"
#include <conio.h>

using namespace cv;
using namespace std;


int next_hypo_id = 0;

struct PersonHypothesis
{   
   int    id;
   Rect   bounding_box;
   double confidence_value;
   bool   confirmed; 
};


bool similar(Rect r1, Rect r2)
{
   // let's say two rectangles are similar if the center
   // point are not to far away

   Point cp1 = r1.tl() + Point2i(r1.width / 2, r1.height / 2);
   Point cp2 = r2.tl() + Point2i(r2.width / 2, r2.height / 2);
   double dx = cp1.x - cp2.x;
   double dy = cp1.y - cp2.y;
   
   double dist_CPs = sqrt(dx*dx+dy*dy);

   bool b = dist_CPs<40;
   return b;
}


class PersonTracker
{
   public:

      void feed(vector<Rect> detection_boxes)
      {
         for (int hypo_nr = 0; hypo_nr < all_hypos.size(); hypo_nr++)
         {
            PersonHypothesis* h = all_hypos[hypo_nr];

            h->confirmed = false;
         }

         // run through all detected persons
         for (int detection_nr = 0; detection_nr < detection_boxes.size(); detection_nr++)
         {
            // 1. get detection bounding box
            Rect r1 = detection_boxes[ detection_nr ];

            // 2. update all similar hypotheses
            bool found_at_least_one_similar = false;
            for (int hypo_nr = 0; hypo_nr < all_hypos.size(); hypo_nr++)
            {
               PersonHypothesis* h = all_hypos[hypo_nr];

               // get hypothesis bounding box
               Rect r2 = h->bounding_box;

               // are r1 and r2 similar?
               if (similar(r1, r2))
               {
                  found_at_least_one_similar = true;
                  h->confirmed = true;

                  // update confidence value
                  h->confidence_value+=0.01;
                  if (h->confidence_value>1.0)
                     h->confidence_value = 1.0;

                  // update bounding box of hypothesis
                  double alpha=0.2;
                  Point2i tl = alpha*r1.tl() + (1-alpha)*r2.tl();
                  Point2i br = alpha*r1.br() + (1-alpha)*r2.br();
                  h->bounding_box = Rect(tl,br);
               }

            } // for (hypos)

            // 3. did we find a similar hypothesis for the current bbox?
            if (!found_at_least_one_similar)
            {
               // generate new person hypothesis
               PersonHypothesis* p = new PersonHypothesis();
               p->id = next_hypo_id++;
               p->confidence_value = 0.1;
               p->bounding_box = r1; 
               p->confirmed = true;     
               
               // add new person hypothesis pointer to vector
               all_hypos.push_back( p );
            }           
         } // for (detections)


         // decrease confidence value of hypotheses not confirmed
         vector<PersonHypothesis*> remaining_hypos;
         for (int hypo_nr = 0; hypo_nr < all_hypos.size(); hypo_nr++)
         {
            PersonHypothesis* h = all_hypos[hypo_nr];

            if (!h->confirmed)
               h->confidence_value -= 0.01;

            if (h->confidence_value>0.0)
               remaining_hypos.push_back(h);
            else
               delete h;
         }
         all_hypos = remaining_hypos;

      } // feed()

     vector<PersonHypothesis*> all_hypos;

};


int main()
{
   // 1. define a video filename
   string videofilename =
      "V:\\01_job\\00_vorlesungen_meine\\2017_SS_multimodal_sensor_systems\\13_testdata\\videos\\dancing_pair_endpart.mp4";

   // 2. try to open the video file
   VideoCapture cap(videofilename);
   if (!cap.isOpened()) {
      printf("Error! Could not open video file %s\n", videofilename.c_str());
      _getch();
      return -1;
   }

   // 3. prepare a HOG person detector & a person tracker
   HOGDescriptor hog;
   hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
   PersonTracker my_tracker;

   // 4. read in frames from video and process them   
   int frame_counter = 0;
   Mat frame_detections;
   Mat frame_tracks;
   while (true)
   {
      // 4.1 show video frame nr
      printf("frame: %d\n", frame_counter);

      // 4.2 get next video frame
      Mat frame;
      cap >> frame;

      // 4.3 is it a valid video frame?
      if (frame.empty())
         break; // no! so quit this loop!

      // 4.4 get frame dimensions
      int width = frame.cols;
      int height = frame.rows;

      // 4.5 resize frame
      resize(frame, frame, Size(width / 3, height / 3));

      // 4.6 detect persons in frame
      vector<Rect> found, found_filtered;
      double t = (double)getTickCount();
      hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 1.5);
      t = (double)getTickCount() - t;
      cout << "detection time = " << (t*1000. / cv::getTickFrequency()) << " ms" << endl;

      // 4.7 show all detection bounding boxes
      frame_detections = frame.clone();
      for (size_t i = 0; i < found.size(); i++)
      {
         Rect r = found[i];
         rectangle(frame_detections, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
      }

      // 4.8 feed person tracker with detections
      my_tracker.feed( found );

      // 4.9 show all person hypotheses
      frame_tracks = frame.clone();

      // show frame nr
      char txt[100];
      sprintf_s(txt, "%04d", frame_counter);
      putText(frame_tracks,
         txt,
         Point(10, frame_tracks.rows - 30),
         FONT_HERSHEY_SIMPLEX, 0.5, // font face and scale
         CV_RGB(0, 255, 0),
         1); // line thickness and type
      
      for (int i=0; i<my_tracker.all_hypos.size(); i++)
      {
         // get pointer to person hypothesis
         PersonHypothesis* h = my_tracker.all_hypos[i];
         
         // show hypo id and current confidence value
         char txt[100];
         sprintf_s(txt, "#%d %.2f", h->id, h->confidence_value);
         putText(frame_tracks,
            txt,
            Point(h->bounding_box.tl().x, h->bounding_box.br().y+15),
            FONT_HERSHEY_SIMPLEX, 0.5, // font face and scale
            CV_RGB(255, 255, 0),
            1); // line thickness and type

         // draw hypo bounding box
         rectangle(frame_tracks, h->bounding_box, CV_RGB(255,255,0), 2);
      }
      printf("There are %d person hypotheses\n", (int) my_tracker.all_hypos.size());


      // 4.10 write images to some file
      int W = frame_tracks.cols;
      int H = frame_tracks.rows;
      Mat big_img = Mat(H*2, W, CV_8UC3);
      Mat region1 = Mat(big_img, Rect(0,0,W,H));
      Mat region2 = Mat(big_img, Rect(0,H,W,H));
      frame_detections.copyTo(region1);
      frame_tracks.copyTo(region2);
      if (0)
      {
         char filename[500];
         sprintf_s(filename, "V:\\tmp\\img%04d.png", frame_counter);
         imwrite(filename, big_img);
      }
      imshow("Detections and tracked hypotheses", big_img);


      // 4.11 important for imshow() to work
      waitKey(0);

      // 4.12 next frame
      frame_counter++;
   }

   return 0;
}
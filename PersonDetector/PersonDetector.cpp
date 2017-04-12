/// Example of a person detection using OpenCV 
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

int main()
{
   // 1. define a video filename
   string videofilename =
    "V:\\01_job\\00_vorlesungen_meine\\2017_SS_multimodal_sensor_systems\\13_testdata\\videos\\dancing_pair.mp4";
    //"V:\\01_job\\00_vorlesungen_meine\\2017_SS_multimodal_sensor_systems\\13_testdata\\videos\\new_york_city_pedestrian_crossing.mp4";
    
   // 2. try to open the video file
   VideoCapture cap(videofilename);
   if (!cap.isOpened()) {
      printf("Error! Could not open video file %s\n", videofilename.c_str());
      _getch();
      return -1;
   }

   // 3. prepare a HOG person detector
   HOGDescriptor hog;
   hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

   // 4. read in frames from video and process them   
   int frame_counter = 0;
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
      resize(frame, frame, Size(width/3,height/3));

      // 4.6 detect persons in frame
      vector<Rect> found, found_filtered;
      double t = (double)getTickCount();
      hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 1.0);

      t = (double)getTickCount() - t;
      cout << "detection time = " << (t*1000. / cv::getTickFrequency()) << " ms" << endl;
            
      // 4.7 show all detection bounding boxes
      for (size_t i = 0; i < found.size(); i++)
      {
         Rect r = found[i];
         rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
      }

      // 4.8 show frame nr
      char txt[100];
      sprintf_s(txt, "%04d", frame_counter);
      putText(frame,
         txt,
         Point(10,frame.rows-30),
         FONT_HERSHEY_SIMPLEX, 0.5, // font face and scale
         CV_RGB(0, 255, 0),
         1); // line thickness and type

      // 4.9 show detected people
      imshow("people detector", frame);

      // 4.10 write images to some file
      char filename[500];
      sprintf_s(filename, "V:\\tmp\\img%04d.png", frame_counter);
      imwrite(filename, frame);

      // 4.11 important for imshow() to work
      waitKey(1);

      // 4.12 next frame
      frame_counter++;
   }

   return 0;
}
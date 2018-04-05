import sys
print ("Your Python version is: " + sys.version)

import cv2
print("Your OpenCV version is: "+ cv2.__version__)

import numpy as np
from scipy import ndimage


class tennis_ball_detector_SLOW:

    def __init__(self):
        print("A new SLOW tennis ball detector has been created.")

    def detect_the_ball(self, frame):

        height = frame.shape[0]
        width  = frame.shape[1]

        sumx = 0
        sumy = 0
        counter = 0

        yellow_pixels = []

        for y in range(0,height):
            for x in range(0,width):

                blue  = frame[y, x, 0]
                green = frame[y, x, 1]
                red   = frame[y, x, 2]

                if red>=150 and green>=150 and blue<100:
                    sumx += x
                    sumy += y
                    counter += 1
                    yellow_pixels.append( np.array((x,y)) )

        detected = False
        cx = -1
        cy = -1
        avg_dist = 0
        if counter>10:
            print("Tennis ball detected. Nr of yellow pixels:", counter)
            detected = True

        if detected:
            cx = int(sumx/counter)
            cy = int(sumy/counter)
            center = np.array((cx,cy))

            sum_dist = 0.0
            for pix in yellow_pixels:
                dist = np.linalg.norm(center - pix)
                sum_dist += dist
            avg_dist = int( sum_dist/len(yellow_pixels) )

            print("cx=", cx, "cy=", cy, "avg_dist=", avg_dist)

        return detected, cx, cy, avg_dist


class tennis_ball_detector_FAST:

    def __init__(self):
        print("A new FAST tennis ball detector has been created.")

    def detect_the_ball(self, frame):

        B = frame[:,:,0]
        G = frame[:,:,1]
        R = frame[:,:,2]
        B = np.where(B  < 90, 1, 0)
        G = np.where(G >= 170, 1, 0)
        R = np.where(R >= 170, 1, 0)
        Y = B & G & R
        nr_yellow_pixels = Y.sum()

        detected = False
        cx = -1
        cy = -1
        avg_dist = 0
        if nr_yellow_pixels>10:
            #print("nr_yellow_pixels=", nr_yellow_pixels)
            detected = True
            center = ndimage.measurements.center_of_mass(Y)
            cy = int(center[0])
            cx = int(center[1])
            avg_dist = 5

        return detected, cx, cy, avg_dist



output_folder = "V:/tmp/"


# open an image source

# image source = default web-cam
#source = 0

# image source = video-file
source = "V:/01_job/12_datasets/test_videos/tennis_ball_green_tshirt.mp4"

# open the image source
cap = cv2.VideoCapture(source)

det = tennis_ball_detector_FAST()

image_nr=0
while (1):

    # 1. read the next frame from the image source
    ret, frame = cap.read()
    if ret==False:
        break
    frame = cv2.resize(frame,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # 2. detect the tennis ball
    detected, cx,cy, avg_dist = det.detect_the_ball(frame)
    if detected:
        cv2.circle(frame, (cx, cy), avg_dist, (0, 0, 255), 2)

    # 3. write image to file
    numberstr = "{0:0>6}".format(image_nr)
    filename = output_folder + "/" + numberstr + ".png"
    cv2.imwrite(filename, frame)

    # 4. one image saved more
    image_nr += 1

    # 5. show original camera image + edge image
    cv2.imshow('Original', frame)

    # 6. does the user want to exit?
    k = cv2.waitKey(5)
    if k == 27:
        break


    if image_nr>=2:
        diff_img = abs(frame2 - frame_gray)
        diff_img = cv2.inRange(diff_img, 100,255)
        cv2.imshow("diff image", diff_img)

    frame2 = frame_gray



cap.release()
cv2.destroyAllWindows()
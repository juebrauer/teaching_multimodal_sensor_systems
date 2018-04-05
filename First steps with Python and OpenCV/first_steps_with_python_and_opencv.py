import sys
print ("Your Python version is: " + sys.version)

import cv2
print("Your OpenCV version is: "+ cv2.__version__)

output_folder = "V:/tmp"

# 1. Open an image source

# default web-cam
#source = 0

# video-file
source = "V:/01_job/12_datasets/test_videos/street_view_kempten.mp4"

# open the image source
cap = cv2.VideoCapture(source)

image_nr=0
while (True):

    # 2. read the next frame from the image source
    ret, frame = cap.read()
    if ret==False:
        break

    # 3. show original camera image + edge image
    cv2.imshow('Original', frame)
    edge_frame = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edge_frame)

    k = cv2.waitKey(5)
    if k == 27:
        break

    # 4. write edge image to file
    numberstr = "{0:0>6}".format(image_nr)
    filename = output_folder + "/" + numberstr + ".png"
    cv2.imwrite( filename, edge_frame )

    # 5. one image saved more
    image_nr +=1

cap.release()
cv2.destroyAllWindows()
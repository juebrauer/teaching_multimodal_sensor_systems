'''
File: selective_search.py (Exercise 15)

In this exercise students shall get OpenCV's "Selective Search"
implementation running.

Note:
1) 
The createSelectiveSearchSegmentation() function is
not available in OpenCV 3.1.0 and it is buggy in OpenCV 3.2.0.
So you should install at least OpenCV 3.3.0 in order to be able
to use it.

See also this blog post by Satya Mallick. At the end of the post
you can find some information about the OpenCV version issue:
https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

2)
The code here is inspired by the code by Satya Mallick posted at
https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

'''


import sys
import cv2

print("Your Python version is: " + sys.version)
print("Your OpenCV version is: " + cv2.__version__)

print()
print("Selective Search algorithm demo.")
print("Selective Search is an algorithm that generates")
print("region proposals that may contain objects or object parts.")
print()
print("For this, it iteratively fuses the two most similar segments")
print("of an initially oversegmented image.")
print()
print("Press 'm' for showing more region candidates.")
print("Press 'l' for showing less region candidates.")

test_img_nr = 3
if test_img_nr==1:
    image_filename = "eating_ice_cream.jpg"
elif test_img_nr==2:
    image_filename = "at_lake_tahoe.jpg"
elif test_img_nr==3:
    image_filename = "a_stanford_university_building.jpg"

fast_or_quality = "f"

# 1. speed-up using multithreads

# this function can be used to dynamically turn on and off
# optimized code (code that uses SSE2, AVX, and other instructions
# on the platforms that support it)
cv2.setUseOptimized(True);

# OpenCV will try to set the number of threads for the
# next parallel region.
cv2.setNumThreads(4);


# 2. read image
im = cv2.imread( image_filename )


# 3. resize image to fixed height
#    but make sure we keep the aspect ratio
newHeight = 800
newWidth = int(im.shape[1] * newHeight / im.shape[0])
im = cv2.resize(im, (newWidth, newHeight))


# 4. create Selective Search Segmentation Object
#    using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


# 5. set input image on which we will run segmentation
ss.setBaseImage(im)


# 6. Switch to fast but low recall Selective Search method
if (fast_or_quality == 'f'):
    ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
else:
    ss.switchToSelectiveSearchQuality()


# 7. run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))


# 8. number of region proposals to show
numShowRects = 100
increment = 50


# 9. increment to increase/decrease total number
#    of region proposals to be shown

while True:

    # 9.1 create a copy of original image
    imOut = im.copy()

    # 9.2 itereate over all the region proposals
    for i, rect in enumerate(rects):

        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    # 9.3 show output
    cv2.imshow("Output", imOut)

    # 9.4 record key press
    k = cv2.waitKey(0) & 0xFF

    # 9.5 m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break

    print("Now showing", numShowRects, "region candidates.")


# 10. close all windows
cv2.destroyAllWindows()
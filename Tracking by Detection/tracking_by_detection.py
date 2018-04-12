import sys
print ("Your Python version is: " + sys.version)

import cv2
print("Your OpenCV version is: "+ cv2.__version__)

import math

import numpy as np



######################
class person_detector:
######################

    hog = None

    def __init__(self):
        print("A new person detector has been created.")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):

        height = frame.shape[0]
        width  = frame.shape[1]

        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
                                                     padding=(8, 8), scale=1.05)

        fweights = [float(w) for w in weights]
        return (rects, fweights)


# from: https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections
###########################################
def bb_intersection_over_union(boxA, boxB):
###########################################

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



#######################################
def do_correspond(det_rect, hypo_rect):
#######################################

    #iou = bb_intersection_over_union(det_rect,hypo_rect)

    dx = det_rect[0] - hypo_rect[0]
    dy = det_rect[1] - hypo_rect[1]
    dist = math.sqrt(dx*dx + dy*dy)

    diff_width  = abs(det_rect[2] - hypo_rect[2])
    diff_height = abs(det_rect[3] - hypo_rect[3])

    if (dist<60 and diff_width<30 and diff_height<30):
        return True
    else:
        return False


##################
class person_hypo:
##################

    id     = -1
    rect   = None
    weight = None
    conf   = 0.0



##############
class tracker:
##############

    next_id = 0
    hypos = []

    def __init__(self):
        print("A new person tracker has been created.")


    def feed(self, detection_rects, detection_weights):


        # 1. for all detections ...
        for det_nr in range(0, len(detection_rects)):

            # 1.1 get the detection rectangle and weight
            det_rect = detection_rects[det_nr]
            det_weight = detection_weights[det_nr]

            # 1.2 for all hypotheses ...
            correspondence_found = False
            for hypo_nr in range(0, len(self.hypos)):

                # get hypothesis rectangle and weight
                hypo_rect = self.hypos[hypo_nr].rect
                hypo_weight = self.hypos[hypo_nr].weight

                # does the detection match with
                # the hypothesis?
                correspond = do_correspond(det_rect, hypo_rect)
                if correspond:
                    correspondence_found = True

                    # update hypothesis by the detection!
                    scale = 0.2
                    self.hypos[hypo_nr].rect   = hypo_rect*scale   + det_rect  *(1.0-scale)
                    self.hypos[hypo_nr].weight = hypo_weight*scale + det_weight*(1.0-scale)
                    self.hypos[hypo_nr].conf   += 2

            # end for (all current hypos)


            # 1.3 did we find at least one correspondence for the
            # detection rectangle?
            # if not: create a new hypothesis!
            if correspondence_found == False:

                # prepare a new hypothesis
                h = person_hypo()
                h.id = tracker.next_id
                h.conf = 2
                h.weight = det_weight
                h.rect = det_rect
                tracker.next_id += 1

                # add new hypothesis to list of hypotheses
                self.hypos.append( h )

                print("Generated a new person hypothesis: "\
                      "id =",h.id)

            # end if

        # end for (all detections)



        # 2. decrease all confidences of hypotheses by one
        for hypo_nr in range(0, len(self.hypos)):
            self.hypos[hypo_nr].conf -= 1


        # 3. for all current hypos:
        # decide whether to take them into the new round or not
        surviving_hypos = []
        for hypo_nr in range(0, len(self.hypos)):

            # 3.1 no confidence any longer in the hypothesis?
            if self.hypos[hypo_nr].conf == 0:
                print("deleting hypothesis nr ", hypo_nr,
                      "with id =", self.hypos[hypo_nr].id)
            else:
                surviving_hypos.append( self.hypos[hypo_nr])
        self.hypos = surviving_hypos


        # 4. print list of current person hypotheses
        print("There are currently ", len(self.hypos), "person hypotheses.")
        for hypo_nr in range(0, len(self.hypos)):
            print("\thypo nr", hypo_nr,
                  " with id = ", self.hypos[hypo_nr].id,
                  " which has conf=", self.hypos[hypo_nr].conf)


        # 5. merge very similar hypotheses to one
        #    idea: if there are two similar ones,
        #          only let the first one into the next round
        survival_flags = []
        for hypo_nr in range(0, len(self.hypos)):
            survival_flags.append(1) # 1=survives, -1=will be killed

        for hypo_nr1 in range(0, len(self.hypos)):
            if (survival_flags[hypo_nr1]==1):
                for hypo_nr2 in range(0, len(self.hypos)):
                    if (hypo_nr2>hypo_nr1):
                        if do_correspond(self.hypos[hypo_nr1].rect, self.hypos[hypo_nr2].rect):
                            survival_flags[hypo_nr1] = 1
                            survival_flags[hypo_nr2] = -1

        surviving_hypos = []
        for hypo_nr in range(0, len(self.hypos)):
            if survival_flags[hypo_nr]==1:
                surviving_hypos.append(self.hypos[hypo_nr])
        self.hypos = surviving_hypos


        # 5. return all the current hypotheses
        return self.hypos




output_folder = "V:/tmp/"




# open an image source

# image source = default web-cam
#source = 0

# image source = video-file
source = "V:/01_job/12_datasets/test_videos/zebra3d.mp4"

# open the image source
cap = cv2.VideoCapture(source)

det = person_detector()
tra = tracker()

image_nr=0
while (1):

    # 1. read the next frame from the image source
    ret, frame = cap.read()
    if ret==False:
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    # 2. detect persons
    (det_rects, det_weights) = det.detect(frame)

    # 3. draw the detection bounding boxes
    dets_visu = frame.copy()
    for det_nr in range(0,len(det_rects)):
        det_rect   = det_rects[det_nr]
        det_weight = det_weights[det_nr]
        x = det_rect[0]
        y = det_rect[1]
        w = det_rect[2]
        h = det_rect[3]
        cv2.rectangle(dets_visu, (x, y), (x + w, y + h), (0, 0, 255), 2)
        txt = "{:.2f}".format(det_weight)
        cv2.putText(dets_visu, txt, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    # 4. feed the detections into the tracker
    hypos = tra.feed(det_rects, det_weights)

    # 5. draw the hypotheses bounding boxes
    hypos_visu = frame.copy()
    for hypo_nr in range(0,len(hypos)):
        if hypos[hypo_nr].conf>10:
            x = int(hypos[hypo_nr].rect[0])
            y = int(hypos[hypo_nr].rect[1])
            w = int(hypos[hypo_nr].rect[2])
            h = int(hypos[hypo_nr].rect[3])
            cv2.rectangle(hypos_visu, (x, y), (x + w, y + h), (0, 255, 0), 2)
            txt = str(hypos[hypo_nr].id) + " - " +\
                  "{:.0f}".format(hypos[hypo_nr].conf)
            cv2.putText(hypos_visu, txt, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 6. show detections & hypotheses
    combi = np.concatenate((dets_visu, hypos_visu), axis=0)
    #cv2.imshow('Person detections', dets_visu)
    #cv2.imshow('Person hypotheses', hypos_visu)
    cv2.imshow('HOG detections (top) and tracking results (bottom)', combi)

    # 7. does the user want to exit?
    k = cv2.waitKey(0)
    if k == 27:
        break

    # 8. write image to file
    numberstr = "{0:0>6}".format(image_nr)
    filename = output_folder + "/" + numberstr + ".png"
    cv2.imwrite(filename, combi)

    # 9. one image saved more
    image_nr += 1



cap.release()
cv2.destroyAllWindows()
###############################################
# AUTHOR : FANTASTIC FOUR                     #
# TITLE  : GOD'S EYE - CAR DETECTION          #
# DATE   : 4/03/2022                          #
###############################################

import cv2
import numpy as np

def detectCar(clrB, clrA):
    new = clrA.copy()
    grayA = cv2.cvtColor(clrA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(clrB, cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh,kernel,iterations = 1) 
    contours, _ = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i,cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr)
        if (w>20) & (h>20):
            cv2.rectangle(new, (x, y),(x+w,y+h),(0, 255, 0))
    cv2.imshow("Video",new)
    out.write(frame)


out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
vidcap = cv2.VideoCapture("assets/carVideo.mp4")
col_images=[]
flag=True

while True :
    ret, frame = vidcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    col_images.append(frame)
    if flag:
        flag=False
        continue
    detectCar(col_images[-2], col_images[-1])
    if cv2.waitKey(33) == 27:
        break
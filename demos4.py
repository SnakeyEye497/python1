# this module is used for machine learning kindly ignore it



# install cvzone and mediapipe
# cvzone is dedicated for hand module processing
# mediapipe is associated with machine learning

import time
import math
import cv2                                                                 # to work with images
import numpy as np                                                         # import numpy
from cvzone.HandTrackingModule import HandDetector                         #import hand specific module

cap = cv2.VideoCapture(0)                                                  #capture continuous  output in cap
detect = HandDetector(maxHands=2)                                          # decide how many hands should be detected module
ImgSize = 300                                                              # simple variable

counter=0

while True:                                                                #infinite loop
    success, img = cap.read()                                              #continous read output of cap in img
    hands, img = detect.findHands(img)                                     # find hands from img

    if hands:                                                              # to crop hands
        hand = hands[0]

        x, y, w, h = hand['bbox']                                          #get for corner of hands
        imageWhite = np.ones((ImgSize,ImgSize,3), np.uint8)*255            #specify color of whiteframe
        imgCrop = img[y-20:y+h+20, x-20:x+w+20]                            #genearte crop image

        aspectR = h/w                                                      # to resize our cropped image on white screen
        if aspectR > 1:                                                    # resize if height is smaller than frame
            k = ImgSize/h
            wCal = math.ceil(k*w)                                          # width use to take cropimg to center of white frame
            imgResize = cv2.resize(imgCrop,(wCal,ImgSize))                 # resize our cropped img for white  frame
            ImjCropRShape = imgResize.shape                                # manage co-ordinate of img cropped of hand
            wGap=math.ceil((ImgSize-wCal)/2)                               # width-gap use to take cropped img to center og white frame
            imageWhite[ :,wGap:wCal+wGap] = imgResize                      # add img cropped of hand on white frame

        else :                                                             # resize if height is smaller than frame
            k = ImgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(hCal,ImgSize))
            ImjCropRShape = imgResize.shape
            hGap=math.ceil((ImgSize-hCal)/2)
            imageWhite[ :,hGap:hCal+hGap] = imgResize

        cv2.imshow("imageCrop", imgCrop)                                 #display
        cv2.imshow("imageWhite", imageWhite)                             # white image


    cv2.imshow("Image", img)                                                  # show out result of hands in a window
    key=cv2.waitKey(1)                                                       # buffer time given to detect hands









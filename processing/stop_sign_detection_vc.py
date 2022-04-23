import cv2 as cv
import numpy as np
import pytesseract
import sys

sys.path.append("/Users/cassini/Desktop/PyVision-main/src/tessexc")
sys.path.append("/Users/cassini/Desktop/PyVision-main/src/OCR")

import text_extraction

#Set pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'/Usr/local/bin/tesseract'

#initialize camera 
camera = cv.VideoCapture(0)

def extractText(mask):
    if 5 >= len(text_extraction.extractText(mask)) >= 2:
        return True
    
    return False 

def isOctagon(contour):
    edges = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    
    if 9 >= len(edges) >= 8:
        return True
    
    return False

def extractContours(mask):
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 2:
        contours = contours[0]
    
    else:
        contours = contours[1]
    
    return contours


def createMask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask_red_1 = cv.inRange(hsv, lower_red_1, upper_red_1)

    lower_red_2 = np.array([160, 100, 100])
    upper_red_2 = np.array([179, 255, 255])
    mask_red_2 = cv.inRange(hsv, lower_red_2, upper_red_2)
    
    mask_red = mask_red_1 + mask_red_2
    return mask_red
 
    

def drawRect(mask, img, contour):
    if isOctagon(contour) and cv.contourArea(contour) > 200 and extractText(mask):
        x, y, w, h = cv.boundingRect(contour)
        if 1.1 >= w / h >= .9:
            cv.rectangle(img, (x, y), (x + w, w + h), (0, 255, 0), 3)
        

#indefinite loop for processing the live feed
while True:
    ret, frame = camera.read()
    for contour in extractContours(createMask(frame)):
        drawRect(createMask(frame), frame, contour)
        
    cv.imshow("frame", frame)
    cv.imshow("mask", createMask(frame))
    key = cv.waitKey(1)
    if key == 27:
        break
    

camera.release()
cv.destroyAllWindows()

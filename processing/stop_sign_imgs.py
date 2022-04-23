import cv2 as cv
import numpy as np
import pytesseract
import random
import sys

sys.path.append("/Users/cassini/Desktop/PyVision-main/src/tessexc")
sys.path.append("/Users/cassini/Desktop/PyVision-main/src/OCR")

import text_extraction

#Set pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'/Usr/local/bin/tesseract'

images = [r"/Users/cassini/Downloads/stop1.png", r"/Users/cassini/Downloads/stop3.jpeg", r"/Users/cassini/Downloads/stop2.jpeg",
          r"/Users/cassini/Downloads/stop5.jpeg", r"/Users/cassini/Downloads/stop7.jpeg", r"/Users/cassini/Downloads/stop6.webp"]

#initialize camera 
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
            min_rect = cv.minAreaRect(contour)
            box = cv.boxPoints(min_rect)
            box = np.int0(box)
            cv.drawContours(img, [box], 0, (0, 255, 0), 3)
            percentage = random.randint(20, 100)
            cv.putText(img, f"{percentage}%", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def detectStopSigns(frame):
    for contour in extractContours(createMask(frame)):
        drawRect(createMask(frame), frame, contour)
        
    return frame


#indefinite loop for processing the live feed
for image in images:
    frame = cv.imread(image)
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    map(detectStopSigns(frame), frame)

    cv.imshow("frame", frame)
    cv.waitKey(0)
    
cv.destroyAllWindows()


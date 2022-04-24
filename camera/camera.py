import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
import progressbar

previous = 0
current = 0

class Camera:
    #class variables
    LENGTH = 640
    WIDTH = 480
    SOURCE = 0
    WINDOWS_MODE = False
    EXPOSURE = -1
    SHOW = False

    def __init__(self, model = "default USB camera", source = 0):
        self.model = model
        self.source = source
        widgets = ['$Initializing Camera Object ->: ', progressbar.AnimatedMarker()]
        bar = progressbar.ProgressBar(widgets = widgets).start()
      
        for i in range(10):
            time.sleep(0.1)
            bar.update(i)
        
       
    #Mutators 
    def setDimensions(self, length, width):
        self.LENGTH = length
        self.WIDTH = width
        
    def setExposure(self, exposure):
        self.EXPOSURE = exposure
        
    def setSource(self, source):
        self.SOURCE = source
    
    def setWindowsMode(windows_mode):
        self.WINDOWS_MODE = windows_mode
        
    #Accessors
    def getDimensions(self):
        return self.LENGTH, self.WIDTH
    
    def getExposure(self):
        return self.EXPOSURE
    
    def getSource(self):
        return self.SOURCE
    
    
    #Handle frame rate
    def displayFPS(feed):
        global current, previous
        current = time.time()
    
        fps = 1/(current - previous)
        previous = current
        fps = int(fps)
        
        cv.putText(feed, str(fps), (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv.LINE_AA)
        
        
    #Initialize camera object
    def initialize(self):
        if Camera.WINDOWS_MODE:
            self.camera = cv.VideoCapture(self.SOURCE, cv.CAP_DSHOW)
        else:
            self.camera = cv.VideoCapture(self.SOURCE)
            
        self.camera = cv.VideoCapture(Camera.SOURCE)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.LENGTH)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.WIDTH)
        self.camera.set(cv.CAP_PROP_EXPOSURE, self.EXPOSURE)
    
    def read(self):
        return self.camera.read()
    
    
    def displayFeed(self):
        while True:
            ret, frame = self.camera.read()
            Camera.displayFPS(frame)
            cv.imshow("frame", frame)
            key = cv.waitKey(1)
            if key == 27:
                break
            

build = input()

if build == "--build":
    widgets = [' $Starting Build ', progressbar.AnimatedMarker()]
    bar = progressbar.ProgressBar(widgets = widgets).start()
          
    for i in range(20):
        time.sleep(0.1)
        bar.update(i)

    print("$Build completed")
    
    




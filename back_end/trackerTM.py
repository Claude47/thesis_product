"""
    Claude Betz (BTZCLA001)
    
    tracker_TM.py
    Class that implements both Simple and Adaptive Template Matching Trackers
"""

# Imports
import numpy as np
import cv2 as cv

# Class imports
from back_end.tracker import tracker

class trackerTM(tracker): # inherits from tracker
    # instance variables
    ttype = "simple" # check what type to apply. True is simple
    threshold = 0.8 # default threshold for matching

    def __init__(self, file_path):
        super().__init__(file_path)  # call to super tracker constructor
    
    # inherited methods
    def load_sequence(self, file_path):
        return super().load_sequence(file_path)

    def select_roi(self, frame):
        return super().select_roi(frame)

    def draw_bounds(self, frame, y, x, h, w):
        return super().draw_bounds(frame, y, x, h, w)

    def create_results_directory(self, dir_name, user_defined=False):
        return super().create_results_directory(dir_name, user_defined=False)

    def image_write(self, image, seq_name="seq"):
        return super().image_write(image, seq_name="seq")
    

    # defined methods
    def set_threshold(threshold):
        """allows setting of sensitivity threshold"""
        self.threshold = threshold

    def setup(self, frame0, ROI, ttype="simple"):
        """initialsises tracker with template model"""
        frame_gray = None
        if(len(frame0.shape)==3):
            frame_gray = cv.cvtColor(frame0,cv.COLOR_BGR2GRAY)
        else:
            frame_gray = frame0
        self.y0 = ROI[0] # get dims (y,x,h,w) - 0,1,2,3
        self.x0 = ROI[1]
        self.h = ROI[2]
        self.w = ROI[3]
        self.template = frame_gray[self.y0:self.y0+self.h, self.x0:self.x0+self.w] # isolate target template
        self.ttype = ttype # set tracker type

    def simpleTM(self, frame):
        """perform simple template matching"""
        res = cv.matchTemplate(frame, self.template, cv.TM_CCOEFF_NORMED)
        if(np.max(res)<self.threshold): # control threshold
            return -1
        loc = np.where(res==np.max(res))
        loc = (loc[0].flatten()[0], loc[1].flatten()[0])        
        return loc

    def adaptiveTM(self, frame):
        """perform adaptive template matching"""
        res = cv.matchTemplate(frame, self.template, cv.TM_CCOEFF_NORMED)
        if(np.max(res)<self.threshold): # control threshold
            return -1
        loc = np.where(res==np.max(res))
        loc = (loc[0].flatten()[0], loc[1].flatten()[0])        
        
        h, w = self.template.shape[:2] # get template dimensions
        self.template = frame[loc[0]:loc[0]+h, loc[1]: loc[1]+w] # update template
        return loc

    def track(self, frame):
        """track for gui"""
        if(frame is None):
            return -1

        # Apply template matching to next frame
        if(len(frame.shape)==3):
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        if(self.ttype=="simple"): # simple matching
            loc = self.simpleTM(frame_gray) 
            return loc

        elif(self.ttype=="adaptive"): # adaptive matching
            loc = self.adaptiveTM(frame_gray) 
            return loc 
        else:
            return -1


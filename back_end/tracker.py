"""
    Claude Betz (BTZCLA001)
    tracker.py

    General tracker class that implements utility functions that are inherited by
    specific tracker implementations.
"""

# Imports
import cv2 as cv
import numpy as np
import os, errno

class tracker:
    """template tracking generic functions"""
    tmp_path = "tmp"
    write_count = 0
    cap = None
    def __init__(self, file_path):
        """create new instance of template tracker class"""
        self.cap = cv.VideoCapture(file_path, cv.CAP_IMAGES) # make object of sequence
        self.create_results_directory(self.tmp_path) # create directory for tracker results

    # utility methods
    def load_sequence(self, file_path):
        """method to open file"""
        self.cap = cv.VideoCapture(file_path, cv.CAP_IMAGES) 

    def select_roi(self, frame):
        """function for user to define tempalte within an image""" 
        # select region of interest
        showCrosshair = False
        fromCenter = False
        (x,y,w,h) = cv.selectROI("Image", frame, fromCenter, showCrosshair)
        return y,x,h,w 

    def draw_bounds(self, frame, y, x, h, w):
        """draws the bounding box at target location"""
        centre = (x+w//2,y+h//2)
        #cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2) # draw rectangle
        cv.circle(frame, centre, 3, (0,255,255),-1) # draw center
        cv.ellipse(frame, centre, (w//2, h//2),0,360,0,1)# draw ellipse   

    def create_results_directory(self, dir_name, user_defined=False):
        """creates directory to store results"""
        self.results_path = os.getcwd()+ "/" + dir_name # newly created results directory
        
        if(user_defined==False):
            self.results_path = dir_name
            
        try:
            os.makedirs(self.results_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise    

    def image_write(self, image, seq_name="seq"):
        """writes images to file"""
        print("writing to file...")
        c = format(self.write_count, '08') # pad count
        path = self.tmp_path + "/" + seq_name + c + ".jpg" # append fname, no, extnsn
        cv.imwrite(path, image) # write image to path
        self.write_count = self.write_count + 1 # increment internal counter


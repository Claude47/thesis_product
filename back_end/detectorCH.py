"""
    Claude Betz (BTZCLA001)

    detectorCH.py
    implements the cooccurance histogram template detector.
"""

# Imports 
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

from back_end.tracker import tracker

class detectorCH(tracker):
    nd = 12 # distances for CCH
    nc = 8 # colour quantization levels
    alpha = 0.7 # threshold
    q = None

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
    def RGB_euclidean_distance(self, pixel):
        """computes euclidean distance between two 2D vectors"""
        return np.sqrt(pixel[0]**2+pixel[1]**2+pixel[2]**2) 

    def RGB_quantize_model(self, template):
        """quantize the model RGB colour space using Kmeans based off RGB euclidean norm"""
        # get template dims
        height, width = template.shape[0], template.shape[1] # get dimensions
        
        rgb_euclidean = np.zeros(shape=(height,width)) # to hold RGB euclidian distances for Kmeans clustering
        for y in range(0,height):
            for x in range(0,width):
                pixel = template[y,x] # get relevant pixel
                rgb_euclidean[y,x] = self.RGB_euclidean_distance(pixel)
        
        # Apply K-means algorithm to quantize pixels according to nc    
        self.kmeans = KMeans(n_clusters=self.nc, random_state=0)
        q_labels = self.kmeans.fit_predict(np.expand_dims(rgb_euclidean.flatten(), axis=1))
        model_labels = np.reshape(q_labels,(height,width)) # return the labels in 2D quantized dims
        return model_labels

    def RGB_quantize_candidate(self, template):
        """quantize candidate RGB colour space using Kmeans based off model"""
        # get template dims
        height, width = template.shape[0], template.shape[1] # get dimensions
        
        rgb_euclidean = np.zeros(shape=(height,width)) # to hold RGB euclidian distances for Kmeans clustering
        for y in range(0,height):
            for x in range(0,width):
                pixel = template[y,x] # get relevant pixel
                rgb_euclidean[y,x] = self.RGB_euclidean_distance(pixel)
            
        print(rgb_euclidean.shape)
        # Apply K-means algorithm to quantize pixels according to nc    
        q_labels = self.kmeans.predict(np.expand_dims(rgb_euclidean.flatten(), axis=1))
        candidate_labels = np.reshape(q_labels,(height,width)) # return the labels in 2D quantized dims
        return candidate_labels

    def CH_vertical(self, q_labels2d):
        """defines the vertical cooccurance histogram for a specific template according to quantized colourspace labels matrix and distances"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_y = np.zeros(shape=(self.nc, self.nc, self.nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1, self.nd):
            for y in range(0,height-k_range):
                for x in range(0,width):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y+k_range,x] # get destincation index for CH_y at (y+k,x)
                    CH_y[src_index,dst_index,k_range-1] = CH_y[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
                    
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_vertical = [np.sum((CH_y[:,:,k_range],CH_y[:,:,k_range].T), axis=0) for k_range in range(0, self.nd-1)]
        return CH_vertical

        
    def CH_horizontal(self, q_labels2d):
        """defines the horizontal cooccurance histogram for a specific template according to quantized colourspace labels matrix and distances"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_x = np.zeros(shape=(self.nc, self.nc, self.nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1, self.nd):
            for y in range(0,height):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_x[src_index,dst_index,k_range-1] = CH_x[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_horizontal = [np.sum((CH_x[:,:,k_range], CH_x[:,:,k_range].T), axis=0) for k_range in range(0, self.nd-1)]
        return CH_horizontal

    def CH_pos_diagonal(self, q_labels2d):
        """defines the positve diagonal cooccurance histogram for a specifica template according to quantized colourspace labels"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_pd = np.zeros(shape=(self.nc, self.nc, self.nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1, self.nd):
            for y in range(height-1, 0+k_range-1, -1):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y-k_range,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_pd[src_index,dst_index,k_range-1] = CH_pd[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_pos_diag = [np.sum((CH_pd[:,:,k_range], CH_pd[:,:,k_range].T), axis=0) for k_range in range(0, self.nd-1)]
        return CH_pos_diag

    def CH_neg_diagonal(self, q_labels2d):
        """defines the negative diagonal cooccurance histogram for a specific template according to quantized colourspace labels"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
            
        CH_nd = np.zeros(shape=(self.nc, self.nc, self.nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1, self.nd):
            for y in range(0,height-k_range):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y+k_range,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_nd[src_index,dst_index,k_range-1] = CH_nd[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_neg_diag = [np.sum((CH_nd[:,:,k_range], CH_nd[:,:,k_range].T), axis=0) for k_range in range(0, self.nd-1)]
        return CH_neg_diag

    def CH_model(self, template):
        """computes overall CH along vertical and horizontal axes"""
        # step1: quantize rgb colourspace according to nc
        q_labels2d = self.RGB_quantize_model(template)
        
        # step2: compute vertical and horizontal CH's
        CH_v  = self.CH_vertical(q_labels2d) # compute vertical CH
        CH_h  = self.CH_horizontal(q_labels2d) # compute horizontal CH
        CH_pd = self.CH_pos_diagonal(q_labels2d) # compute positve diagonal CH
        CH_nd = self.CH_neg_diagonal(q_labels2d) # compute negative diagonal CH

        # step3: return overall hisgtogram
        return np.sum((CH_v, CH_h, CH_pd, CH_nd), axis=0)

    def CH_candidate(self, template):
        """computes overall CH along vertical and horizontal axes"""
        # step1: quantize rgb colourspace according to nc
        q_labels2d = self.RGB_quantize_candidate(template)
        
        # step2: compute vertical and horizontal CH's
        CH_v  = self.CH_vertical(q_labels2d) # compute vertical CH
        CH_h  = self.CH_horizontal(q_labels2d) # compute horizontal CH
        CH_pd = self.CH_pos_diagonal(q_labels2d) # compute positve diagonal CH
        CH_nd = self.CH_neg_diagonal(q_labels2d) # compute negative diagonal CH

        # step3: return overall CH
        return np.sum((CH_v, CH_h, CH_pd, CH_nd), axis=0)


    def intersection(self, CH_1, CH_2):
        """computes the intersection between two Cooccurance histrograms""" 
        I = 0 # intesection
        for k in range(0, self.nd-1):
            for i in range(0, self.nc):
                for j in range(0, self.nc):
                    #print("CH1, CH2, min: ",CH_1[k,i,j],  CH_2[k,i,j], np.minimum(CH_1[k,i,j], CH_2[k,i,j]))
                    I = I + np.minimum(CH_1[k,i,j], CH_2[k,i,j])
        return I # return the intersection of CH_1 and CH_2

    def setup(self, frame0, ROI):
        """setup the model"""
        self.h = frame0.shape[0] # frame height
        self.w = frame0.shape[1] # frame width

        self.y0 = ROI[0] # model y0
        self.x0 = ROI[1] # model x0
        self.hm = ROI[2] # model height
        self.wm = ROI[3] # model width
        template = frame0[self.y0:self.y0+self.hm, self.x0:self.x0+self.wm] # isolate target template
        self.q = self.CH_model(template) # compute target histogram, q from t1
 
    def search(self, frame, sw_height, sw_width, step_y, step_x):
        """performs course search for particular model CH in frame"""
        height, width = frame.shape[0], frame.shape[1] # frame dimensions
        
        I_mm = self.intersection(self.q, self.q) # compute I_mm
        I_best, y_best, x_best = 0, 0, 0
        print("I_mm: ", I_mm)        
        # iterate image
        for y in range(0, height-step_y-1, step_y):
            for x in range(0, width-step_x-1, step_x):
                candidate = frame[y:y+sw_height, x:x+sw_width] # candidate window 
                print(candidate.shape)
                p = self.CH_candidate(candidate) # compute candidate histogram p using model quantization
                I_cm = self.intersection(self.q, p) # compute candidate and model intersection

                print(I_cm)
                if(I_cm > (self.alpha*I_mm) and I_cm > I_best): # check quality of match aginst previous matches and threshold
                    print(I_cm)
                    I_best = I_cm # update best Intersection
                    y_best, x_best = y, x # update best coordinates
                    
        # return best matching candidate CH in frame and it's dimensions
        return y_best, x_best, sw_height, sw_width

    def detect(self, frame, sf=2):
        # obtain dimensions of frame, model and search window

        print(self.kmeans)
        print("coarse")
        # step1: Coarse Search
        sw_height, sw_width = sf*self.hm, sf*self.wm # get search window dimensions 
        step_y = sw_height//2 # y overlap 
        step_x = sw_width//2 # x overlap
        y0, x0, h0, w0 = self.search(frame, sw_height, sw_width, step_y, step_x) # returns best coordinates in coarse window
        
        print("fine")
        # fine search
        sub_frame = frame[y0:y0+h0, x0:x0+w0] # reduce search dims to best match of coarse search
        sw_height2, sw_width2 = self.hm, self.wm
        step_y2 = sw_height2//4
        step_x2 = sw_width2//4
        y1, x1, h1, w1 = self.search(sub_frame, sw_height2, sw_width2, step_y2, step_x2) # returns refined best coordinates
        
        print(x0, y0, x0+x1, y0+y1)
        # return two identified bounding boxes
        return (y0, x0), (y0+y1, x0+x1)


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
    self.nd = 12 # distances for CCH
    self.nc = 8 # colour quantization levels

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
        self.kmeans = KMeans(n_clusters=nc, random_state=0)
        q_labels = self.kmeans.fit_predict(np.expand_dims(rgb_euclidean.flatten(),axis=1))
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
            
        # Apply K-means algorithm to quantize pixels according to nc    
        q_labels = self.kmeans.predict(np.expand_dims(rgb_euclidean.flatten(),axis=1))
        candidate_labels = np.reshape(q_labels,(height,width)) # return the labels in 2D quantized dims
        return candidate_labels

    def CH_vertical(q_labels2d):
        """defines the vertical cooccurance histogram for a specific template according to quantized colourspace labels matrix and distances"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_y = np.zeros(shape=(nc,nc,nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1,nd):
            for y in range(0,height-k_range):
                for x in range(0,width):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y+k_range,x] # get destincation index for CH_y at (y+k,x)
                    CH_y[src_index,dst_index,k_range-1] = CH_y[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
                    
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_vertical = [np.sum((CH_y[:,:,k_range],CH_y[:,:,k_range].T), axis=0) for k_range in range(0,nd-1)]
        return CH_vertical

        
    def CH_horizontal(q_labels2d, nc, nd):
        """defines the horizontal cooccurance histogram for a specific template according to quantized colourspace labels matrix and distances"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_x = np.zeros(shape=(nc,nc,nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1,nd):
            for y in range(0,height):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_x[src_index,dst_index,k_range-1] = CH_x[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_horizontal = [np.sum((CH_x[:,:,k_range], CH_x[:,:,k_range].T), axis=0) for k_range in range(0,nd-1)]
        return CH_horizontal

    def CH_pos_diagonal(q_labels2d, nc, nd):
        """defines the positve diagonal cooccurance histogram for a specifica template according to quantized colourspace labels"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
        
        CH_pd = np.zeros(shape=(nc,nc,nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1,nd):
            for y in range(height-1, 0+k_range-1, -1):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y-k_range,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_pd[src_index,dst_index,k_range-1] = CH_pd[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_pos_diag = [np.sum((CH_pd[:,:,k_range], CH_pd[:,:,k_range].T), axis=0) for k_range in range(0,nd-1)]
        return CH_pos_diag

    def CH_neg_diagonal(q_labels2d, nc, nd):
        """defines the negative diagonal cooccurance histogram for a specific template according to quantized colourspace labels"""
        # get label dims
        height, width = q_labels2d.shape[0], q_labels2d.shape[1] # get dimensions
            
        CH_nd = np.zeros(shape=(nc,nc,nd-1)) # initialize zeros for CH histogram according to quantizations size
        for k_range in range(1,nd):
            for y in range(0,height-k_range):
                for x in range(0,width-k_range):
                    src_index = q_labels2d[y,x] # get source index for CH_y at (y,x)
                    dst_index = q_labels2d[y+k_range,x+k_range] # get destincation index for CH_y at (y+k,x)
                    CH_nd[src_index,dst_index,k_range-1] = CH_nd[src_index,dst_index,k_range-1] + 1 # update histogram count in bin (src_index,dst_index)
        
        # make the histogram symmetric by multiplying with transpose of unidirectional cooccurance histogram
        CH_neg_diag = [np.sum((CH_nd[:,:,k_range], CH_nd[:,:,k_range].T), axis=0) for k_range in range(0,nd-1)]
        return CH_neg_diag

    def CH_overall_model(template, nc, nd):
        """computes overall CH along vertical and horizontal axes"""
        # step1: quantize rgb colourspace according to nc
        kmeans, q_labels2d = RGB_quantize_model(template, nc)
        
        # step2: compute vertical and horizontal CH's
        CH_v  = CH_vertical(q_labels2d, nc, nd) # compute vertical CH
        CH_h  = CH_horizontal(q_labels2d, nc, nd) # compute horizontal CH
        CH_pd = CH_pos_diagonal(q_labels2d, nc, nd) # compute positve diagonal CH
        CH_nd = CH_neg_diagonal(q_labels2d, nc, nd) # compute negative diagonal CH
        return kmeans, np.sum((CH_v, CH_h, CH_pd, CH_nd), axis=0)


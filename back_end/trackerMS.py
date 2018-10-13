"""
    Claude Betz (BTZCLA001)
    tracker_MS.py

    Implementation of the mean shift tracking algorithm
"""

# Imports
import numpy as np
from back_end.tracker import tracker 

class trackerMS(tracker): # inherits from tracker
    # Instance Variables
    eps = 4 # step tolerance value
    m = 8 # number of bins

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

    # setters
    def set_eps(eps):
        """set the step size"""
        self.eps = eps

    def set_m(m):
        """set the bin size"""
        self.m = m

    # methods:
    def bhatt_coeff(self, q, p):
        """compute the battacharyya coefficient between two pdfs """
        height, width = q.shape[0], q.shape[1]
        batt = 0
        for y in range(0,height):
            for x in range(0,width):
                batt = batt + np.sqrt(q[y,x]*p[y,x])
        return batt  
            
    def elliptical_mask(self, height, width):
        """computes elliptical mask for particular template dimensions"""
        y0, x0 = height//2, width//2 # get center of kernel (treated as 0,0) - note these are also hy and hx 
        yy, xx = np.mgrid[:height, :width] # create layers for subsequent mask
        ellipse_eqn = ((yy-y0)/y0)**2 + ((xx-x0)/x0)**2 # ellipse equation
        return np.logical_and(ellipse_eqn<=1, ellipse_eqn<=1) # elliptical mask for kernel

    def euclidean_distance(self, v1, v2):
        """computes euclidean distance between two 2D vectors"""
        return np.sqrt((v2[0]-v1[0])**2+(v2[1]-v1[1])**2)
    
    def epanechnikov_eqn(self, x):
        """equation of epanechnikov kernel"""
        d = 2 #number of dimensions
        c_d = np.pi# area of a unit circle in d  
        return 1/2*(1/c_d)*(d+2)*(1-x) # |x|<=1 i.e. normalised

    def epanechnikov_kernel(self, template):
        """computes epanechnikov kernel values given distance from center of kernel"""    
        # get dimensions and generate mask
        height, width = (template.shape[0],template.shape[1])
        y0, x0 = (height//2, width//2) # get center of kernel (treated as 0,0) - note these are also hy and hx 
        elliptical_mask = self.elliptical_mask(height,width) 
        
        # get epanechnikov weights for the template dimensions
        kernel = np.zeros(shape=(height,width)) 
        for y in range(0,height):
            for x in range(0,width):
                if(elliptical_mask[y,x]==True):
                    y_s, x_s  = (y-y0)/y0, (x-x0)/x0 # compute normalized point 
                    norm = self.euclidean_distance((y_s,x_s),(0,0)) # get norm of normalized point (x_i*) i.e y0=hy, x0=hx
                    kernel[y,x] = self.epanechnikov_eqn(norm**2) # apply norm^2 to epanechnikov equation
        return kernel
    
    def histogram(self, template, m=8):
        """compute 2d colour histogram given template,bin number,m and kernel function"""
        # step1: limit pixels to elliptical region inscribed in rectange
        height, width = template.shape[0], template.shape[1]
        y0, x0 = (height//2, width//2) # get center of kernel (treated as 0,0) - note these are also hy and hx 
        elliptical_mask = self.elliptical_mask(height, width)
           
        # step2: fetch pixel values
        hist = np.zeros(shape=(256//self.m, 256//self.m)) # 2d-histogram to hold values 
        kernel = self.epanechnikov_kernel(template) # kernel and normalisation constant, C
        for y in range(0,height):
            for x in range(0,width): 
                if(elliptical_mask[y,x]==True): # only deal with points in the mask
                    u, v = template[y,x,0],template[y,x,1] # get the R and G components
                    index_r, index_g = u//self.m, v//self.m # used to index 2d-histogram
                    hist[index_r,index_g] = hist[index_r,index_g] + kernel[y,x] # add point to histogram with weight
        
        # normalize histogram
        C = np.sum(hist) # normalisation constant
        histogram = np.true_divide(hist, C)
        return histogram 

    def pixel_weights(self, template, q, p, m=8):
        """compute the weights necessary for the mean shift algorithm given target and candidate distributions q and p"""
        # step1: limit pixels to elliptical region inscribed in rectange
        height, width = template.shape[0],template.shape[1]
        y0, x0 = (height//2, width//2) # get centre 
        elliptical_mask = self.elliptical_mask(height, width)
        
        # step2: compute weights
        weights = np.zeros(shape=(height,width))
        for y in range(0,height):
            for x in range(0,width): 
                if(elliptical_mask[y,x]==True):  
                    u, v = template[y,x,0], template[y,x,1] # get colour index ,u with which we can index the histograms
                    index_r, index_g = u//self.m, v//self.m # used to index 2d-histogram
                    if(p[index_r,index_g]!=0):
                        weights[y,x] = np.sqrt(q[index_r,index_g]/p[index_r,index_g]) # compute weights based on equation
        return weights 

    def mean_shift(self, template, weights):
        """compute the mean shift vector of the template based on the weights"""
        # step1: limit pixels to elliptical region inscribed in rectange
        height, width = template.shape[0], template.shape[1]
        y0, x0 = (height//2, width//2) # get centre 
        elliptical_mask = self.elliptical_mask(height,width)

        # step2: estimate shift vector
        v_y,v_x = 0,0 # to hold estimated shift vector
        for y in range(0,height):
            for x in range(0,width): 
                if(elliptical_mask[y,x]==True):
                    v_y = v_y + weights[y,x]*((y-y0)) 
                    v_x = v_x + weights[y,x]*((x-x0))
        w_sum = np.sum(weights) # compute denominator of expression
        return int(v_y/w_sum),int(v_x/w_sum)

    def mean_shift_loop(self, frame, q, y0, x0, h, w):
        """perform mean shift to get new location"""
        t1 = frame[y0:y0+h, x0:x0+w] # candidate template
        p0 = self.histogram(t1) # candidate histogram

        for i in range(0,10): # mean shift loop 20 iterations
            weights = self.pixel_weights(t1, q, p0) # calculate weights
            vy, vx = self.mean_shift(t1, weights) # calculate mean shift vector
            y1, x1 = int(y0+vy), int(x0+vx) # y1

            step_size = self.euclidean_distance((vy,vx), (0,0))
            if(step_size < self.eps): # likely to converge so check last             
                return y1, x1

            else: # step not small enough use battacharyya coefficient to refine step
                tc = frame[y1:y1+h, x1:x1+w] # get ROI at x1
                p1 = self.histogram(tc) # compute p(x1)
                batt0 = self.bhatt_coeff(q, p0) # similarity measure between q(x0) and p(x0) 
                batt1 = self.bhatt_coeff(q, p1) # similiarty measure betweem q(x0) and p(x1)

                while(batt0>batt1):
                # so we don't step too far
                    vy, vx = vy//2, vx//2 # halve the step size
                    y1, x1 = int(y0+vy), int(x0+vx) # get new smaller distance 
                    tc = frame[y1:y1+h,x1:x1+w]
                    p1 = self.histogram(tc)
                    batt1 = self.bhatt_coeff(q, p1)
                    if(np.abs(vy)<=1 and np.abs(vx)<=1): # avoid infinite loop
                        break
                return y1, x1
        return y1, x1
    
    def setup(self, frame0, ROI):
        """initialsises tracker with template model"""
        self.y0 = ROI[0] # get dims (y,x,h,w) - 0,1,2,3
        self.x0 = ROI[1]
        self.h = ROI[2]
        self.w = ROI[3]
        t0 = frame0[self.y0:self.y0+self.h, self.x0:self.x0+self.w] # isolate target template
        self.q = self.histogram(t0) # compute target histogram, q from t1
 
    def track2(self):
        """mean shift algorithm applied to the video or image sequence"""
        # frame 1: 
        ret,frame0 = self.cap.read() # get first frame
        y0, x0, h, w = self.select_roi(frame0) # user select target template
        t0 = frame0[y0:y0+h, x0:x0+w] # isolate target template
        q = self.histogram(t0) # compute target histogram, q from t1
        while(1): # main loop
            ret,frame1 = self.cap.read() # get next frame n+1
            if(frame1 is None): # safety check
                break 
            y0, x0 = self.mean_shift_loop(frame1, q, y0, x0, h, w)
            
            # draw box at location and display
            self.draw_bounds(frame1,y0,x0,h,w)
            

            # write image to tmp.
            #self.image_write(frame1)

            # will be replace with write to file
            cv.imshow('mean_shift', frame1)
            cv.waitKey(0)

    def track(self, frame): # gets frame from front end
        """track for gui"""
        if(frame is not None): # safety check 
            y0, x0 = self.mean_shift_loop(frame, self.q, self.y0, self.x0, self.h, self.w) 
            self.y0 = y0
            self.x0 = x0
            return y0, x0     
        else:
            return -1, -1

if __name__=='__main__':
    track()

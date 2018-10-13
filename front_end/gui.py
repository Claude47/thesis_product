"""
    Claude Betz (BTZCLA001)
    app.py

    Implements the GUI of the MOT System
"""

#import sys
import os, re, os.path # directory management
import time # threads

# PyQt5 imports
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QWidgetAction, qApp, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStyle
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, pyqtSlot, pyqtSignal, QTimer 

# cv for reading
import cv2 as cv

# own classes
from front_end.imageview import imageView

# Backend imports 
from back_end.tracker_MS import tracker_MS # one of our API

class GUI(QMainWindow):
    # instance variables for loading a sequence
    seq_dir = "" # path to sequence folder
    seq_paths = [] # paths to all the files in sequence
    seq_length = 0 # length of sequence loaded
    cur_index = 0 # index for sequence paths
    cur_path = "" # current image path 
    cur_img = None # current image 
    
    # instance variables for playing 
    frame_delay = 0.1 # frame delay (seconds)
    seq_play = False # truth value to inform whether to play or not
    
    # instance variables for algorithms
    alg_running = False # set to true to render bounding box
    mst_delay = 0.01

    # writing 
    write_path = "" # path to directory to write to 

    def __init__(self, parent=None):
        """super constructor + app constructor"""
        super(GUI, self).__init__(parent) # Super constructor
      
        # Set window dimensions, title and icon
        self.setGeometry(50,50, 500, 500)
        self.setWindowTitle('Motion Tracking Application')
        self.setWindowIcon(QIcon('pythonlogo.png'))
                
        # Initialise menu
        self.menu() 

        # Initialise window components
        self.layout()

        # Initialise API
        self.API()

        # Render
        self.show()
             
    def menu(self):               
        """Menu Bar, Task Bar and Status Bar intitialisation"""     
        # ACTIONS        
        # 1. fileMenu Actions
        ## fileSelectAction: lets user select a file from a directory 
        fileSelectAction = QAction(QIcon('exit24.png'), 'File', self) # Exit Action Object
        fileSelectAction.setShortcut('Ctrl+O') # bind shortcut "Ctrl+Q" to Exit Button
        fileSelectAction.setStatusTip('Open File') # status bar exit message
        fileSelectAction.triggered.connect(self.openSequence) # connect QtGui quit() method 
        
        ## exitAction: exits the application
        exitAction = QAction(QIcon('exit24.png'), 'Exit', self) # Exit Action Object
        exitAction.setShortcut('Ctrl+Q') # bind shortcut "Ctrl+Q" to Exit Button
        exitAction.setStatusTip('Exit application') # status bar exit message
        exitAction.triggered.connect(self.close) # connect QtGui quit() method 
        
        # 2. algorithmMenu Actions 
        # Simple Template Matching
        simpleTemplateMatchingAction = QAction('Simple Template Matching', self)
        simpleTemplateMatchingAction.setShortcut('Ctrl+1')
        simpleTemplateMatchingAction.setStatusTip('Apply Simple Template Matching Algorithm')
        simpleTemplateMatchingAction.triggered.connect(self.simpleTemplateMatching)

        # Adaptive Template Matching
        adaptiveTemplateMatchingAction = QAction('Adaptive Template Matching', self)
        adaptiveTemplateMatchingAction.setShortcut('Ctrl+2')
        adaptiveTemplateMatchingAction.setStatusTip('Apply Adaptive Template Matching Algorithm')
        adaptiveTemplateMatchingAction.triggered.connect(self.adaptiveTemplateMatching)

        # Mean Shift
        meanShiftTrackingAction = QAction('Mean Shift Tracking', self)
        meanShiftTrackingAction.setShortcut('Ctrl+3')
        meanShiftTrackingAction.setStatusTip('Apply Mean Shift Tracking Algorithm')
        meanShiftTrackingAction.triggered.connect(self.meanShiftTracking)


        # *** menuBar ***
        menuBar = self.menuBar() 

        ## 1. fileMenu - drop down buttons
        fileMenu = menuBar.addMenu('&File') # Add File option to Menu Bar
        fileMenu.addAction(fileSelectAction) # Add fileSelectAction to fileMenu 
        fileMenu.addAction(exitAction) # Add exitAction to fileMenu
        
        ## 2. algorithmMenu - select algorithms
        algorithmMenu = menuBar.addMenu('&Algorithms')
        algorithmMenu.addAction(simpleTemplateMatchingAction) # Add simpleTemplateMatchingAction to algorithmMenu
        algorithmMenu.addAction(adaptiveTemplateMatchingAction) # Add adaptiveTemplateMatchingAction to algorithmMenu
        algorithmMenu.addAction(meanShiftTrackingAction) # Add meanShiftTracking to algorithmMenu
        
        # *** statusBar ***
        self.statusBar()
        self.statusBar().showMessage('Status: Ready to Track')

        # *** Tool Bar ***
        toolBar = self.addToolBar('Exit') # Add Exit icon to Tool Bar  
        
        toolBar.addAction(fileSelectAction) # Add File Select Object to Tool Bar
        toolBar.addAction(exitAction) # Add Exit Action Object to Tool Bar

    def layout(self):
        """Builds the application area where backend images displayed"""
        # screen
        self.centralWidget = QWidget() # central widget for screen
        self.setCentralWidget(self.centralWidget)

        # main screen layout
        self.screenLayout = QHBoxLayout(self.centralWidget) # main screen

        self.rightWidget = QWidget() # widget for right hand side
        self.rightLayout = QVBoxLayout(self.rightWidget) # right part of screen

        # buttons for our screen
        self.buttonsWidget = QWidget() # buttons in below right part
        self.buttonsWidgetLayout = QHBoxLayout(self.buttonsWidget)
        
        # create buttons for buttonWidget
        controls = ['','', '', '', '','']
        self.buttons = [QPushButton(b) for b in controls]
        for button in self.buttons:
            self.buttonsWidgetLayout.addWidget(button)
        
        # Button Actions
        # 0. First
        self.buttons[0].setToolTip('Click to go to Start of Sequence')
        self.buttons[0].clicked.connect(self.startImage) # action for 'start' button
        self.buttons[0].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaSkipBackward))

        # 1. Prev
        self.buttons[1].setToolTip('Click to go to Previous Frame')
        self.buttons[1].clicked.connect(self.prevImage)
        self.buttons[1].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaSeekBackward))

        # 2. Pause
        self.buttons[2].setToolTip('Click to Pause Sequence')
        self.buttons[2].clicked.connect(self.pauseSequence)
        self.buttons[2].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPause))

        # 3. Play
        self.buttons[3].setToolTip('Click to Play Sequence')
        self.buttons[3].clicked.connect(self.playSequence)
        self.buttons[3].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))

        # 4. Next
        self.buttons[4].setToolTip('Click to go to Next Frame')
        self.buttons[4].clicked.connect(self.nextImage) # action for 'next' button
        self.buttons[4].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaSeekForward))

        # 5. End
        self.buttons[5].setToolTip('Click to go to end of Sequence')
        self.buttons[5].clicked.connect(self.endImage) # action for 'end' button
        self.buttons[5].setIcon(QApplication.style().standardIcon(QStyle.SP_MediaSkipForward))

        # where to display our Main Image
        self.imageLabel = imageView()
        self.imageLabel.setMinimumWidth(480)
        self.imageLabel.setMinimumHeight(500)
        self.imageLabel.setStyleSheet('* {background: gray;}')
        self.imageLabel.setAlignment(Qt.AlignCenter)

        # where to display templates and buttons
        self.templateLabel = imageView()
        self.templateLabel.setMinimumWidth(240)
        self.templateLabel.setMinimumHeight(250)
        self.templateLabel.setStyleSheet('* {background: black;}')
        self.templateLabel.setAlignment(Qt.AlignCenter)
        # add widgets to layout
        self.screenLayout.addWidget(self.imageLabel)
        self.rightLayout.addWidget(self.templateLabel)
        self.rightLayout.addWidget(self.buttonsWidget)
        self.screenLayout.addWidget(self.rightWidget)
        
    def API(self):
        """initialised our API"""
        self.MST = tracker_MS(self.seq_dir+"00000001.jpg") # initialise tracker_MS object
                
    # function for sequence Input and Output
    @pyqtSlot()
    def openSequence(self):
        """function to point to a folder with a desired sequence"""
        self.seq_dir = QFileDialog.getExistingDirectory(self,'Open File') # store path to sequence
        self.readSequence(self.seq_dir) # read in file names in sequence into self.paths
        
    def readSequence(self, seq_dir):
        """read in sequence"""
        for path in sorted(os.listdir(seq_dir)):
            full_path = os.path.join(seq_dir, path)
            if os.path.isfile(full_path):
                self.seq_paths.append(full_path)
        
        self.seq_length = len(self.seq_paths) # update sequence length
        self.cur_index = 0 # reset image index
        self.cur_path = self.seq_paths[self.cur_index] # set cur image to first image in self.seq_paths
        self.loadImage() # load in first image
     
    @pyqtSlot()
    def loadImage(self):
        """function to display using filepath"""
        self.cur_img = cv.imread(self.cur_path) # use cv to load in self.cur_img
        qformat = QImage.Format_Indexed8 # define format of image

        if(len(self.cur_img.shape)==3): # format is [0,1,2] [rows,cols,components]
            if(self.cur_img.shape[2]==4): # check dimensions
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.cur_img, self.cur_img.shape[1], self.cur_img.shape[0], self.cur_img.strides[0], qformat)
        img = img.rgbSwapped() # convert bgr to rgb

        self.cur_pixmap = QPixmap.fromImage(img) # update our internal current pixmap
        self.renderImage() # render

    def renderImage(self):
        """renders the image to the QLabel"""
        if(self.alg_running == True):
            self.drawBounds() # draw bounds if we are in algorithm
            self.renderTemplate() # load template into widget
        self.imageLabel.setPixmap(self.cur_pixmap) # populate label
        self.imageLabel.setMinimumHeight(self.cur_img.shape[0])
        self.imageLabel.setMinimumWidth(self.cur_img.shape[1])
        
    def renderTemplate(self):
        """renders template into templateQLabel when algorithm running"""
        # extract template from cur_pixmap
        template = QRect(self.x0, self.y0, self.w+1, self.h+1)
        self.tmp_pixmap = self.cur_pixmap.copy(template)

        self.templateLabel.setPixmap(self.tmp_pixmap) # populate label
        self.templateLabel.setMinimumHeight(self.h)
        self.templateLabel.setMinimumWidth(self.w)
        

    def updateImage(self):
        """updates the image being displayed in self.imageLabel"""
        self.loadImage() # load updated image

    # functions to navigate sequences
    @pyqtSlot()
    def startImage(self):
        """function to go to start of loaded sequence"""
        self.cur_index = 0
        self.cur_path = self.seq_paths[self.cur_index] # update internal current image path
        self.updateImage() # update our imageLabel

    @pyqtSlot()
    def endImage(self):
        """function to go to end of loaded sequence"""
        self.cur_index = self.seq_length - 1
        self.cur_path = self.seq_paths[self.cur_index] # update internal current image path
        self.updateImage() # update our imageLabel

    @pyqtSlot()
    def nextImage(self):
        """button for moving to next image in self.seq_dir"""
        if(self.cur_index < self.seq_length - 1):
            self.cur_index = self.cur_index + 1 # increment to next img
        else: # wrap around
            self.cur_index = 0

        self.cur_path = self.seq_paths[self.cur_index] # update internal current image path
        self.updateImage() # update our imageLabel

    @pyqtSlot()
    def prevImage(self):
        """button for movin to previous image in self.seq_dir"""
        if(self.cur_index > 0):
            self.cur_index = self.cur_index - 1 # decrement to prev image
        else: # wrap around
            self.cur_index = self.seq_length - 1
        
        self.cur_path = self.seq_paths[self.cur_index] # update internal current image path
        self.updateImage()

    def play(self):
        """gets next image while """
        if(self.seq_play==True):
            self.nextImage() # get next Image 
        else:
            self.seq_timer.stop() # stop playing
    
    @pyqtSlot()
    def pauseSequence(self):
        """pauses loaded sequence for manual iteration"""
        self.seq_play = False # don't play 

    @pyqtSlot()
    def playSequence(self):
        """plays loaded Sequence at specified playback rate""" 
        self.seq_play = True # play sequence
        self.seq_timer = QTimer() # timere for frame rate
        self.seq_timer.timeout.connect(self.play) # connect timeouts to fetching next image
        self.seq_timer.start(int(self.frame_delay * 1000)) # set timer countdown rate

    def drawBounds(self):
        """draws bounds on the appropriate object according to backend coordinates"""
        painter = QPainter(self.cur_pixmap)
        pen = QPen(Qt.yellow)
        painter.setPen(pen)
        painter.drawRect(self.x0, self.y0, self.w, self.h)

    # BACKEND functions 
    def algorithm_reset():
        """reset parameters after an algorithm has been run"""
        self.alg_running = False # set flag false again
        

    def simpleTemplateMatching(self):
        """call to external simpleTemplateMatching"""
        self.statusBar().showMessage('Status: Tracking (Simple Template)')

    def adaptiveTemplateMatching(self):
        """call to external adaptiveTemplateMatching"""    
        self.statusBar().showMessage('Status: Tracking (Adaptive Template)')

    @pyqtSlot()
    def meanShiftTracking(self):
        self.statusBar().showMessage('Status: Tracking (Mean Shift)')


        self.meanShiftTrackingInit()
        self.MST_timer = QTimer() # timer for frame rate
        self.MST_timer.timeout.connect(self.meanShiftTrackingLoop) # connect timeouts to fetching next image
        self.MST_timer.start(int(self.mst_delay * 1000)) # set timer countdown rate

    def meanShiftTrackingInit(self):
        """initialise meanShiftTracking"""
        self.cur_index = 0 # start at beginning of sequence
        self.alg_running = True # tell system and Algorithm is running
        self.MST.setup(self.cur_img, self.imageLabel.currentQRect) # setup mean shift tracker with coords
        self.y0 = self.imageLabel.currentQRect[0]
        self.x0 = self.imageLabel.currentQRect[1]
        self.h = self.imageLabel.currentQRect[2]
        self.w = self.imageLabel.currentQRect[3]
        self.nextImage() # load frame 1

    def meanShiftTrackingLoop(self):
        """implement loop"""   
        if(self.cur_index<self.seq_length-1):
            coords = self.MST.track2(self.cur_img) # track    
            print("coords: ", coords)
            self.y0 = coords[0]
            self.x0 = coords[1]
            self.nextImage() # load image    
        else:
            self.MST_timer.stop() # terminate algorithm
            self.algorithm_reset() # cleanup and reset


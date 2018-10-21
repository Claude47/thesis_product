# Motion Tracker (MT) System 
## Claude Betz (BTZCLA001)
### University of Cape Town
### Departiment of Electrical Engineering

## Acknowledgements
The icons used within the pyQt5 GUI are made by Freepik from www.flaticon.com

## Background to project
This project constitutes the implementation accompanying my final year thesis. 
It is a study into the field of motion tracking, specifically the class of
kernel based motion tracker.

## Motivation
The MT System developed loosely draws inspiration from sport analysis segments, where
analysts highlight the movement of a particular player to highlight certain
actions or plays. 
In line with this, the MT System implements a kernel based tracking approaches
that rely on user detection object in a chosen image seqeuence which is then subsequently
tracked for the remainder of the the particular sequence.
Interaction with the system is facilitated by a front-end GUI.

## Tech Framework
The language used for this project is the Python3 programming language. The
pip3 package manager was used to obtain all the relevant libraries that are
subsequently listed.

The code was developed in an isolated virtual environment using the virtualenv
and virtualenvwrapper packages

### Dependencies
- NumPy - used by the colour co-occurrence histogram detector (CCHD) and the mean shift tracker (MST) implementations.
- PyQt - (used if the GUI to be run)
- OpenCV - used by the template matching trackers.
- scikit-learn - used by the CCHD for the quantization of the CCH descriptor

The Installation section goes over a simple means to setup these dependencies

## Features 
In addition to facilitating user selection of image sequences, and selection of
regions of objects to track. The MT System allows for the setting of different algorithm parameters and
provides visual feeback to a particular user

## Installation
The project requires Python version 3.6, and pip3 python package manager.
It is recommended that a user have virtualenv or virtualenv wrapper installed on
their machine so that there are no clashes with the global python environment when installing the requisite dependencies for the MT System.
virtualenv installation instructions can be found at the following url: 
https://python-docs.readthedocs.io/en/latest/dev/virtualenvs.html

virtualenvwrapper installation instructions can be found by following this url:
https://virtualenvwrapper.readthedocs.io/en/latest/install.html

Once either virtualenv or virtualenvwrapper are installed, navigate to a
directory of your choice and clone the repository via the following command.

```bash
git clone git@github.com:Claude47/thesis_product.git
```

Create a new virtual environment within which the dependices for the MT System
can be installed in isolation. The instructions for doing this should be on the
previously mentioned installation pages.

Once the relevant virtualenv or virtualenvwrapper is activated, the following
command will download all the relevant dependencies necessary to run the MT
System or to use the back end tracker APIs

```bash
pip3 install -r requirements.txt
```

## API Reference
In the case that the back-end tracker algorithms are to be integrated into a
seperate application or program. The relevant APIs are easy to use.

The detector or tracker object exposes two APIs to a user. The first method is 
```python
    setup(frame0, ROI)
```

This method initialises the tracker with the target model. It takes two parameters for this, the initial frame of interest and a user selected region of
interest (ROI) containing the object to be tracked. 

```python
    track(frame)
```
This method takes as an argument the frame of interest in particular image
sequence for in which the target model should be tracked. This function is
called for each subsequent frame in a particular sequence.

The APIs is standardised across all the implemented detector and tracker
classes. 

In addition these two APIs. There are various setters for the different algorithm parameters of the various implementatons that are
defined at the top of the class definitions of the various trackers.




## How to use API
Import the detector or tracker via a simple import from their respective
modules, this is shown below
```python
from trackerMS import trackerMS # import tracker from it's module
```

A user is then free to create multiple detector or tracker instances
An example of this is shown below.
```python
MST = trackerMS() # create tracker object
```

Two usage patterns of the API are outlined below
Assuming a user, wants to handle their own I/O as is the case with the
integrated MS System which has the PyQt5 GUI handle IO the following code sample is relevant.
The sample assumes that:
    next_frame() - avails the next frame in a sequence starting from frame 0 
    select_roi() - allows user selection of a ROI in said
that initial frame. 

```python
from trackerMS import trackerMS

def main():
    """track sequence"""
    MST = trackerMS() # create tracker object
    frame0 = next_frame() # get initial frame
    template = select_roi() # select region of interest

    MST.setup(frame0,template) # initialise tracker
    
    # loop through sequence
    track = [] # array to hold MST results
    frame = next_frame()
    while(frame is not None):
        coords = MST.track(frame) # track for frame
        track.append(coords) # add coordinates to track
        frame = next_frame() # get next frame in sequence 
```

Alternatively using the OpenCV library for I/O, The MST tracker has can be used
in an entriely self contained manner. A sample usage is displayed below, in this
scenario, the MST handles IO internally only outputting processed frames with
bounding boxes around the selected target.

```python
from trackerMS import trackerMS

seq_path = "bird/00001.jpg" # path to first image in sequence

def main():
    """track sequence"""
    MST = trackerMS(seq_path) # create tracker object
    MST.track2() # alternate track function reliant of OpenCV.
```

# Using the Graphical Tool
Provided the required dependencies are met, the GUI can be launched using the
the following command within the project root directory.

```bash
python MTS.py
```


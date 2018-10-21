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

## Installation
It is recommended that a user have virtualenv or virtualenv wrapper installed on
their machine so that there are no dependency clashes in installing the
requisite dependencies.

## API Reference
In the case that the back-end tracker algorithms are to be integrated into a
seperate application or program. The relevant APIs are easy to use.

Import the detector or tracker via a simple import from their respective
modules.
An example is shown below.
```python
    from trackerMS import trackerMS 
```

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

## How to use GUI
Provided the required dependencies are met, the GUI can be launched using the
the following command within the project root directory.

```bash
python MTS.py
```



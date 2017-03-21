#!/usr/bin/python
import io
import random
import picamera
import picamera.array
import numpy as np
import os
import datetime
import time
import sys
import png

from parameters import *
from PIL import Image

# -----------------------------------------------------------------------------
# --- Customization options
# -----------------------------------------------------------------------------


 hypothesis = None           # Background image used for motion detection
# -----------------------------------------------------------------------------
# --- Filepath information
# -----------------------------------------------------------------------------
# Find the full path of this python script
mypath=os.path.abspath(__file__)

# get the path location only (excluding script name)
baseDir=mypath[0:mypath.rfind("/")+1]
baseFileName=mypath[mypath.rfind("/")+1:mypath.rfind(".")]
progName = os.path.basename(__file__)

motionPath= baseDir + motionDir

# -----------------------------------------------------------------------------
# --- Make sure that the output directory exists
# -----------------------------------------------------------------------------
def checkImagePath():
   # Checks for image folders and creates them if they do not already exist.
   if not os.path.isdir(motionPath):
       msgStr = "Creating Image Motion Detection Storage Folder" + motionPath
       showMessage ("checkImagePath", msgStr)
       os.makedirs(motionPath)
   return

# -----------------------------------------------------------------------------
# --- Used for filename generation and debugging
# --- Returns a string representing the current date and time
# -----------------------------------------------------------------------------
def showTime():
    rightNow = datetime.datetime.now()
    currentTime = "%04d-%02d-%02d_%02d-%02d-%02d" % (
            rightNow.year,
            rightNow.month,
            rightNow.day,
            rightNow.hour,
            rightNow.minute,
            rightNow.second)
    return currentTime

# -----------------------------------------------------------------------------
# --- Displays messages when in verbose mode
# -----------------------------------------------------------------------------
def showMessage(functionName, messageStr, alwaysShow=False):
    if verbose or alwaysShow:
        now = showTime()
        print ("%s %s - %s " % (now, functionName, messageStr))
    return

# -----------------------------------------------------------------------------
# --- Shows ... to let you know program is running
# -----------------------------------------------------------------------------
def showAnotherDot(endLine):
    if not endLine:
        sys.stdout.write('.')
        sys.stdout.flush()
    else:
        print("")



def getRGBPicture(camera):

    with picamera.array.PiRGBArray(camera, size=testSize) as output:
        camera.capture(output, 'rgb', resize=testSize)
        pic = output.array
    return pic
# -----------------------------------------------------------------------------
# --- Motion Detection Algorithm
# --- Compares previous picture to current. Returns true if motion is found
# -----------------------------------------------------------------------------
def detectMotion(camera):
    global hypothesis
    global bgChangeRate
    global threshold
    global sensitivity
    global dayMode
    #stream = io.BytesIO()

    #take a picture to compare to background
    currentImage = getRGBPicture(camera)

    # if there is no background, set this as background
    if hypothesis is None:
        hypothesis = currentImage
        result = False
        return result
    # get avg pixel value, if below dusk, then switch to night mode
    avgPixelVal = np.sum(currentImage)/(np.prod(testSize)*3)

    # Compare currentImage to hypothesis to detect motion. This is
    # get the absolute difference between ba/ckground and new picture for
    # each pixel, then multiply the change in rgb values for each pixel
    # together (Ex. change in rgb = (2, 20, 40), then pixel change = 1600)
    squaredErrors = np.power((hypothesis-currentImage),2)
    costFuncJ = np.sum(squaredErrors, axis=2)
    #costFuncJ = np.prod(np.ceil((np.absolute(hypothesis-currentImage))),axis=2)

    # get the number of pixels with a greater change than min threshold
    pixChanges = np.sum(costFuncJ > threshold)

    # if enough pixels have changed, then motion has been detected
    if pixChanges > sensitivity:
        result = True
    else:
        result = False
    # allow background to adjust slightly towards the newest pixel values
    # this allows for objects in the view to be moved, and for the
    # background to gradually adjust for changes in lighting. If an object
    # is moved, the background image will slowly adjust after
    # it remains staionary for some time. Since this adjustment is
    # divided by bgChangeRate, higher values will slow down adjustments
    hypothesis = hypothesis - alpha * (hypothesis - currentImage)

    # create message for verbose mode
    message = "pixChanges = %i, avg(diff) = %i avg(rgbVal) = %i" % (
            pixChanges, np.sum(costFuncJ)/np.prod(testSize).astype(int), avgPixelVal)
    showMessage("detectMotion", message)

    #return whether or not motion has been detected
    return result

# -----------------------------------------------------------------------------
# --- Record video to file after it has been captured
# --- video should begin ~5 seconds before motion occurs and continue
# --- until after motion has stopped
# -----------------------------------------------------------------------------
def writeVideo(stream1, stream2, filename):
    # Write the entire content of the circular buffer (before motion) to disk
    with stream1.lock:
        with io.open(filename, 'wb') as output:
            for frame in stream1.frames:
                if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                    stream1.seek(frame.position)
                    break
            while True:
                buf = stream1.read1()
                if not buf:
                    break
                output.write(buf)
            # write the second buffer (after motion) to disk
            stream2.seek(0)
            output.write(stream2.read())
        # Wipe the streams once we're done
        stream1.seek(0)
        stream1.clear()
        stream2.seek(0)
        stream2.truncate()

# -----------------------------------------------------------------------------
# --- Get the file path and name to save the recording
# -----------------------------------------------------------------------------
def getVideoName(path):
    # build image file names by number sequence or date/time
    global cameraName
    rightNow = showTime()
    filename = "%s/%s_%s.h264" % ( path, rightNow, cameraName)
    return filename

# -----------------------------------------------------------------------------
# --- Main program, contains logic for operating camera continuously
# -----------------------------------------------------------------------------
def main():
    global dayMode
    global motionPath
    #-- works for local paths, not sure about paths on server
    checkImagePath()
    with picamera.PiCamera() as camera:
        #set up initial camera settings
        camera.resolution = frameSize
        stream1 = picamera.PiCameraCircularIO(camera, seconds=5)
        camera.video_stabilization = stabilize
        if not dayMode:
            camera.exposure_mode = 'night'
            camera.iso = nightIso
            time.sleep(3)
        #start recording to the buffer (so we can roll back to before motion)
        camera.start_recording(stream1, format='h264')
        showMessage("main", "checking for motion")
        # INFINITE LOOP --- to exit, hit ctrl-c
        try:
            while True:
                # check for motion once per second
                #camera.wait_recording(1)
                #testing
                testpic = getRGBPicture(camera)

                motionDetected = detectMotion(camera)
                showAnotherDot(motionDetected)
                # motion has been detected, record until it stops
                if motionDetected:
                    showMessage('detectMotion','Motion detected!', True)
                    # As soon as we detect motion, split the recording to
                    # record the frames "after" motion
                    stream2 = io.BytesIO()
                    camera.split_recording(stream2)
                    filename = getVideoName(motionPath)
                    # Wait until motion is no longer detected, then split
                    # recording back to the in-memory circular buffer
                    while detectMotion(camera):
                        camera.wait_recording(1)
                    # -- record an extra 2 seconds after motion stops
                    camera.wait_recording(2)
                    # print message for verbose mode
                    showMessage('detectMotion','Motion Stopped')
                    camera.split_recording('/dev/null')
                    # write both streams to disk
                    writeVideo(stream1, stream2, filename)
                    camera.split_recording(stream1)
                    # start over
                    showMessage("main", "checking for motion")

        finally:
            camera.stop_recording()
    return

# -----------------------------------------------------------------------------
# --- Calls the main program when run from this file
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

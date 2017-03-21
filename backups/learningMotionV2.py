#!/usr/bin/python

import numpy as np
import scipy.stats as stats
import os
import io
import sys
import picamera
import picamera.array

import datetime
import time
import math


motionDir = "motion"            #dirctory to store motion files
cameraName = "cam1"             #name of camera doing the recording
verbose = True                  #print extra output
fullFrameSize = (1640,1232)     #frame size of video recordings
testSize = (192,128)            #frame size of test images for detecting motion
alpha = 0.01                    #background learning rate
alphaInit = .02                 #learning rate for initialization
iterations = 256                #number of training examples to gather
sleepTime = 1.0                 #time to sleep between test photos
standard = 32                   #simplify setting window width, height, and step

windowWidth = standard
windowHeight = standard
step = standard

movementBuffer = 2              #makes movement detection less sensitive
framesToStopMotion = 10         #consecutive frames w/out motion to stop video

epsilon = .02                   #maximum allowable probability

#unicode symbols
motionUni = u'\U0001F6B6'
stopUni = u'\U0001F6B7'
moonPhases = [u'\U0001F311'
              ,u'\U0001F312'
              ,u'\U0001F313'
              ,u'\U0001F314'
              ,u'\U0001F315'
              ,u'\U0001F316'
              ,u'\U0001F317'
              ,u'\U0001F318']
moonPhase = 0

#holds and stores information for detecting motion
class Motion:
    # -----------------------------------------------------------------
    # --- take initial background pic and calculate number of features
    # --- required for given window width, height, and step
    # -----------------------------------------------------------------
    def __init__(self, camera):
        self.camera = camera
        self.hypothesis = self.takeTestPicture()
        self.featureCount = self.getFeatureCount()
        self.X_Train = np.matrix

    # -----------------------------------------------------------------
    # --- Take a picture with the camera to detect motion
    # -----------------------------------------------------------------
    def takeTestPicture(self):
        with picamera.array.PiRGBArray(self.camera, size = testSize) as output:
            self.camera.capture(output, 'rgb', resize=testSize, use_video_port=True)
            return output.array

    # -----------------------------------------------------------------
    # --- calculate feature vector xi for a given picture
    # -----------------------------------------------------------------
    def calculateFeatures(self, picture):
        #squared error of each pixel in new picture
        squaredErrors = np.sum(np.power((self.hypothesis - picture),2), axis=2)
        xOffset = 0
        yOffset = 0
        xi = np.zeros(self.featureCount)
        j = 0
        #calculate the sum of squaredErrors in each window
        while xOffset <= testSize[1] - windowWidth:
            while yOffset <= testSize[0] - windowHeight:
                info = np.sum((squaredErrors[xOffset : (xOffset + windowWidth)
                                             , yOffset: (yOffset + windowHeight)]))

                #store the log(sum(squaredErrorsInWindow)) as feature xi[j]
                xi[j] = np.log(info) if (info != 0) else 0
                j += 1;
                yOffset += step
            yOffset = 0
            xOffset += step
        #print(xi)
        return xi


    # -----------------------------------------------------------------
    # ---calculate the number of features (number of windows)
    # -----------------------------------------------------------------
    def getFeatureCount(self):
        count = 0
        xOffset = 0
        yOffset = 0
        while xOffset <= testSize[1] - windowWidth:
            while yOffset <= testSize[0] - windowHeight:
                count += 1
                yOffset += step
            yOffset = 0
            xOffset += step
        return count


    # -----------------------------------------------------------------
    # --- Gather m training examples and store in matrix X_Train
    # -----------------------------------------------------------------
    def gatherTrainingData(self, m):

        #First use gradient descent to get a good background image
        printMessage("Initializing Background using Stochastic Gradient Decent ")
        self.miniBatchUpdate(m)

        # gather up a bunch of pictures and store them in X_Train
        print "Background Image Training Complete " + u'\u2713' + '           '
        printMessage("Gathering Training Examples for Anomaly Detection")
        i = 0

        self.X_Train = np.matrix(np.zeros((m, self.featureCount)))

        #gather m training examples
        while i < m:
            #reduce processor load and increase variance in photos
            time.sleep(sleepTime)
            pic = self.takeTestPicture()
            self.X_Train[i] = self.calculateFeatures(pic)
            printSameLine("Training Example: " + str(i) + " of " + str(m))
            self.stochasticUpdate(pic, alpha)
            i += 1

        print "Gathering Training Samples Complete " + u'\u2713'

    # -----------------------------------------------------------------
    # --- update the background image using stochastic Gradient Descent
    # -----------------------------------------------------------------
    def stochasticUpdate(self, picture, alpha):
        self.hypothesis = self.hypothesis - alpha * (self.hypothesis - picture)

    # -----------------------------------------------------------------
    # --- do a series of SGD updates, used to initialize bg quickly
    # -----------------------------------------------------------------
    def miniBatchUpdate(self,m):
        i = 0
        while i < m:
            newPic = self.takeTestPicture()
            self.stochasticUpdate(newPic, alphaInit)
            printSameLine( "Background error: "
                          + str(np.sum(self.calculateFeatures(newPic))))
            i += 1


    #verified with np.var
    # -----------------------------------------------------------------
    # --- calculate the variance of X_train based on average, mu
    # -----------------------------------------------------------------
    def calculateSigma2(self,mu):
        printMessage("calculating sigma2 (Variance)")
        m = self.X_Train.shape[0]
        sigma2 = ( 1/(float)(m) * np.sum(np.power((self.X_Train - mu), 2),0) )
        print(sigma2)
        return sigma2



    #verified with np.average
    # -----------------------------------------------------------------
    # --- Calculate mu for each feature (each window)
    # -----------------------------------------------------------------
    def calculateMu(self):

        mu = np.array(1/((float)(self.X_Train.shape[0]))
                      * np.sum(self.X_Train,0))
        printMessage("calculating Mu (Avarage)")
        print(mu)
        mu = mu.reshape(-1)

        return mu



    # -----------------------------------------------------------------
    # --- Calculate the probablility that xi is not an anomaly
    # --- MathSpeak: returns 1 - cumulative density function of a
    # --- Gaussian distribution(mu,sigma) (sigma=stddev, sigma2=var)
    # -----------------------------------------------------------------
    def probability(self,xi,mu,sigma2):
        sigma2 = np.array(sigma2).flatten()
        statsn = stats.norm(mu,np.sqrt(sigma2)).sf(xi)
        statsn = np.prod(statsn)
        #multiply probabilities of each variable together
        np.prod((1/(2*math.pi )))
        return statsn

    # -------------------------------------------------------------------------
    # --- Convenience method for taking a picture and calculating probablility
    # --- of being an anomaly
    # -------------------------------------------------------------------------
    def checkForMotion(self,mu,sigma2):
        pic = self.takeTestPicture()
        xi = self.calculateFeatures(pic)
        prob = self.probability(xi,mu,sigma2)
        self.stochasticUpdate(pic,alpha)
        return prob



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
# --- Put a moon face at the beginning of same line output
# -----------------------------------------------------------------------------
def getPhase():
    global moonPhase
    moonPhase = (moonPhase + 1) % 8
    return moonPhases[moonPhase]


# -----------------------------------------------------------------------------
# --- Print and put cursor back at the beginning of the line
# -----------------------------------------------------------------------------
def printSameLine(string):
    phase = getPhase()
    sys.stdout.write("  %s   %s\r" % (phase, string))
    sys.stdout.flush()


# -----------------------------------------------------------------
# --- Print a big fancy message that gets peoples attention
# -----------------------------------------------------------------
def printMessage(message, top=True, bottom=True):
    if top == True:
        print("")
        print("#############################################################")
    print("## " + message)
    if bottom == True:
        print("#############################################################")


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


# -----------------------------------------------------------------
# --- main script, runs infinite loop checking for motion
# -----------------------------------------------------------------
def main():

    checkImagePath()
    with picamera.PiCamera() as camera:
        #set up initial camera settings
        camera.resolution = fullFrameSize
        camera.exposure_mode = 'auto'
        camera.video_stabilization = True
        stream1 = picamera.PiCameraCircularIO(camera, seconds=5)
        #camera.start_recording(stream1, format='h264')
        try:

            camera.start_recording(stream1, format='h264')
            motion = Motion(camera)
            camera.iso = 300
            time.sleep(2)
            camera.exposure_mode = "off"

            g = camera.awb_gains
            camera.awb_mode = 'off'
            camera.awb_gains = g
            motion.gatherTrainingData(iterations)
            i = 0

            mu = motion.calculateMu()
            sigma2 = motion.calculateSigma2(mu)

            #movementBuffer increases the average changes of each section
            mu += movementBuffer
            printMessage("Checking For Movement")

            motionStopCount = 0
            motionInProgress = False
            while True:
                time.sleep(sleepTime)
                prob = motion.checkForMotion(mu,sigma2)
                probabilityString = "Probability of Movement: " + str(1 - prob) + "              "

                #motion detected
                if (prob < epsilon):
                    #motion just started
                    if motionInProgress == False:
                        motionInProgress = True
                        printMessage("Motion Detected! " + showTime() + motionUni, bottom=False)
                        stream2 = io.BytesIO()
                        camera.split_recording(stream2)
                        filename = getVideoName(motionPath)

                    #everytime motion is detected, restart motionStopCount
                    motionStopCount = 0

                #motion is in progress but no longer being detected
                elif motionInProgress == True:
                    motionStopCount += 1
                    # we've gone framesToStopMotion w/out detecting motion
                    if motionStopCount > framesToStopMotion:
                        motionInProgress = False
                        printMessage("Motion Stopped   " + showTime() + stopUni, top=False)
                        camera.split_recording('/dev/null')
                        printMessage('Saving to ' + filename)
                        writeVideo(stream1,stream2,filename)
                        camera.split_recording(stream1)
                printSameLine(probabilityString)

        finally:
            
            camera.stop_recording()
            print('')
            print('ending')

    return





# -----------------------------------------------------------------------------
# --- Calls the main program when run from this file
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

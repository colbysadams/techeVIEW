#!/usr/bin/python

import numpy as np
import scipy.stats as stats
import sys
import picamera
import picamera.array
import time
import math


fullFrameSize = (1664,1232)
testSize = (192,128)
alpha = 0.02
iterations = 64

standard = 32

windowWidth = standard
windowHeight = standard
step = standard


#anomaly threshold
epsilon = .01

class Motion:



    def __init__(self, camera, delay):
        self.camera = camera

        self.hypothesis = self.takeTestPicture()
        self.featureCount = self.getFeatureCount()

        self.X_Train = np.matrix



    # -----------------------------------------------------------------
    # --- Take a picture with the camera to detect motion
    # -----------------------------------------------------------------
    def takeTestPicture(self):

        with picamera.array.PiRGBArray(self.camera, size = testSize) as output:
            self.camera.capture(output, 'rgb', resize=testSize)
            return output.array

    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def calculateFeatures(self, picture):

        #squared error of each pixel in new picture
        squaredErrors = np.sum(np.power((self.hypothesis - picture),2), axis=2)

        #print(squaredErrors)
        xOffset = 0
        yOffset = 0
        xi = np.zeros(self.featureCount)
        i = 0
        while xOffset <= testSize[1] - windowWidth:
            while yOffset <= testSize[0] - windowHeight:
                #print(xOffset,yOffset)

                info = np.sum((squaredErrors[xOffset : (xOffset + windowWidth), yOffset: (yOffset + windowHeight)]))

                xi[i] = np.log(info) if (info != 0) else 0
                i += 1;
                yOffset += step
            yOffset = 0
            xOffset += step
        #print(xi)
        return xi


    # -----------------------------------------------------------------
    # ---
    # ----------------------------------------------------------------- #calculate the number of featues based on size of testFrame, windows, and step
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


    #m = number of training examples to gather
    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def gatherTrainingData(self, m):

        #printMessage("Initializing bg using Stochastic Gradient Decent ")
        #self.miniBatchUpdate(m)

        printMessage("Gathering Training Examples")
        i = 0
        temp1 = m
        temp2 = self.featureCount
        np.zeros((temp1, temp2))
        self.X_Train = np.matrix(np.zeros((m, self.featureCount)))

        while i < m:

            pic = self.takeTestPicture()
            self.X_Train[i] = self.calculateFeatures(pic)
            printSameLine("Training Example: " + str(i) + " of " + str(m))
            self.stochasticUpdate(pic)

            i += 1

        print ""

        return
    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def stochasticUpdate(self, picture):
        self.hypothesis = self.hypothesis - alpha * (self.hypothesis - picture)


    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def miniBatchUpdate(self,m):

        i = 0

        while i < m:
            newPic = self.takeTestPicture()
            self.stochasticUpdate(newPic)
            #print(np.sum((np.power((self.hypothesis - newPic),2))))
            print(np.sum(self.calculateFeatures(newPic)))
            i += 1



    #verified with np.var
    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def calculateSigma2(self,mu):
        printMessage("calculating sigma2 (Variance)")
        m = self.X_Train.shape[0]
        sigma2 = ( 1/(float)(m) * np.sum(np.power((self.X_Train - mu), 2),0) )

        #npvar = np.var(self.X_Train, axis=0)
        #print('npvar', npvar)
        #print('m', m)
        print(sigma2)
        return sigma2



    #verified with np.average
    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def calculateMu(self):


        mu = np.array(1/((float)(self.X_Train.shape[0])) * np.sum(self.X_Train,0))

        printMessage("calculating Mu (Avarage)")
        #print(np.average(self.X_Train,0).shape)
        #print("numpy.average: ", np.average(self.X_Train,0))
        #print(mu.shape)
        print(mu)
        mu1 = mu.reshape(-1)

        #self.mu = mu
        return mu1



    # -----------------------------------------------------------------
    # ---
    # -----------------------------------------------------------------
    def probability(self,xi,mu,sigma2):
        #print(xi)
        #print(mu)
        sigma2 = np.array(sigma2).flatten()
        #print(sigma2)
        #print('xi-mu', xi-mu)
        #print('xi-mu2',np.power((xi-mu),2))
        sigma2 = np.array(sigma2).flatten()
        statsn = np.prod(stats.norm(mu,np.sqrt(sigma2)).sf(xi))
        #print('stats.norm', statsn)

        np.prod((1/(2*math.pi )))

        part1 = np.array(1/(np.sqrt(2*math.pi * sigma2)))
        #print('part1', part1)
        part2 = np.exp( - (np.power((xi - mu),2) / (2 * sigma2)))
        #print ('part2', part2)
        prob =  part1 * part2
        #print('prob', prob)
        #print('prob product', np.prod(prob))


        return statsn
    #given new feature vector xi, predict whether or not it is
    #anomalous
    #


def printSameLine(string):
    sys.stdout.write('%s\r' % string)
    sys.stdout.flush()
    #file.write(data)

# -----------------------------------------------------------------
# ---
# -----------------------------------------------------------------
def printMessage(message):
    print("")
    print("#############################################################")
    print("## " + message)
    print("#############################################################")

# -----------------------------------------------------------------
# ---
# -----------------------------------------------------------------
def main():



    with picamera.PiCamera() as camera:
        #set up initial camera settings
        camera.resolution = fullFrameSize
        #stream1 = picamera.PiCameraCircularIO(camera, seconds=5)
        camera.video_stabilization = True


        try:

            motion = Motion(camera,iterations)
            motion.gatherTrainingData(iterations)
            i = 0

            mu = motion.calculateMu()
            sigma2 = motion.calculateSigma2(mu)

            #sigma = motion.calculateCovarianceMatrix(mu)
            printMessage("Checking For Movement")
            while i < iterations:
                pic = motion.takeTestPicture()
                xi = motion.calculateFeatures(pic)
                i += 1
                prob = motion.probability(xi,mu,sigma2)
                probabilityString = "Probability of Movement: " + str(1 - prob) + "              "

                if (prob < epsilon):
                    print("-----------Motion Detected!------------")
                printSameLine(probabilityString)
                motion.stochasticUpdate(pic)

        finally:
            print('')
            print('ending')

    return





# -----------------------------------------------------------------------------
# --- Calls the main program when run from this file
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

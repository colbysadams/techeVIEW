

testSize = (192,128)        # size of frame used for motion detection
motionDir = "motion"        # name of the directory that holds recordings
threshold = 1000            # minimum change for a pixel to be counted
sensitivity = 60            # number of pixels that must be changed
alpha = .2                  # Learning rate
frameSize = (1640,1232)     # size of frame for recordings (best=1640x1232)
cameraName = "cam1"         # name of camera, added to recording filenames
verbose = True              # set to True for more detailed output
stabilize = True            # stabilize the camera
dusk = 40                   # average pixel intensity to go to night mode
dawn = 120                  # average pixel intensity to go to day mode
nightIso = 1600              # make night images brighter (0 to 800)
dayMode = False             # tells whether or not we start in day mode

import KinectNI as ki
import matplotlib.pyplot as plt
import cv2
import time

kinect = ki.open_kinect()

while True:
    image = ki.capture_images(kinect, (kinect.video_mode.resolutionX, kinect.video_mode.resolutionY))
    plt.ion()
    plt.imshow(image)
    plt.show()
    time.sleep(0.1)

#cv2.imshow("IMg", images)

#cv2.waitForKey(30)
#print images

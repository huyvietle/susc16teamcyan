# PyKinect
# Copyright(c) Microsoft Corporation
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED ON AN  *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY
# IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
#
# See the Apache Version 2.0 License for specific language governing
# permissions and limitations under the License.

import thread
import itertools
import ctypes

import numpy as np
import time

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

import sklearn.datasets, sklearn.linear_model, sklearn.neighbors
import matplotlib.pyplot as plt
#import seaborn as sns
import sys, os, time
import scipy.io.wavfile, scipy.signal
# %matplotlib inline
import matplotlib as mpl
from IPython.core.display import HTML
mpl.rcParams['figure.figsize'] = (18.0, 10.0)
import pandas as pd

import socket
import sys
import time

HOST, PORT = "10.100.10.194", 5555
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a UDP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

isRecording = False
KINECTEVENT = pygame.USEREVENT
DEPTH_WINSIZE = 320,240
VIDEO_WINSIZE = 640,480
pygame.init()
LIMB_LIST = {}
RECORD_LIST = []
class Limb:
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y

	def getName(self):
		return self.name

	def getX(self):
		return self.x

	def getY(self):
		return self.y


SKELETON_COLORS = [THECOLORS["red"],
                   THECOLORS["blue"],
                   THECOLORS["green"],
                   THECOLORS["orange"],
                   THECOLORS["purple"],
                   THECOLORS["yellow"],
                   THECOLORS["violet"]]

LEFT_ARM = (JointId.ShoulderCenter,
            JointId.ShoulderLeft,
            JointId.ElbowLeft,
            JointId.WristLeft,
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter,
             JointId.ShoulderRight,
             JointId.ElbowRight,
             JointId.WristRight,
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter,
            JointId.HipLeft,
            JointId.KneeLeft,
            JointId.AnkleLeft,
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter,
             JointId.HipRight,
             JointId.KneeRight,
             JointId.AnkleRight,
             JointId.FootRight)
SPINE = (JointId.HipCenter,
         JointId.Spine,
         JointId.ShoulderCenter,
         JointId.Head)

lastImage = time.time()
skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image

########### MACHINE LEARNING STUFF
next_training = np.concatenate((np.load("./data/next_slide_right.npz")['arr_0'], np.load("./data/next_slide_left.npz")['arr_0']))
# print next_training.shape[0]

prev_training = np.concatenate((np.load("./data/prev_slide_right.npz")['arr_0'], np.load("./data/prev_slide_left.npz")['arr_0']))
# print prev_training.shape[0]

video_on = np.load("./data/video_on.npz")['arr_0']
# print video_on.shape[0]

change_state = np.load("./data/change_state.npz")['arr_0']
# print change_state.shape[0]

start_presenter = np.load("./data/whatever1.npz")['arr_0']
# print start_presenter.shape[0]

gestures_classes = { 0: "next slide", 1: "prev slide", 2:"play video", 3:"change_state", 4:"start pres" }

current_gesture = None

train_features = np.concatenate((next_training, prev_training, video_on, change_state, start_presenter))
train_labels = np.concatenate((np.ones(next_training.shape[0])*0,
                               np.ones(prev_training.shape[0])*1,
                               np.ones(video_on.shape[0])*2,
                               np.ones(change_state.shape[0])*3,
                               np.ones(start_presenter.shape[0])*4))

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)

def perform_machine_learning():
	return None

def store_pos_data(skeletons):
    global lastImage, gestures_classes, current_gesture
    for index, data in enumerate(skeletons):
        # draw the Head
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], dispInfo.current_w, dispInfo.current_h)
        draw_skeleton_data(data, index, SPINE, 10)
        pygame.draw.circle(screen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)

        # SUSC16: It has frames in which the head is not tracked. Hence, better check if it is not 0,0!!
        LIMB_LIST[JointId.Head] = Limb(JointId.Head, int(HeadPos[0]), int(HeadPos[1]))

        for positions in {LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG}:
            start = data.SkeletonPositions[positions[0]]

            for position in itertools.islice(positions, 1, None):
                next = data.SkeletonPositions[position.value]

                curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h)
                curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

                # SUSC16: Get the position of all limbs and store them to the globally
                # available LIMB_LIST.
                if (curstart[0] != 0.0 and curstart[1] != 0.0):
                    LIMB_LIST[position.value] = Limb(position.value,curstart[0], curstart[1])
                start = next


        data = []
        # right side
        #print len(LIMB_LIST)
        if (len(LIMB_LIST) > 5):
            if ((time.time()*1000) - lastImage > 200):
                data.append(LIMB_LIST[JointId.HandRight].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.HandRight].getY() - LIMB_LIST[JointId.Head].getY())
                data.append(LIMB_LIST[JointId.ElbowRight].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.ElbowRight].getY() - LIMB_LIST[JointId.Head].getY())
                data.append(LIMB_LIST[JointId.ShoulderRight].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.ShoulderRight].getY() - LIMB_LIST[JointId.Head].getY())

                # Left side
                data.append(LIMB_LIST[JointId.HandLeft].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.HandLeft].getY() - LIMB_LIST[JointId.Head].getY())
                data.append(LIMB_LIST[JointId.ElbowLeft].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.ElbowLeft].getY() - LIMB_LIST[JointId.Head].getY())
                data.append(LIMB_LIST[JointId.ShoulderLeft].getX() - LIMB_LIST[JointId.Head].getX())
                data.append(LIMB_LIST[JointId.ShoulderLeft].getY() - LIMB_LIST[JointId.Head].getY())

                data = np.array(data).reshape(1, -1)
                classification_result = knn.predict(data)[0]
                res = knn.kneighbors_graph(data, mode="distance").toarray()
                # print res.shape
                dist = res.sum()/570
                print dist
                print gestures_classes[classification_result]
                #print type(dist)
                #sock.sendto(gesture_classes[classification_result] + "\n", (HOST, PORT))
                current_gesture = gestures_classes[classification_result]
                lastImage = time.time() * 1000      
		#if (JointId.HandLeft in LIMB_LIST):
			#print LIMB_LIST[JointId.HandRight].getX(), LIMB_LIST[JointId.HandRight].getY()

def draw_skeleton_data(pSkelton, index, positions, width = 4):
    start = pSkelton.SkeletonPositions[positions[0]]

    for position in itertools.islice(positions, 1, None):
        next = pSkelton.SkeletonPositions[position.value]

        curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h)
        curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

		# SUSC16: Get the position of all limbs and store them to the globally
		# available LIMB_LIST.
        LIMB_LIST[position.value] = Limb(position.value,curstart[0], curstart[1])
        pygame.draw.line(screen, SKELETON_COLORS[index], curstart, curend, width)

        start = next

# recipe to get address of surface: http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
   Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
   Py_ssize_t = ctypes.c_int64
else:
   raise TypeError("Cannot determine type of Py_ssize_t")

_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(Py_ssize_t)]

def surface_to_array(surface):
   buffer_interface = surface.get_buffer()
   address = ctypes.c_void_p()
   size = Py_ssize_t()
   _PyObject_AsWriteBuffer(buffer_interface,
                          ctypes.byref(address), ctypes.byref(size))
   bytes = (ctypes.c_byte * size.value).from_address(address.value)
   bytes.object = buffer_interface
   return bytes

def draw_skeletons(skeletons):
    for index, data in enumerate(skeletons):
        # draw the Head
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], dispInfo.current_w, dispInfo.current_h)
        draw_skeleton_data(data, index, SPINE, 10)
        pygame.draw.circle(screen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)

        # SUSC16: It has frames in which the head is not tracked. Hence, better check if it is not 0,0!!
        LIMB_LIST[JointId.Head] = Limb(JointId.Head, int(HeadPos[0]), int(HeadPos[1]))

        # drawing the limbs
        draw_skeleton_data(data, index, LEFT_ARM)
        draw_skeleton_data(data, index, RIGHT_ARM)
        draw_skeleton_data(data, index, LEFT_LEG)
        draw_skeleton_data(data, index, RIGHT_LEG)



def depth_frame_ready(frame):
    if video_display:
        return

    #sta = surface_to_array(screen)
    #frame.image.copy_bits(sta)
    #img = pygame.image.fromstring(sta.object.raw, (640, 480), 'P', True)
    #pygame.image.save(img, "myFile.bmp")
    #print dir(sta.object.raw)
    #print type(sta.object.raw)
    #print "______________\n"

    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            #print skeletons
            draw_skeletons(skeletons)
        pygame.display.update()


def video_frame_ready(frame):
    if not video_display:
        return

    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        pygame.display.update()

if __name__ == '__main__':
    full_screen = False
    draw_skeleton = True
    video_display = False

    screen_lock = thread.allocate()

    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
    pygame.display.set_caption('Python Kinect Demo')
    skeletons = None
    screen.fill(THECOLORS["black"])

    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons = frame.SkeletonData))
        except:
            # event queue full
            pass

    kinect.skeleton_frame_ready += post_frame

    kinect.depth_frame_ready += depth_frame_ready
    kinect.video_frame_ready += video_frame_ready

    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240, nui.ImageType.Depth)

    print('Controls: ')
    print('     d - Switch to depth view')
    print('     v - Switch to video view')
    print('     s - Toggle displaing of the skeleton')
    print('     u - Increase elevation angle')
    print('     j - Decrease elevation angle')

    # main game loop
    done = False

    commandState = False
    commandTimer = 200

    while not done:

        if commandState:
            commandTimer -= 1
            if commandTimer == 0:
                commandState = False

        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                #draw_skeletons(skeletons)
                store_pos_data(skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    video_display = False
            elif e.key == K_v:
                with screen_lock:
                    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)
                    video_display = True
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2
            elif e.key == K_q:
                exit(0)
            elif e.key == K_e:
                print "Send current gesture."
                sock.sendto(current_gesture, (HOST, PORT))
            elif e.key == K_w:
                isRecording = not isRecording
                
                if (isRecording):
                    print "Started recording"
                else:
                    print "Saving file.."
                    np.savez("change_state", np.array(RECORD_LIST))
                    print "File Saved..."



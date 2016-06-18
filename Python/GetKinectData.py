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

def perform_machine_learning():
	return None

def store_pos_data(skeletons):
    global lastImage
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

    if (isRecording):
        data = []
        # right side
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

            RECORD_LIST.append(np.array(data))
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

    while not done:
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
            elif e.key == K_w:
                isRecording = not isRecording
                
                if (isRecording):
                    print "Started recording"
                else:
                    print "Saving file.."
                    np.savez("change_state", np.array(RECORD_LIST))
                    print "File Saved..."



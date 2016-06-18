import numpy as np
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

train_features = np.concatenate((next_training, prev_training, video_on, change_state, start_presenter))
train_labels = np.concatenate((np.ones(next_training.shape[0])*0,
                               np.ones(prev_training.shape[0])*1,
                               np.ones(video_on.shape[0])*2,
                               np.ones(change_state.shape[0])*3,
                               np.ones(start_presenter.shape[0])*4))

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)
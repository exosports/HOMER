"""
Contains functions related to the NN model.

NNModel: loads the NN model.

"""


import sys, os
import time
import random
from io import StringIO
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from scipy.misc import logsumexp

# Keras
import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.metrics import binary_accuracy
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input, Lambda, Wrapper, merge, concatenate
from keras.engine import InputSpec
from keras.layers.core import Dense, Dropout, Activation, Layer, Lambda, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adadelta, adam
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import GPyOpt

#import callbacks as C
#import loader    as L
#import utils     as U
#import plotter   as P

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


"""
fillin
"""

def NNModel(weight_file='weights.h5'):
    """
    Loads an NN model
    """
    def r2_keras(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    # Build model
    return load_model(weight_file, 
                      custom_objects={"r2_keras": r2_keras})



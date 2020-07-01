"""
Contains functions related to the NN model.

NNModel: loads the NN model.

"""
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


"""
fillin
"""

def NNModel(model_file):
    """
    Loads an NN model
    """
    return load_model(model_file)



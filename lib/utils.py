"""
This file contains utility functions that improve the usage of HOMER.

make_dir: Creates a directory if it does not already exist.

scale: Scales some data according to min, max, and a desired range.

descale: Descales some data according to min, max, and scaled range.

normalize: Normalizes some data according to mean & stdev.

denormalize: Denormalizes some data according to mean & stdev.

"""

import sys, os
import numpy as np


def make_dir(some_dir):
    """
    Handles creation of a directory.

    Inputs
    ------
    some_dir: string. Directory to be created.

    Outputs
    -------
    None. Creates `some_dir` if it does not already exist. 
    Raises an error if the directory cannt be created.
    """
    try:
      os.mkdir(some_dir)
    except OSError as e:
      if e.errno == 17: # Already exists
        pass
      else:
        print("Cannot create folder '{:s}'. {:s}.".format(model_dir,
                                              os.strerror(e.errno)))
        sys.exit()
    return


def scale(val, vmin, vmax, scalelims):
    """
    Scales a value according to min/max values and scaling limits.

    Inputs
    ------
    val      : array. Values to be scaled.
    vmin     : array. Minima of `val`.
    vmax     : array. Maxima of `val`.
    scalelims: list, floats. [min, max] of range of scaled data.

    Outputs
    -------
    Array of scaled data.
    """
    return (scalelims[1] - scalelims[0]) * (val - vmin) / \
           (vmax - vmin) + scalelims[0]


def descale(val, vmin, vmax, scalelims):
    """
    Descales a value according to min/max values and scaling limits.

    Inputs
    ------
    val      : array. Values to be descaled.
    vmin     : array. Minima of `val`.
    vmax     : array. Maxima of `val`.
    scalelims: list, floats. [min, max] of range of scaled data.

    Outputs
    -------
    Array of descaled data.
    """
    return (val - scalelims[0]) / (scalelims[1] - scalelims[0]) * \
           (vmax - vmin) + vmin


def normalize(val, vmean, vstd):
    """
    Normalizes a value according to a mean and standard deviation.

    Inputs
    ------
    val  : array. Values to be normalized.
    vmean: array. Mean  values of `val`.
    vstd : array. Stdev values of `val`.

    Outputs
    -------
    Array of normalized data.
    """
    return (val - vmean) / vstd


def denormalize(val, vmean, vstd):
    """
    Denormalizes a value according to a mean and standard deviation.

    Inputs
    ------
    val  : array. Values to be denormalized.
    vmean: array. Mean  values of `val`.
    vstd : array. Stdev values of `val`.

    Outputs
    -------
    Array of denormalized data.
    """
    return val * vstd + vmean



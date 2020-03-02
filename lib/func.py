import sys, os
import time
import numpy as np

import utils as U


def eval(params, model, 
         x_mean, x_std, y_mean, y_std, 
         x_min,  x_max, y_min,  y_max, scalelims, 
         wavenum=None, 
         starspec=None, factor=1, 
         filters=None, ifilt=None, 
         conv=False, ilog=False, olog=False):
    """
    Function to be evaluated for each MCMC iteration
    """
    # Normalize & scale
    pars = U.scale(U.normalize(params, x_mean, x_std), 
                   x_min, x_max, scalelims)
    if len(pars.shape) == 1:
        pars = np.expand_dims(pars, 0)
    if conv:
        pars = np.expand_dims(pars, -1)

    # Predict
    pred = model.predict(pars)

    # Post-process
    # Descale & denormalize
    pred = U.denormalize(U.descale(pred, y_min, y_max, scalelims), 
                         y_mean, y_std)
    # De-log
    if olog:
        pred = 10**pred
    # Divide by stellar spectrum
    if starspec is not None:
        pred = pred / starspec
    # Multiply by any conversion factors, e.g., R_p/R_s, unit conversion
    pred *= factor

    return pred


def eval_binned(params, model, 
                x_mean, x_std, y_mean, y_std, 
                x_min,  x_max, y_min,  y_max, scalelims, 
                wavenum=None, 
                starspec=None, factor=1, 
                filters=None, ifilt=None, 
                conv=False, ilog=False, olog=False):
    """
    Evaluates the model for given inputs. Bins the output according to filters.
    """
    if ilog:
        params = np.log10(params)
    # Normalize & scale
    pars = U.scale(U.normalize(params, x_mean, x_std), 
                   x_min, x_max, scalelims)
    # Add channel if convolutional layers are used
    if len(pars.shape) == 1:
        pars = np.expand_dims(pars, 0)
    if conv:
        pars = np.expand_dims(pars, -1)

    # Predict
    pred = model.predict(pars)

    # Post-process
    # Descale & denormalize
    pred = U.denormalize(U.descale(pred, y_min, y_max, scalelims), 
                         y_mean, y_std)
    # De-log
    if olog:
        pred = 10**pred
    # Divide by stellar spectrum
    if starspec is not None:
        pred = pred / starspec

    # Multiply by any conversion factors, e.g., R_p/R_s, unit conversion
    pred *= factor

    # Band integrate according to filters
    nfilters = len(filters)
    results  = np.zeros((pars.shape[0], nfilters))
    for i in range(nfilters):
        results[:, i] = np.trapz(pred[:,ifilt[i,0]:ifilt[i,1]] * filters[i], wavenum[ifilt[i,0]:ifilt[i,1]], axis=-1)

    return results



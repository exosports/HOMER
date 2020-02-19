import sys, os
import time
import numpy as np
import scipy.interpolate as si

import utils as U


def eval(params, model, 
         x_mean, x_std, y_mean, y_std, 
         x_min,  x_max, y_min,  y_max, scalelims, 
         wavenum=None, wnfact=1, 
         starspec=None, factor=1, 
         filters=None, filt2um=1, 
         conv=False, olog=False):
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
                wavenum=None, wnfact=1, 
                starspec=None, factor=1, 
                filters=None, filt2um=1, 
                conv=False, olog=False):
    """
    Evaluates the model for given inputs. Bins the output according to filters.
    """
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
        pred = pred / starspecw

    # Multiply by any conversion factors, e.g., R_p/R_s, unit conversion
    pred *= factor

    # Make sure `wavenum` is units of cm-1
    wavenum *= wnfact

    # Read the filters, resample to `wavenum`, integrate according to filter
    nfilters = len(filters)
    results  = np.zeros((params.shape[0], nfilters))
    for i in range(nfilters):
        datfilt = np.loadtxt(filters[i])
        # Convert filter wavelenths to microns, then convert um -> cm-1
        finterp = si.interp1d(10000. / (filt2um * datfilt[:,0]), 
                              datfilt[:,1],
                              bounds_error=False, fill_value=0)
        # Interpolate and normalize
        tranfilt = finterp(wavenum)
        tranfilt = tranfilt / np.trapz(tranfilt, wavenum)
        # Band integrate
        results[:, i] = np.trapz(pred * tranfilt, wavenum, axis=-1)

    return results



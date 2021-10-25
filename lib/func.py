"""
Contains functions to be evaluated at each MCMC iteration. 
All functions must have the same set of inputs, even if they are not all used.
These are denoted in documentation as "filler", and optional inputs are 
denoted as such.

eval       : makes predictions on inputs.
eval_binned: makes predictions on inputs, then integrates according to filters.

"""

import sys, os
import time
import multiprocessing as mp
import numpy as np

import utils as U


def eval(params, nn, 
         x_mean, x_std, y_mean, y_std, 
         x_min,  x_max, y_min,  y_max, scalelims, 
         xvals=None, 
         starspec=None, factor=None, 
         filters=None, ifilt=None, 
         ilog=False, olog=False, 
         kll=None, count=0, burnin=0):
    """
    Evaluates the model for given inputs. 
    Integrates the output according to filters.

    Inputs
    ------
    params   : array. Parameters to be predicted on.
    nn       : object. Trained NN model.
    x_mean   : array. Mean of inputs.
    x_std    : array. Standard deviation of inputs.
    y_mean   : array. Mean of outputs.
    y_std    : array. Standard deviation of outputs.
    x_min    : array. Minima of inputs.
    x_max    : array. Maxima of inputs.
    y_min    : array. Minima of outputs.
    y_max    : array. Maxima of outputs.
    scalelims: list.  [Lower, upper] bounds for scaling.
    xvals    : array. (filler) X-axis values associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `xvals`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `xvals` indices where the filters are nonzero.
    ilog     : bool. True if the NN input  is the log10 of the inputs.
    olog     : bool. True if the NN output is the log10 of the outputs.
    kll      : object. (optional) Streaming quantiles calculator.
    count    : array.  Ensures that burned iterations do not 
                       contribute to quantiles.
                       Must be specified if KLL is not None.
    burnin   : int.    Number of burned iterations. 
                       Must be specified if KLL is not None.

    Outputs
    -------
    results: array. Integrated predicted values.
    """
    # Input must be 2D
    if len(params.shape) == 1:
        params = np.expand_dims(params, 0)

    # Log-scale
    if ilog:
        params[:, ilog] = np.log10(params[:, ilog])

    # Normalize & scale
    pars = U.scale(U.normalize(params, x_mean, x_std), 
                   x_min, x_max, scalelims)

    # Predict
    pred = nn.predict(pars)

    # Post-process
    # Descale & denormalize
    pred = U.denormalize(U.descale(pred, y_min, y_max, scalelims), 
                         y_mean, y_std)
    # De-log
    if olog:
        pred[:, olog] = 10**pred[:, olog]

    # Divide by stellar spectrum
    if starspec is not None:
        pred = pred / starspec

    # Multiply by any conversion factors, e.g., R_p/R_s, unit conversion
    if factor is not None:
        pred *= factor

    # Update the streaming quantiles
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred)

    return pred


def eval_binned(params, nn, 
                x_mean, x_std, y_mean, y_std, 
                x_min,  x_max, y_min,  y_max, scalelims, 
                xvals, 
                starspec=None, factor=None, 
                filters=None, ifilt=None, 
                ilog=False, olog=False, 
                kll=None, count=0, burnin=0):
    """
    Evaluates the model for given inputs. 
    Integrates the output according to filters.

    Inputs
    ------
    params   : array. Parameters to be predicted on.
    nn       : object. Trained NN model.
    x_mean   : array. Mean of inputs.
    x_std    : array. Standard deviation of inputs.
    y_mean   : array. Mean of outputs.
    y_std    : array. Standard deviation of outputs.
    x_min    : array. Minima of inputs.
    x_max    : array. Maxima of inputs.
    y_min    : array. Minima of outputs.
    y_max    : array. Maxima of outputs.
    scalelims: list.  [Lower, upper] bounds for scaling.
    xvals    : array. X-axis values associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `xvals`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. Transmission of filters.
    ifilt    : array. `xvals` indices where the filters are nonzero.
    ilog     : bool. True if the NN input  is the log10 of the inputs.
    olog     : bool. True if the NN output is the log10 of the outputs.
    kll      : object. (optional) Streaming quantiles calculator.
    count    : array.  Ensures that burned iterations do not 
                       contribute to quantiles.
                       Must be specified if KLL is not None.
    burnin   : int.    Number of burned iterations. 
                       Must be specified if KLL is not None.

    Outputs
    -------
    results: array. Integrated predicted values.
    """
    # Input must be 2D
    if len(params.shape) == 1:
        params = np.expand_dims(params, 0)

    # Log-scale
    if ilog:
        params[:, ilog] = np.log10(params[:, ilog])

    # Normalize & scale
    pars = U.scale(U.normalize(params, x_mean, x_std), 
                   x_min, x_max, scalelims)

    # Predict
    pred = nn.predict(pars)

    # Post-process
    # Descale & denormalize
    pred = U.denormalize(U.descale(pred, y_min, y_max, scalelims), 
                         y_mean, y_std)
    # De-log
    if olog:
        pred[:, olog] = 10**pred[:, olog]

    # Divide by stellar spectrum
    if starspec is not None:
        pred = pred / starspec

    # Multiply by any conversion factors, e.g., R_p/R_s, unit conversion
    if factor is not None:
        pred *= factor

    # Update the streaming quantiles
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred)

    # Band integrate according to filters
    nfilters = len(filters)
    results  = np.zeros((pars.shape[0], nfilters))
    for i in range(nfilters):
        results[:, i] = np.trapz(pred[:,ifilt[i,0]:ifilt[i,1]] * filters[i], xvals[ifilt[i,0]:ifilt[i,1]], axis=-1)

    return results



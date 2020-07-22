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


def eval_Smith(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return pred


def eval_Smith_binned2(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:,0::2]+pred[:,1::2])/2.


def eval_Smith_binned4(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:,0::4] + pred[:,1::4] + pred[:,2::4] + pred[:,3::4]) / 4.


def eval_Smith_binned7(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:,0::7] + pred[:,1::7] + pred[:,2::7] + pred[:,3::7] + 
            pred[:,4::7] + pred[:,5::7] + pred[:,6::7]) / 7.


def eval_Smith_binned14(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:, 0::14] + pred[:, 1::14] + pred[:, 2::14] + pred[:, 3::14] + 
            pred[:, 4::14] + pred[:, 5::14] + pred[:, 6::14] + pred[:, 7::14] + 
            pred[:, 8::14] + pred[:, 9::14] + pred[:,10::14] + pred[:,11::14] + 
            pred[:,12::14] + pred[:,13::14]) / 14.


def eval_Smith_binned23(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:, 0::23] + pred[:, 1::23] + pred[:, 2::23] + pred[:, 3::23] + 
            pred[:, 4::23] + pred[:, 5::23] + pred[:, 6::23] + pred[:, 7::23] + 
            pred[:, 8::23] + pred[:, 9::23] + pred[:,10::23] + pred[:,11::23] + 
            pred[:,12::23] + pred[:,13::23] + pred[:,14::23] + pred[:,15::23] + 
            pred[:,16::23] + pred[:,17::23] + pred[:,18::23] + pred[:,19::23] + 
            pred[:,20::23] + pred[:,21::23] + pred[:,22::23]) / 23.


def eval_Smith_binned28(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    return (pred[:, 0::28] + pred[:, 1::28] + pred[:, 2::28] + pred[:, 3::28] + 
            pred[:, 4::28] + pred[:, 5::28] + pred[:, 6::28] + pred[:, 7::28] + 
            pred[:, 8::28] + pred[:, 9::28] + pred[:,10::28] + pred[:,11::28] + 
            pred[:,12::28] + pred[:,13::28] + pred[:,14::28] + pred[:,15::28] + 
            pred[:,16::28] + pred[:,17::28] + pred[:,18::28] + pred[:,19::28] + 
            pred[:,20::28] + pred[:,21::28] + pred[:,22::28] + pred[:,23::28] + 
            pred[:,24::28] + pred[:,25::28] + pred[:,26::28] + pred[:,27::28]) / 28.


def eval_Smith_binned46(params, nn, 
               x_mean, x_std, y_mean, y_std, 
               x_min,  x_max, y_min,  y_max, scalelims, 
               wavenum=None, 
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
    wavenum  : array. (filler) Wavenumbers (cm-1) associated with the NN output.
    starspec : array. (optional) Stellar spectrum at `wavenum`.
    factor   : float. (optional) Multiplication factor to convert the 
                      de-normalized NN output.
    filters  : list, arrays. (filler) Transmission of filters.
    ifilt    : array. (filler) `wavenum` indices where the filters are nonzero.
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
    if ilog:
        params = np.log10(params)
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
        pred = 10**pred

    # Enforce cloudtop pressure < surface pressure
    pred[params[:,-1] >= params[:,3]] = -100
    # Update the streaming quantiles for valid models
    if kll is not None:
        count += 1
        if count[0] > burnin:
            kll.update(pred[params[:,-1] < params[:,3]])

    out = 0
    for i in range(46):
        out += pred[:, i::46]
    out /= 46.

    return out



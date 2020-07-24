import sys
import numpy as np


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


def model(pars, nn, inD, pstep, pinit, 
          ilog, olog, 
          x_mean, x_std, x_min, x_max, 
          y_mean, y_std, y_min, y_max, scalelims, 
          xvals=None, filters=None, ifilt=None, 
          fullout=False):
    # Load params
    params = np.zeros(inD, dtype=float)
    n = 0
    for i in np.arange(inD)[pstep>0]:
        params[i] = pars[n]
        n += 1
    params[pstep==0] = pinit[pstep==0]

    # Standardize
    if ilog:
        params[:, ilog] = np.log10(params[:, ilog])
    params = scale(normalize(params, x_mean, x_std), 
                   x_min, x_max, scalelims)
    # Predict
    pred = nn.predict(params[None, :])
    # De-standardize
    pred = denormalize(descale(pred, y_min, y_max, scalelims), 
                       y_mean, y_std)
    if olog:
        pred[:, olog] = 10**pred[:, olog]

    if not fullout and filters is not None and xvals is not None:
        nfilters = len(filters)
        results  = np.zeros(nfilters)
        for i in range(nfilters):
            results[i] = np.sum(pred[:,ifilt[i,0]:ifilt[i,1]] * filters[i]) / filters[i].sum()

        return results
    else:
        return pred


def prior(cube, ndim, nparams, pmin, pmax, pstep):
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    for i in range(ndim):
        cube[i] = cube[i]                                 \
                  * (pmax[pstep>0][i] - pmin[pstep>0][i]) \
                  +  pmin[pstep>0][i]
    return cube


def loglikelihood(cube, ndim, nparams, data, uncert, 
                  nn, inD, pstep, pinit, 
                  ilog, olog, 
                  x_mean, x_std, x_min, x_max, 
                  y_mean, y_std, y_min, y_max, scalelims, 
                  xvals, filters, ifilt):
    ymodel = model(cube, nn, inD, pstep, pinit, 
                   ilog, olog, 
                   x_mean, x_std, x_min, x_max, 
                   y_mean, y_std, y_min, y_max, scalelims, 
                   xvals, filters, ifilt)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike



"""
This module handles calculation of statistics about the posterior's 
credible regions.

credregion: computes the credible region of a posterior for given percentiles.

"""

import sys, os
import numpy as np
import scipy.stats as stats
import scipy.interpolate as si


def credregion(posterior, percentile=[0.6827, 0.9545, 0.9973], 
               lims=(None,None), numpts=100):
    """
    posterior : 1d array of parameter value at each iteration.
    percentile: 1D float ndarray, list, or float.
                The percentile (actually the fraction) of the credible region.
                A value in the range: (0, 1).
    lims: tuple, floats. Minimum and maximum allowed values for posterior. 
                         Should only be used if there are physically-imposed 
                         limits.
    numpts: int. Number of points to use when calculating the PDF.
    """
    # Make sure `percentile` is a list or array
    if type(percentile) == float:
        percentile = np.array([percentile])

    # Compute the posterior's PDF:
    kernel = stats.gaussian_kde(posterior)
    # Use a Gaussian kernel density estimate to trace the PDF:
    # Interpolate-resample over finer grid (because kernel.evaluate
    #  is expensive):
    if lims[0] is not None:
        lo = min(np.amin(posterior), lims[0])
    else:
        lo = np.amin(posterior)
    if lims[1] is not None:
        hi = max(np.amax(posterior), lims[1])
    else:
        hi = np.amax(posterior)
    x    = np.linspace(lo, hi, numpts)
    f    = si.interp1d(x, kernel.evaluate(x))
    xpdf = np.linspace(lo, hi, 100*numpts)
    pdf  = f(xpdf)


    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])

    # List to hold boundaries of CRs
    # List is used because a given CR may be multiple disconnected regions
    CRlo = []
    CRhi = []
    # Find boundary for each specified percentile
    for i in range(len(percentile)):
        # Indices of the highest posterior density:
        iHPD = np.where(cdf >= percentile[i]*cdf[-1])[0][0]
        # Minimum density in the HPD region:
        HPDmin   = np.amin(pdf[ip][0:iHPD])
        # Find the contiguous areas of the PDF greater than or equal to HPDmin
        HPDbool  = pdf >= HPDmin
        idiff    = np.diff(HPDbool) # True where HPDbool changes T to F or F to T
        iregion, = idiff.nonzero()  # Indexes of Trues. Note , because returns tuple
        # Check boundaries
        if HPDbool[0]:
            iregion = np.insert(iregion, 0, -1) # This -1 is changed to 0 below when 
        if HPDbool[-1]:                       #   correcting start index for regions
            iregion = np.append(iregion, len(HPDbool)-1)
        # Reshape into 2 columns of start/end indices
        iregion.shape = (-1, 2)
        # Add 1 to start of each region due to np.diff() functionality
        iregion[:,0] += 1
        # Store the min and max of each (possibly disconnected) region
        CRlo.append(xpdf[iregion[:,0]])
        CRhi.append(xpdf[iregion[:,1]])

    return pdf, xpdf, CRlo, CRhi



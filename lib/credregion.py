"""
This module handles calculation of statistics about the posterior's 
credible regions.

credregion: computes the credible region of a posterior for given percentiles.

convergetest: performs the Gelman & Rubin test

gelmanrubin: Gelman & Rubin test

sig: computes the uncertainties on credible regions based on the ESS

ess: computes the effective sample size (ESS).

"""

import sys, os
import numpy as np
import scipy.stats as stats
import scipy.interpolate as si


def credregion(posterior, percentile=[0.6827, 0.9545, 0.9973], 
               lims=(None,None), numpts=100):
    """
    Computes the credible region of a 1D posterior for given percentiles.

    Inputs
    ------
    posterior : 1d array of parameter value at each iteration.
    percentile: 1D float ndarray, list, or float.
                The percentile (actually the fraction) of the credible region.
                A value in the range: (0, 1).
    lims: tuple, floats. Minimum and maximum allowed values for posterior. 
                         Should only be used if there are physically-imposed 
                         limits.
    numpts: int. Number of points to use when calculating the PDF.
                 Note: large values have significant compute cost.

    Outputs
    -------
    pdf : array. Probability density function.
    xpdf: array. X values associated with `pdf`.
    CRlo: list.  Lower bounds on credible region(s).
    CRhi: list.  Upper bounds on credible region(s).
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


def convergetest(chains):
    """
    Wrapper for the Gelman & Rubin (1992) convergence test on a MCMC
    chain parameters.

    Parameters
    ----------
    chains : ndarray
        A 3D array of shape (nchains, nparameters, chainlen) containing
        the parameter MCMC chains.

    Returns
    -------
    psrf : ndarray
        The potential scale reduction factors of the chain.  If the
        chain has converged, each value should be close to unity.  If
        they are much greater than 1, the chain has not converged and
        requires more samples.  The order of psrfs in this vector are
        in the order of the free parameters.

    Previous (uncredited) developers
    --------------------------------
    Chris Campo

    Notes
    -----
    Taken from MCcubed's gelman_rubin.py
    """
    # Allocate placeholder for results:
    npars = np.shape(chains)[1]
    psrf  = np.zeros(npars)
    
    # Calculate psrf for each parameter:
    for i in range(npars):
        psrf[i] = gelmanrubin(chains[:, i, :])
    
    return psrf


def gelmanrubin(chains):
    """
    Calculate the potential scale reduction factor of the Gelman & Rubin
    convergence test on a fitting parameter

    Parameters
    ----------
    chains: 2D ndarray
       Array containing the chains for a single parameter.  Shape
       must be (nchains, chainlen)

    Notes
    -----
    Taken from MCcubed's gelman_rubin.py
    """
    # Get length of each chain and reshape:
    nchains, chainlen = np.shape(chains)
    
    # Calculate W (within-chain variance):
    W = np.mean(np.var(chains, axis=1))
    
    # Calculate B (between-chain variance):
    means = np.mean(chains, axis=1)
    mmean = np.mean(means)
    B     = (chainlen/(nchains-1.0)) * np.sum((means-mmean)**2)
    
    # Calculate V (posterior marginal variance):
    V = W*((chainlen - 1.0)/chainlen) + B*((nchains + 1.0)/(chainlen*nchains))
    
    # Calculate potential scale reduction factor (PSRF):
    psrf = np.sqrt(V/W)
    
    return psrf




def sig(ess, p_est=np.array([0.68269, 0.86639, 0.95450, 0.98758, 0.99730])):
    """
    Computes the 1, 1.5, 2, 2.5, 3 sigma uncertainties given an effective 
    sample size. Assumes flat prior on a credible region, with the mean 
    and standard deviation estimated from the posterior.

    Inputs
    ------
    ess  : int.   Effective sample size.
    p_est: array. Credible regions to estimate uncertainty.

    Revisions
    ---------
    2019-09-13  mhimes            Original implementation.
    """
    return ((1.-p_est)*p_est/(ess+3))**0.5


def ess(allparams):
    """
    Computes the effective sample size (ESS) for an MCMC run.

    Inputs
    ------
    allparams: 3D array. Posterior distribution of shape 
                         (num_chains, num_params, num_iter)

    Outputs
    -------
    speis: int.   Number of steps per effective independent sample (SPEIS).
    ess  : float. Effective sample size.

    Notes
    -----
    This code is adapted from Ryan Challener's MCcubed pull request
    """
    totiter = allparams.shape[-1] * allparams.shape[0]
    # Calculate the ESS
    nisamp = np.zeros((allparams.shape[0], allparams.shape[1]))
    for nc in range(allparams.shape[0]):
        allpac = []
        for p in range(allparams.shape[1]):
            # Autocorrelation
            meanapi   = np.mean(     allparams[nc,p])
            autocorr  = np.correlate(allparams[nc,p]-meanapi,
                                     allparams[nc,p]-meanapi, mode='full')
            # It's symmetric, keep only lags >= 0
            allpac.append(autocorr[np.size(autocorr)//2:] / np.max(autocorr))
            # Sum adjacent pairs (see Geyer 1992, Sec. 3.3)
            adjsum = allpac[p][:-1:2] + allpac[p][1::2]
            # Find first negative value (IPS method)
            if np.any(adjsum<0):
                cutoff = np.where(adjsum < 0)[0][0]
            else:
                cutoff = len(adjsum)
            # Number of independent samples
            nisamp[nc, p] = 1 + 2 * np.sum(adjsum[:cutoff])

    speis = int(np.ceil(np.amax(nisamp)))
    
    return speis, totiter/speis





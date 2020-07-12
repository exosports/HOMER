"""
Contains functions related to plotting the best-fit model.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

plt.ion()


def get_bestfit(allmodel, allparam, flog, ifree):
    """
    Extracts the best-fit model from the posterior.

    Inputs
    ------
    allmodel: array. All evaluated models.
    allparam: array. Parameters of evaluated models.
    flog    : str.   MC3 log file.
    ifree   : array, bools. True for free parameters, 
                            False for fixed/shared parameters.

    Outputs
    -------
    bestfit: array. Best-fit model.
    """
    if flog is None:
        print("No MC3 logfile found.  Best-fit parameters will not be " \
            + "determined.")
        return None
    else:
        # Read the log
        log   = open(flog, 'r')
        lines = np.asarray(log.readlines())
        log.close()
        # Find where best-fit params begin
        for i in np.arange(len(lines)):
            if lines[i].startswith(' Best-fit params'):
                break
        i   += 1
        end  = i
        for end in np.arange(i, len(lines)):
            if lines[end].strip() == "":
                break
        # Read the params
        bestp = np.zeros(end-i, np.double)
        for n in np.arange(i, end):
            parvals    = lines[n].split()
            bestp[n-i] = parvals[0]

        # Find the corresponding model by minimizing sum of squared differences
        ind = np.argmin(np.sum((allparam - bestp[ifree, None])**2, axis=0))
        bestfit = allmodel[:, ind]

        return bestfit


def plot_bestfit(outputdir, xvals, data, uncert, meanwn, ifilt, bestfit, 
                 xlabel, ylabel, kll=None, wn=True):
    """
    Plots the best-fit model.

    Inputs
    ------
    outputdir: string. Directory where plot will be saved.
    xvals  : array.  X values values for unbinned models. 
                     Must be either in cm-1 (with wn=True) or um (with wn=False)
    data   : array.  Data points being fit.
    uncert : array.  Uncertainties on data points being fit.
    meanwn : array.  Mean wavenumuber of each filter.
    ifilt  : array.  Indices of `xvals` corresponding to the filter bandpass.
                     shape: (len(filters), 2)
                     ifilt[0, 0] gives the first filter's starting index
                     ifilt[0, 1] gives the first filter's ending   index
    bestfit: array.  Best-fit model values, corresponding to `data`.
    xlabel : string. X-axis label.
    ylabel : string. Y-axis label.
    kll    : object. Streaming quantiles calculator, for 1-2-3 sigma spectra.
    wn     : bool.   Determines if `xvals` is in wavenumber or wavelength.

    Outputs
    -------
    plot of the best-fit model
    """
    # Get the 1, 2, 3sigma models
    if kll is not None:
        lo1    = kll.get_quantiles(0.1587)[:, 0]
        hi1    = kll.get_quantiles(0.8413)[:, 0]
        lo2    = kll.get_quantiles(0.0228)[:, 0]
        hi2    = kll.get_quantiles(0.9772)[:, 0]
        lo3    = kll.get_quantiles(0.0014)[:, 0]
        hi3    = kll.get_quantiles(0.9986)[:, 0]

    # Convert wavenumber --> microns
    if wn:
        xvals  = 1e4/xvals
    # Set up the x-axis error array
    if ifilt is None:
        xerr = np.zeros((2, len(xvals)))
        xerr[0, 1:  ] = np.abs(xvals[1:  ] - xvals[ :-1])/2
        xerr[1,  :-1] = np.abs(xvals[ :-1] - xvals[1:  ])/2
        xerr[0, 0   ] = xerr[0, 1]
        xerr[1,-1   ] = xerr[0,-1]
        xax           = xvals
    else:
        if wn:
            meanwn = 1e4/meanwn
            ifilt = ifilt[:,::-1]
        xerr = np.abs(xvals[ifilt].T - meanwn)
        xax  = meanwn

    # Plot
    plt.figure(42, dpi=600)
    plt.clf()
    ax = plt.subplot(111)
    ymin =  np.inf
    ymax = -np.inf
    if kll is not None:
        ax.fill_between(xvals, lo3, hi3, facecolor="#d9ecff", 
                        edgecolor="#d9ecff", label="3$\sigma$")
        ax.fill_between(xvals, lo2, hi2, facecolor="#C0DFFF", 
                        edgecolor="#C0DFFF", label="2$\sigma$")
        ax.fill_between(xvals, lo1, hi1, facecolor="cornflowerblue", 
                        edgecolor="cornflowerblue", label="1$\sigma$")
        plt.plot(xvals, median, "royalblue", label="Median")
        ymin = np.amin([ymin, lo3.min()])
        ymax = np.amax([ymax, hi3.max()])
    plt.scatter( xax, bestfit, c="k", label="Best fit", zorder=30, 
                 lw=1, s=6)
    plt.errorbar(xax, data, yerr=uncert, xerr=xerr, 
                 fmt="or", markersize=1.5, capsize=1.5, elinewidth=1, 
                 ecolor='tab:red', label="Data", zorder=20)
    ymin = np.amin([ymin, data.min()-uncert.max(), bestfit.min()-uncert.max()])
    ymax = np.amax([ymax, data.max()+uncert.max(), bestfit.max()+uncert.max()])
    plt.legend(loc='best')
    plt.ylim(ymin, ymax)
    ax.set_ylabel(r""+ylabel)
    ax.set_xlabel(r""+xlabel)
    ax.set_xscale('log')
    formatter = tck.FuncFormatter(lambda y, _: '{:.8g}'.format(y))
    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_xaxis().set_minor_formatter(formatter)
    plt.savefig(outputdir+'bestfit_spectrum.png', bbox_inches='tight')
    plt.close()



"""
Contains functions related to plotting the best-fit model.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

plt.ion()


def get_bestfit(allmodel, allparam, flog):
    """
    Extracts the best-fit model from the posterior.

    Inputs
    ------
    allmodel: array. All evaluated models.
    allparam: array. Parameters of evaluated models.
    flog    : str.   MC3 log file.

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
        ind = np.argmin(np.sum((allparam - bestp[:, None])**2, axis=0))
        bestfit = allmodel[:, ind]

        return bestfit


def plot_bestfit(outputdir, xvals, data, uncert, meanwn, ifilt, bestfit, kll=None, wn=True):
    """
    Plots the best-fit model.

    Inputs
    ------
    outputdir: string. Directory where plot will be saved.
    xvals  : array. X values values for unbinned models. 
                    Must be either in cm-1 (with wn=True) or um (with wn=False)
    data   : array. Data points being fit.
    uncert : array. Uncertainties on data points being fit.
    meanwn : array. Mean wavenumuber of each filter.
    ifilt  : array. Indices of `xvals` corresponding to the filter bandpass.
                    shape: (len(filters), 2)
                    ifilt[0, 0] gives the first filter's starting index
                    ifilt[0, 1] gives the first filter's ending   index
    bestfit: array. Best-fit model values, corresponding to `data`.
    kll    : object. Streaming quantiles calculator, for 1-2-3 sigma spectra.
    wn     : bool.  Determines if `xvals` is in wavenumber or wavelength.

    Outputs
    -------
    plot of the best-fit model
    """
    # Get the median, 1, 2, 3sigma models
    if kll is not None:
        median = kll.get_quantiles(0.5000)[:, 0]*1e3
        lo1    = kll.get_quantiles(0.1587)[:, 0]*1e3
        hi1    = kll.get_quantiles(0.8413)[:, 0]*1e3
        lo2    = kll.get_quantiles(0.0228)[:, 0]*1e3
        hi2    = kll.get_quantiles(0.9772)[:, 0]*1e3
        lo3    = kll.get_quantiles(0.0014)[:, 0]*1e3
        hi3    = kll.get_quantiles(0.9986)[:, 0]*1e3
        # The *1e3 factor is for plotting

    if wn:
        xvals  = 1e4/xvals
        meanwn = 1e4/meanwn
        ifilt = ifilt[:,::-1]

    # Plot
    plt.figure(42, dpi=600)
    plt.clf()
    ax = plt.subplot(111)
    if kll is not None:
        ax.fill_between(xvals, lo3, hi3, facecolor="#d9ecff", 
                        edgecolor="#d9ecff", label="3$\sigma$")
        ax.fill_between(xvals, lo2, hi2, facecolor="#C0DFFF", #62B1FF
                        edgecolor="#C0DFFF", label="2$\sigma$")
        ax.fill_between(xvals, lo1, hi1, facecolor="cornflowerblue", #1873CC
                        edgecolor="cornflowerblue", label="1$\sigma$")
        plt.plot(xvals, median, "royalblue", label="Median")
    plt.scatter(meanwn, bestfit*1e3, c="k", label="Best fit", zorder=30, 
                lw=1, s=16)
    plt.errorbar(meanwn, data*1e3, 
                 yerr=uncert*1e3, xerr=np.abs(xvals[ifilt].T - meanwn), 
                 fmt="or", markersize=3, capsize=2, elinewidth=1, 
                 ecolor='tab:red', label="Data", zorder=20)
    plt.legend(loc='best')
    ax.set_xlim(np.amin(xvals), np.amax(xvals))
    ax.set_ylabel(r"$F_p/F_s$ (10$^{-3}$)")
    ax.set_xlabel("Wavelength ${\\rm(\u03bcm)}$")
    ax.set_xscale('log')
    formatter = tck.FuncFormatter(lambda y, _: '{:.8g}'.format(y))
    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_xaxis().set_minor_formatter(formatter)
    plt.savefig(outputdir+'bestfit_spectrum.png')
    plt.close()



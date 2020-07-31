"""
Contains functions related to plotting the best-fit model.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

plt.ion()


def plot_bestfit(outputdir, xvals, data, uncert, meanwave, ifilt, bestfit, 
                 xlabel, ylabel, kll=None, wn=True, 
                 bestpars=None, truepars=None, title=False, ndec=None):
    """
    Plots the best-fit model.

    Inputs
    ------
    outputdir: string. Directory where plot will be saved.
    xvals   : array.  X values values for unbinned models. 
                     Must be either in cm-1 (with wn=True) or um (with wn=False)
    data    : array.  Data points being fit.
    uncert  : array.  Uncertainties on data points being fit.
    meanwave: array.  Mean wavenumuber/wavelength of each filter.
    ifilt   : array.  Indices of `xvals` corresponding to the filter bandpass.
                      shape: (len(filters), 2)
                      ifilt[0, 0] gives the first filter's starting index
                      ifilt[0, 1] gives the first filter's ending   index
    bestfit : array.  Best-fit model values, corresponding to `data`.
    xlabel  : string. X-axis label.
    ylabel  : string. Y-axis label.
    kll     : object. Streaming quantiles calculator, for 1-2-3 sigma spectra.
    wn      : bool.   Determines if `xvals` is in wavenumber or wavelength.
    bestpars: array.  If None, will not be used.  Otherwise, array of best-fit 
                      parameter values.  If title is True, they will be 
                      included in the title.
    truepars: array.  If None, will not be used.  Otherwise, array of true 
                      values.  If title is True, they will be 
                      included in the title.
    title   : bool.   If False, does not plot title. 
                      If True, plots the best-fit parameters as the title.
    ndec    : array.  Number of places to round title values.

    Outputs
    -------
    plot of the best-fit model
    """
    pad = 1.04 # to pad the edges of the plot
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
            meanwave = 1e4/meanwave
            ifilt = ifilt[:,::-1]
        xerr = np.abs(xvals[ifilt].T - meanwave)
        xax  = meanwave

    xlims = np.amin(meanwave)-pad*xerr[0,0], np.amax(meanwave)+pad*xerr[-1,-1]

    # Plot
    fig1 = plt.figure(42, dpi=600)
    plt.clf()
    ax = fig1.add_axes((.1, .3, .8, .6))
    #ax = plt.subplot(111)
    ymin =  np.inf
    ymax = -np.inf
    # 1-2-3sigma plots
    if kll is not None:
        ax.fill_between(xvals, lo3, hi3, facecolor="#d9ecff", 
                        edgecolor="#d9ecff", label="$99.73\%$ region")
        ax.fill_between(xvals, lo2, hi2, facecolor="#C0DFFF", 
                        edgecolor="#C0DFFF", label="$95.45\%$ region")
        ax.fill_between(xvals, lo1, hi1, facecolor="cornflowerblue", 
                        edgecolor="cornflowerblue", label="$68.27\%$ region")
        ymin = np.amin([ymin, lo3.min()-0.01*lo3.min()])
        ymax = np.amax([ymax, hi3.max()+0.01*hi3.min()])
    # Best fit plot
    ax.scatter( xax, bestfit, c="k", label="Best fit", zorder=90, 
                lw=1, s=6)
    ax.errorbar(xax, data, yerr=uncert, xerr=xerr, 
                fmt="or", markersize=1.5, capsize=1.5, elinewidth=1, 
                ecolor='tab:red', label="Data", zorder=80)
    ymin = np.amin([ymin, data.min()-pad*uncert.max(), bestfit.min()-pad*uncert.max()])
    ymax = np.amax([ymax, data.max()+pad*uncert.max(), bestfit.max()+pad*uncert.max()])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(ymin, ymax)
    plt.xlim(*xlims)
    ax.set_ylabel(r""+ylabel, fontsize=12)
    ax.set_xscale('log')
    for label in ax.xaxis.get_ticklabels(which='both'):
        label.set_visible(False)
    # Title
    if title and bestpars is not None:
        if ndec is not None:
            truernd = np.array([np.around(truepars[i], ndec[i]) 
                                for i in range(len(truepars))])
            bestrnd = np.array([np.around(bestpars[i], ndec[i]) 
                                for i in range(len(bestpars))])
        else:
            truernd = truepars.copy()
            bestrnd = bestpars.copy()
        if type(title) == str:
            titlestr = title + '\n'
        else:
            titlestr = ''
        if truepars is not None:
            titlestr = titlestr + 'True parameters: ' + ', '.join(list(truernd.astype(str))) + '\n'
        titlestr = titlestr + 'Best-fit parameters: ' + ', '.join(list(bestrnd.astype(str)))
        plt.title(titlestr, fontsize=10)
    # Residuals
    ax2 = fig1.add_axes((.1, .1, .8, .2))
    resid  = data - bestfit
    ax2.scatter(xax, resid, s=0.8, label='Full resolution', c='b')
    plt.hlines(0, xlims[0], xlims[1])
    yticks = ax2.yaxis.get_major_ticks()
    #yticks[-1].label.set_visible(False)
    ax2.set_ylabel('Residuals', fontsize=12)
    ylims = plt.ylim()
    ax2.set_xlabel(r""+xlabel, fontsize=12)
    ax2.set_xscale('log')
    formatter = tck.FuncFormatter(lambda y, _: '{:g}'.format(y))
    ax2.get_xaxis().set_major_formatter(formatter)
    ax2.get_xaxis().set_minor_formatter(formatter)
    # Prevent overlapping xaxis tick labels
    fig1.canvas.draw()
    lbls = []
    for label in ax2.xaxis.get_ticklabels(minor=True):
        lbls.append(label.get_text())
    lbls = np.asarray(lbls).astype(float)
    loglbls = np.round(np.log10(lbls))
    if loglbls[-1] - loglbls[0] >= 2 and loglbls[-1] - loglbls[0] <= 5:
        for label in ax2.xaxis.get_ticklabels(minor=True):
            digit = int(label.get_text().replace('0','').replace('.','')[0])
            if digit%2 == 1:
                label.set_visible(False)
    elif loglbls[-1] - loglbls[0] > 5:
        for label in ax2.xaxis.get_ticklabels(minor=True):
            digit = int(label.get_text().replace('0','').replace('.','')[0])
            if digit%5 > 0:
                label.set_visible(False)
    plt.xlim(*xlims)
    # Histogram of residuals
    '''
    ax3 = fig1.add_axes((0.9, .1, .1, .2))
    plt.hist(resid[np.abs(resid)!=np.inf], density=True, 
             orientation="horizontal")
    plt.xlabel('PDF', fontsize=12)
    plt.ylim(*ylims)
    plt.yticks(visible=False)
    plt.setp(ax3.get_xticklabels()[0], visible=False)
    '''
    plt.savefig(outputdir+'bestfit_spectrum.png', bbox_inches='tight')
    plt.close()



"""
Module that contains functions to overplot posteriors.

comp_histogram: Plots the probability density functions of 1D marginalized 
                posteriors for two MCMC runs.

comp_pairwise: Plots the probability density functions of 2D pairwise 
               posteriors for two MCMC runs.

comp_pt: Plots the explored temperature--pressure profiles for two MCMC runs.

"""

import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import plotter as P


def comp_histogram(stack1, stack2, name1, name2, 
                   parname=None, 
                   fignum=-12, fs=30, savefile=None, bins=60):
    """
    Plots the probability density functions of 1D marginalized posteriors 
    for two MCMC runs.
    
    Inputs
    ------
    stack1  : array.  Posterior, shaped (nparams, niterations)
    stack2  : array.  Same as `stack1`, but for the posterior to be compared.
    name1   : string. Label name for `stack1`
    name2   : string. Label name for `stack2`
    parname : list, strings. Parameter names.
    fs      : int.    Font size for plots.
    savefile: string. Path/to/file where the plot will be saved.
    bins    : int.    Number of bins for the histograms.
    """
    if np.shape(stack1)[0] != np.shape(stack2)[0]:
        raise ValueError('The posteriors must have the same ' + \
                         'number of parameters.')
    npars, niter1 = np.shape(stack1)
    npars, niter2 = np.shape(stack2)

    # Set default parameter names:
    if parname is None:
        namelen = int(2+np.log10(np.amax([npars-1,1])))
        parname = np.zeros(npars, "<U%d"%namelen)
        for i in np.arange(npars):
            parname[i] = "P" + str(i).zfill(namelen-1)

    # Set number of rows:
    if npars < 10:
        nperrow = 3
    else:
        nperrow = 4
    nrows = (npars - 1)//nperrow + 1
    # Set number of columns:
    if   npars > 9:
        ncolumns = 4
    elif npars > 4:
        ncolumns = 3
    else:
        ncolumns = (npars+2)//3 + (npars+2)%3  # (Trust me!)

    histheight = 4 + 4*(nrows)
    if nrows == 1:
        bottom = 0.25
    else:
        bottom = 0.15

    fig = plt.figure(fignum, figsize=(18, histheight))
    plt.clf()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=bottom, top=0.9,
                        hspace=0.8, wspace=0.25)

    for i in np.arange(npars):
        ax = plt.subplot(nrows, ncolumns, i+1)
        a  = plt.xticks(size=fs-4, rotation=90)
        a  = plt.yticks(size=fs-4)
        if i%ncolumns == 0:
            plt.ylabel('Normalized PDF', size=fs)
        plt.xlabel(parname[i], size=fs)
        rng   = min(stack1[i].min(), stack2[i].min()), \
                max(stack1[i].max(), stack2[i].max())
        a = plt.hist(stack1[i], bins, range=rng, alpha=0.5, label=name1, density=True, 
                     color='b')
        a = plt.hist(stack2[i], bins, range=rng, alpha=0.5, label=name2, density=True, 
                     color='r')
        if i == npars - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={"size":fs})
    fig.align_labels()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    plt.close()


def comp_pairwise(stack1, stack2, name1, name2, 
                  parname=None, 
                  fignum=-11, fs=24, savefile=None):
    """
    Plots the probability density functions of 2D pairwise posteriors for two 
    MCMC runs.
    
    Inputs
    ------
    stack1  : array.  Posterior, shaped (nparams, niterations)
    stack2  : array.  Same as `stack1`, but for the posterior to be compared.
    name1   : string. Label name for `stack1`
    name2   : string. Label name for `stack2`
    parname : list, strings. Parameter names.
    fignum  : int.    Figure number.
    fs      : int.    Font size for plots.
    savefile: string. Path/to/file where the plot will be saved.
    """
    if np.shape(stack1)[0] != np.shape(stack2)[0]:
        raise ValueError('The posteriors must have the same ' + \
                         'number of parameters.')
    npars, niter1 = np.shape(stack1)
    npars, niter2 = np.shape(stack2)

    # Don't plot if there are no pairs:
    if npars == 1:
        return

    # Set default parameter names:
    if parname is None:
        namelen = int(2+np.log10(np.amax([npars-1,1])))
        parname = np.zeros(npars, "<U%d"%namelen)
        for i in np.arange(npars):
            parname[i] = "P" + str(i).zfill(namelen-1)

    # Colors for plotting
    n        = 256
    cmap     = mpl.cm.get_cmap('bwr', n)
    # Blues for HOMER
    palette1 = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.4, b=0),
            cmap(np.linspace(0.4, 0, n)))
    palette1.set_under(alpha=0.0)
    palette1.set_bad(alpha=0.0)
    # Reds for other code
    palette2 = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.6, b=1.0),
            cmap(np.linspace(0.6, 1, n)))
    palette2.set_under(alpha=0.0)
    palette2.set_bad(alpha=0.0)

    # Make the figure, then plot
    fig = plt.figure(fignum, figsize=(1.5*npars, 1.5*npars))
    plt.clf()
    h = 1 # Subplot index
    plt.subplots_adjust(left  =0.15, right =0.85, bottom=0.15, top=0.85,
                        hspace=0.3,  wspace=0.3)

    for     j in np.arange(npars): # Rows
        for i in np.arange(npars): # Columns
            if j > i or j == i:
                a = plt.subplot(npars, npars, h)
                # Y labels:
                if i == 0 and j != 0:
                    plt.yticks(size=fs-4)
                    plt.ylabel(parname[j], size=fs+4, multialignment='center')
                elif i == 0 and j == 0:
                    plt.yticks(visible=False)
                    plt.ylabel(parname[j], size=fs+4, multialignment='center')
                else:
                    a = plt.yticks(visible=False)
                # X labels:
                if j == npars-1:
                    plt.xticks(size=fs-4, rotation=90)
                    plt.xlabel(parname[i], size=fs+4)
                else:
                    a = plt.xticks(visible=False)
                # The plot:
                if j > i:
                    # HOMER
                    hist2d, xedges, yedges = np.histogram2d(stack1[i],
                                                            stack1[j], 
                                                            20, density=True)
                    vmin = 0.0
                    hist2d[np.where(hist2d == 0)] = np.nan
                    a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], 
                                                     yedges[0], yedges[-1]), 
                                   cmap=palette1, vmin=vmin, aspect='auto',
                                   origin='lower', interpolation='bilinear', 
                                   alpha=0.5)
                    # Other code
                    hist2d, xedges, yedges = np.histogram2d(stack2[i],
                                                            stack2[j], 
                                                            20, density=True)
                    vmin = 0.0
                    hist2d[np.where(hist2d == 0)] = np.nan
                    a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], 
                                                     yedges[0], yedges[-1]), 
                                   cmap=palette2, vmin=vmin, aspect='auto',
                                   origin='lower', interpolation='bilinear', 
                                   alpha=0.5)
                else:
                    a = plt.hist(stack1[i], 20, color='b', label=name1, 
                                 alpha=0.5, density=True)
                    a = plt.hist(stack2[i], 20, color='r', label=name2, 
                                 alpha=0.5, density=True)
                    if i == 0 and j == 0:
                        a = plt.gca()
                        a.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                 prop={"size":fs-4})
                a = plt.gca()
                # Make sure ticks are read-able
                if   len(a.get_xticks()[::2]) > 4:
                    a.set_xticks(a.get_xticks()[::3])
                elif len(a.get_xticks()[::2]) > 2:
                    a.set_xticks(a.get_xticks()[::2])
                if   len(a.get_yticks()[::2]) > 4:
                    a.set_yticks(a.get_yticks()[::3])
                elif len(a.get_yticks()[::2]) > 2:
                    a.set_yticks(a.get_yticks()[::2])

            h += 1
    fig.align_labels()
    # The colorbar:
    if npars > 2:
        a = plt.subplot(2, 6, 5, frameon=False)
        a.yaxis.set_visible(False)
        a.xaxis.set_visible(False)
    bounds = np.linspace(0, 1.0, 64)
    norm   = mpl.colors.BoundaryNorm(bounds, palette1.N)
    ax2    = fig.add_axes([0.7, 0.45, 0.025, 0.36])
    cb     = mpl.colorbar.ColorbarBase(ax2, cmap=palette1, norm=norm,
                                       spacing='proportional', 
                                       boundaries=bounds, format='%.1f')
    cb.ax.set_yticklabels([])
    ax2    = fig.add_axes([0.75, 0.45, 0.025, 0.36])
    cb     = mpl.colorbar.ColorbarBase(ax2, cmap=palette2, norm=norm,
                                       spacing='proportional', 
                                       boundaries=bounds, format='%.1f')
    cb.set_label("Normalized Point Density", fontsize=fs)
    cb.set_ticks(np.linspace(0, 1, 5))
    plt.draw()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    plt.close()


def comp_PT(pressure, stack1, stack2, name1, name2, 
            PTargs, fs=26, savefile=None):
    """
    Plots the median, 1sigma, and 2sigma regions for the temperature--pressure 
    profiles explored for two MCMC runs.
    
    Inputs
    ------
    pressure: array.  Pressures associated with each layer of the atmosphere.
    stack1  : array.  Posterior, shaped (nparams, niterations)
    stack2  : array.  Same as `stack1`, but for the posterior to be compared.
    name1   : string. Label name for `stack1`
    name2   : string. Label name for `stack2`
    PTargs  : list.   Path to .txt file containing values necessary to 
                      calculate the temperature--pressure profile.
                      Currently, only option is Line et al. (2013) method.
                      Format: R_star (meters), T_star (Kelvin), T_int (Kelvin),
                              sma    (meters), grav (cm s-2)
    fs      : int.    Font size for plots.
    savefile: string. Path/to/file where the plot will be saved.
    """
    # Calculate PT profiles
    pt1 = np.zeros((np.shape(stack1)[1], len(pressure)))
    pt2 = np.zeros((np.shape(stack2)[1], len(pressure)))
    for i in np.arange(0, np.shape(stack1)[1]):
        PTparams = np.concatenate((stack1[:,i], PTargs))
        pt1[i]   = P.PT_line(pressure, *PTparams)
    for i in np.arange(0, np.shape(stack2)[1]):
        PTparams = np.concatenate((stack2[:,i], PTargs))
        pt2[i]   = P.PT_line(pressure, *PTparams)

    lo1_1sig = np.percentile(pt1, 15.87, axis=0)
    hi1_1sig = np.percentile(pt1, 84.13, axis=0)
    lo1_2sig = np.percentile(pt1,  2.28, axis=0)
    hi1_2sig = np.percentile(pt1, 97.72, axis=0)
    med1     = np.median(    pt1,        axis=0)

    lo2_1sig = np.percentile(pt2, 15.87, axis=0)
    hi2_1sig = np.percentile(pt2, 84.13, axis=0)
    lo2_2sig = np.percentile(pt2,  2.28, axis=0)
    hi2_2sig = np.percentile(pt2, 97.72, axis=0)
    med2     = np.median(    pt2,        axis=0)

    # plot figure
    plt.figure(2, dpi=300)
    plt.clf()
    ax1=plt.subplot(111)
    ax1.fill_betweenx(pressure, lo1_2sig, hi1_2sig, facecolor="#62B1FF", 
                      label=name1+' 2$\sigma$', alpha=0.5, edgecolor="0.5")
    ax1.fill_betweenx(pressure, lo1_1sig, hi1_1sig, facecolor="#1873CC",
                      label=name1+' 1$\sigma$', alpha=0.5, edgecolor="#1873CC")
    plt.semilogy(med1, pressure, "-", lw=2, label=name1+' Median',color="k")
    ax1.fill_betweenx(pressure, lo2_2sig, hi2_2sig, facecolor="#ff6a62", 
                      label=name2+' 2$\sigma$', alpha=0.5, edgecolor="0.5")
    ax1.fill_betweenx(pressure, lo2_1sig, hi2_1sig, facecolor="#cc3318",
                      label=name2+' 1$\sigma$', alpha=0.5, edgecolor="#cc3318")
    plt.semilogy(med2, pressure, "-", lw=2, label=name2+' Median',color="k", 
                 ls='--')
    plt.ylim(pressure[0], pressure[-1])
    plt.legend(loc="best")
    plt.xlabel("Temperature  (K)", size=15)
    plt.ylabel("Pressure  (bar)",  size=15)
    plt.gca().invert_yaxis()

    # save figure
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()



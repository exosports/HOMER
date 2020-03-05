"""
Makes plots based on the MCMC posterior.  Adapted from MCcubed:
# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

trace: plots histories of MCMC parameters.

pairwise: plots the 2D pairwise posteriors for each combination of parameters.

histogram: Plots the 1D marginalized posteriors for each parameter.

mcmc_pt: Plots the temperature--pressure profiles explored by the MCMC.

PT_line: computes the temperature--pressure profile according to 
         Line et al. (2013)

xi: used by PT_line to compute T(p) profile

"""
import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp


__all__ = ["trace", "pairwise", "histogram", "mcmc_pt", "PT_line", "xi"]


def trace(allparams, title=None, parname=None, thinning=1,
          fignum=-10, savefile=None, fmt=".", sep=None, fs=14):
  """
  Plot parameter trace MCMC sampling

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  fmt: String
     The format string for the line and marker.
  sep: Integer
     Number of samples per chain. If not None, draw a vertical line
     to mark the separation between the chains.

  Uncredited developers
  ---------------------
  Kevin Stevenson (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Get location for chains separations:
  xmax = len(allparams[0,0::thinning])
  if sep is not None:
    xsep = np.arange(sep/thinning, xmax, sep/thinning)

  # Make the trace plot:
  plt.figure(fignum, figsize=(18, npars))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=16)

  plt.subplots_adjust(left=0.15, right=0.95, bottom=0.10, top=0.90,
                      hspace=0.15)

  for i in np.arange(npars):
    a = plt.subplot(npars, 1, i+1)
    plt.plot(allparams[i, 0::thinning], fmt)
    yran = a.get_ylim()
    if sep is not None:
      plt.vlines(xsep, yran[0], yran[1], "0.3")
    plt.xlim(0, xmax)
    plt.ylim(yran)
    plt.ylabel(parname[i], size=fs, multialignment='center')
    plt.yticks(size=fs)
    if i == npars - 1:
      plt.xticks(size=fs)
      if thinning > 1:
        plt.xlabel('MCMC (thinned) iteration', size=fs)
      else:
        plt.xlabel('MCMC iteration', size=fs)
    else:
      plt.xticks(visible=False)
    # Align labels
    a.yaxis.set_label_coords(-0.05, 0.5)

  if savefile is not None:
    plt.savefig(savefile)


def pairwise(allparams, title=None, parname=None, thinning=1,
             fignum=-11, savefile=None, style="hist", fs=14):
  """
  Plot parameter pairwise posterior distributions

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  style: String
     Choose between 'hist' to plot as histogram, or 'points' to plot
     the individual points.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  Ryan Hardy  (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Don't plot if there are no pairs:
  if npars == 1:
    return

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set palette color:
  palette = mpl.cm.get_cmap('YlOrRd', 256)
  palette.set_under(alpha=0.0)
  palette.set_bad(alpha=0.0)

  fig = plt.figure(fignum, figsize=(18, 18))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=16)

  h = 1 # Subplot index
  plt.subplots_adjust(left=0.15,   right=0.95, bottom=0.15, top=0.9,
                      hspace=0.05, wspace=0.05)

  for   j in np.arange(npars): # Rows
    for i in np.arange(npars):  # Columns
      if j > i or j == i:
        a = plt.subplot(npars, npars, h)
        # Y labels:
        if i == 0 and j != 0:
          plt.yticks(size=fs-4)
          plt.ylabel(parname[j], size=fs, multialignment='center')
        elif i == 0 and j == 0:
          plt.yticks(visible=False)
          plt.ylabel(parname[j], size=fs, multialignment='center')
        else:
          a = plt.yticks(visible=False)
        # X labels:
        if j == npars-1:
          plt.xticks(size=fs-4, rotation=90)
          plt.xlabel(parname[i], size=fs)
        else:
          a = plt.xticks(visible=False)
        # The plot:
        if style=="hist":
          if j > i:
            hist2d, xedges, yedges = np.histogram2d(allparams[i, 0::thinning],
                                                    allparams[j, 0::thinning], 
                                                    20, normed=False)
            vmin = 0.0
            hist2d[np.where(hist2d == 0)] = np.nan
            a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], yedges[0],
                           yedges[-1]), cmap=palette, vmin=vmin, aspect='auto',
                           origin='lower', interpolation='bilinear')
          else:
            a = plt.hist(allparams[i,0::thinning], 20, normed=False)
        elif style=="points":
          if j > i:
            a = plt.plot(allparams[i], allparams[j], ",")
          else:
            a = plt.hist(allparams[i,0::thinning], 20, normed=False)

        # Make sure ticks are readable
        plt.draw()
        a = plt.gca()
        if len(a.xaxis.get_ticklabels()) > 2:
          if len(a.xaxis.get_ticklabels()) > 3:
            a.xaxis.get_ticklabels()[-1].set_visible(False)
          a.xaxis.get_ticklabels()[0].set_visible(False)
        if len(a.yaxis.get_ticklabels()) > 2:
          if len(a.yaxis.get_ticklabels()) > 3:
            a.yaxis.get_ticklabels()[-1].set_visible(False)
          a.yaxis.get_ticklabels()[0].set_visible(False)

        # Align labels
        if j == npars-1 and i == npars-1:
          axs = fig.get_axes()
          for ax in axs:
            ss = ax.get_subplotspec()
            nrows, ncols, start, stop = ss.get_geometry()
            if start//nrows == nrows-1:
              ax.xaxis.set_label_coords(0.5, -npars/20)
            if start%ncols == 0:
              ax.yaxis.set_label_coords(-npars/20, 0.5)
      h += 1
  # The colorbar:
  if style == "hist":
    if npars > 2:
      a = plt.subplot(2, 6, 5, frameon=False)
      a.yaxis.set_visible(False)
      a.xaxis.set_visible(False)
    bounds = np.linspace(0, 1.0, 64)
    norm = mpl.colors.BoundaryNorm(bounds, palette.N)
    ax2 = fig.add_axes([0.85, 0.535, 0.025, 0.36])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
          spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Normalized point density", fontsize=fs)
    cb.set_ticks(np.linspace(0, 1, 5))
    plt.draw()

  # Save file:
  if savefile is not None:
    plt.savefig(savefile)


def histogram(allparams, title=None, parname=None, thinning=1,
              fignum=-12, savefile=None):
  """
  Plot parameter marginal posterior distributions

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)
  fs = 14  # Fontsize

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set number of rows:
  if npars < 10:
    nrows = (npars - 1)/3 + 1
  else:
    nrows = (npars - 1)/4 + 1
  # Set number of columns:
  if   npars > 9:
    ncolumns = 4
  elif npars > 4:
    ncolumns = 3
  else:
    ncolumns = (npars+2)/3 + (npars+2)%3  # (Trust me!)

  histheight = 4 + 4*(nrows)
  if nrows == 1:
    bottom = 0.25
  else:
    bottom = 0.15

  plt.figure(fignum, figsize=(18, histheight))
  plt.clf()
  plt.subplots_adjust(left=0.1, right=0.95, bottom=bottom, top=0.9,
                      hspace=0.4, wspace=0.1)

  if title is not None:
    a = plt.suptitle(title, size=16)

  maxylim = 0  # Max Y limit
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    a  = plt.xticks(size=fs, rotation=90)
    if i%ncolumns == 0:
      a = plt.yticks(size=fs)
    else:
      a = plt.yticks(visible=False)
    plt.xlabel(parname[i], size=fs)
    a = plt.hist(allparams[i,0::thinning], 20, normed=False)
    maxylim = np.amax((maxylim, ax.get_ylim()[1]))

  # Set uniform height:
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    ax.set_ylim(0, maxylim)

  if savefile is not None:
    plt.savefig(savefile)


def mcmc_pt(outp, pressure, PTargs, 
            savefile=None):
    """
    Computes the median, 1sigma from median, and 2sigma from median PT profiles.
    """
    # Calculate PT profiles
    PTprofiles = np.zeros((np.shape(outp)[1], len(pressure)))
    for i in np.arange(0, np.shape(outp)[1]):
        PTparams      = np.concatenate((outp[:,i], PTargs))
        PTprofiles[i] = PT_line(pressure, *PTparams)

    low1   = np.percentile(PTprofiles, 16.0, axis=0)
    hi1    = np.percentile(PTprofiles, 84.0, axis=0)
    low2   = np.percentile(PTprofiles,  2.5, axis=0)
    hi2    = np.percentile(PTprofiles, 97.5, axis=0)
    median = np.median(    PTprofiles,       axis=0)

    # plot figure
    plt.figure(2, dpi=300)
    plt.clf()
    ax1=plt.subplot(111)
    ax1.fill_betweenx(pressure, low2, hi2, facecolor="#62B1FF", 
                      label='2$\sigma$', edgecolor="0.5")
    ax1.fill_betweenx(pressure, low1, hi1, facecolor="#1873CC",
                      label='1$\sigma$', edgecolor="#1873CC")
    plt.semilogy(median, pressure, "-", lw=2, label='Median',color="k")
    plt.ylim(pressure[0], pressure[-1])
    plt.legend(loc="best")
    plt.xlabel("Temperature  (K)", size=15)
    plt.ylabel("Pressure  (bar)",  size=15)
    plt.gca().invert_yaxis()

    # save figure
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def PT_line(pressure, kappa,  gamma1, gamma2, alpha, beta, 
            R_star,   T_star, T_int,  sma,    grav):
  '''
  Copied from BART/code/PT.py

  Generates a PT profile based on input free parameters and pressure array.
  If no inputs are provided, it will run in demo mode, using free
  parameters given by the Line 2013 paper and some dummy pressure
  parameters.
  Inputs
  ------
  pressure: 1D float ndarray
     Array of pressure values in bars.
  kappa : float, in log10. Planck thermal IR opacity in units cm^2/gr
  gamma1: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
  gamma2: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
  alpha : float.           Visible-stream partition (0.0--1.0).
  beta  : float.           A 'catch-all' for albedo, emissivity, and day-night
                           redistribution (on the order of unity)
  R_star: Float
     Stellar radius (in meters).
  T_star: Float
     Stellar effective temperature (in Kelvin degrees).
  T_int:  Float
     Planetary internal heat flux (in Kelvin degrees).
  sma:    Float
     Semi-major axis (in meters).
  grav:   Float
     Planetary surface gravity (at 1 bar) in cm/second^2.
  Returns
  -------
  T: temperature array
  Example:
  --------
  >>> import PT as pt
  >>> import scipy.constants as sc
  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>> Rsun = 6.995e8 # Sun radius in meters
  >>> # Pressure array (bars):
  >>> p = np.logspace(2, -5, 100)
  >>> # Physical (fixed for each planet) parameters:
  >>> Ts = 5040.0        # K
  >>> Ti =  100.0        # K
  >>> a  = 0.031 * sc.au # m
  >>> Rs = 0.756 * Rsun  # m
  >>> g  = 2192.8        # cm s-2
  >>> # Fitting parameters:
  >>> kappa  = -1.5   # log10(3e-2)
  >>> gamma1 = -0.8   # log10(0.158)
  >>> gamma2 = -0.8   # log10(0.158)
  >>> alpha  = 0.5
  >>> beta   = 1.0
  >>> T0 = pt.PT(p, kappa, gamma1, gamma2, alpha, beta, Rs, Ts, Ti, a, g)
  >>> plt.figure(1)
  >>> plt.clf()
  >>> plt.semilogy(T0, p, lw=2, color="b")
  >>> plt.ylim(p[0], p[-1])
  >>> plt.xlim(800, 2000)
  >>> plt.xlabel("Temperature  (K)")
  >>> plt.ylabel("Pressure  (bars)")
  Developers:
  -----------
  Madison Stemm      astromaddie@gmail.com
  Patricio Cubillos  pcubillos@fulbrightmail.org
  Modification History:
  ---------------------
  2014-09-12  Madison   Initial version, adapted from equations (13)-(16)
                        in Line et al. (2013), Apj, 775, 137.
  2014-12-10  patricio  Reviewed and updated code.
  2015-01-22  patricio  Receive log10 of free parameters now.
  2019-02-13  mhimes    Replaced `params` arg with each parameter for 
                        consistency with other PT models
  '''

  # Convert kappa, gamma1, gamma2 from log10
  kappa  = 10**(kappa )
  gamma1 = 10**(gamma1)
  gamma2 = 10**(gamma2)

  # Stellar input temperature (at top of atmosphere):
  T_irr = beta * (R_star / (2.0*sma))**0.5 * T_star

  # Gray IR optical depth:
  tau = kappa * (pressure*1e6) / grav # Convert bars to barye (CGS)

  xi1 = xi(gamma1, tau)
  xi2 = xi(gamma2, tau)

  # Temperature profile (Eq. 13 of Line et al. 2013):
  temperature = (0.75 * (T_int**4 * (2.0/3.0 + tau) +
                         T_irr**4 * (1-alpha) * xi1 +
                         T_irr**4 * alpha     * xi2 ) )**0.25

  return temperature


def xi(gamma, tau):
  """
  Copied from BART/code/PT.py

  Calculate Equation (14) of Line et al. (2013) Apj 775, 137
  Parameters:
  -----------
  gamma: Float
     Visible-to-thermal stream Planck mean opacity ratio.
  tau: 1D float ndarray
     Gray IR optical depth.
  Modification History:
  ---------------------
  2014-12-10  patricio  Initial implemetation.
  """
  return (2.0/3) * \
         (1 + (1./gamma) * (1 + (0.5*gamma*tau-1)*np.exp(-gamma*tau)) +
          gamma*(1 - 0.5*tau**2) * sp.expn(2, gamma*tau)              )









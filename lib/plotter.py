"""
Makes plots based on the MCMC posterior.  Adapted from BART.

pt_post: Plots the temperature--pressure profiles explored by the sampler.

PT_line: computes the temperature--pressure profile according to 
         Line et al. (2013)

xi: used by PT_line to compute T(p) profile

"""
import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp


__all__ = ["pt_post", "PT_line", "xi"]


def pt_post(outp, pressure, PTargs, 
            savefile=None):
    """
    Computes the median, 1sigma from median, and 2sigma from median PT profiles.
    """
    # Calculate PT profiles
    PTprofiles = np.zeros((np.shape(outp)[1], len(pressure)))
    for i in np.arange(0, np.shape(outp)[1]):
        PTparams      = np.concatenate((outp[:,i], PTargs))
        PTprofiles[i] = PT_line(pressure, *PTparams)

    low1   = np.percentile(PTprofiles, 15.87, axis=0)
    hi1    = np.percentile(PTprofiles, 84.13, axis=0)
    low2   = np.percentile(PTprofiles,  2.28, axis=0)
    hi2    = np.percentile(PTprofiles, 97.72, axis=0)
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

  Calculate Equation (14) of Line et al. (2013) ApJ 775, 137
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
          gamma*(1 - 0.5*tau**2) * sp.expn(2, gamma*tau)               )


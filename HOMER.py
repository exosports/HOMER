#!/usr/bin/env python3
'''
Driver for HOMER
'''


import sys, os
import configparser
import importlib
import numpy as np

import keras
from keras import backend as K

libdir = os.path.dirname(__file__) + '/lib'
sys.path.append(libdir)

import NN
import utils   as U
import mcplots as P

mc3dir = os.path.dirname(__file__) + '/modules/MCcubed'
sys.path.append(mc3dir)
mcdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed'
sys.path.append(mcdir)
mcmcdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed/mc'
sys.path.append(mcmcdir)
import MCcubed as mc3
#import mc as mc3

def HOMER(cfile):
    """
    Main driver for the software.

    Inputs
    ------
    cfile : path/to/configuration file.

    Examples
    --------
    See config.cfg in the top-level directory.
    Run it from a terminal like
      user@machine:~/dir/to/HOMER$ ./HOMER.py config.cfg 
    """
    # Load configuration file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(cfile, 'r'))

    # Run everything specified in config file
    for section in config:
        if section != "DEFAULT":
            conf = config[section]
            ### Unpack the variables ###
            # Top-level params
            onlyplot  = conf.getboolean("onlyplot")
            normalize = conf.getboolean("normalize")
            scale     = conf.getboolean("scale")

            # Directories
            inputdir  = os.path.join(os.path.abspath(conf["inputdir" ]), '')
            outputdir = os.path.join(os.path.abspath(conf["outputdir"]), '')
            # Create the output directory if it does not exist
            U.make_dir(outputdir)

            # Data & model info
            if conf["data"][-4:] == '.npy':
                data    = np.load(inputdir + conf["data"])
            else:
                data    = np.array([float(num)                      \
                                    for num in conf["data"].split()])
            if conf["uncert"][-4:] == '.npy':
                uncert  = np.load(inputdir + conf["uncert"])
            else:
                uncert  = np.array([float(num)                        \
                                    for num in conf["uncert"].split()])
            if conf["filters"] != 'None':
                filters = conf["filters"].split()
                filt2um = float(conf["filt2um"])
            else:
                filters = None
                filt2um = 1


            if conf["starspec"] != 'None':
                if not os.path.isabs(conf["starspec"]):
                    starspec = inputdir + conf["starspec"]
                else:
                    starspec = conf["starspec"]
            else:
                starspec = None

            if conf["starspec"] != 'None':
                if not os.path.isabs(conf["starspec"]):
                    starspec = np.load(inputdir + conf["starspec"])
                else:
                    starspec = np.load(conf["starspec"])
            else:
                starspec = None

            try:
                factor = np.load(conf["factor"])
            except:
                try:
                    factor = np.load(inputdir + conf["factor"])
                except:
                    factor = float(conf["factor"])
            

            if not os.path.isabs(conf["weight_file"]):
                weight_file = inputdir + conf["weight_file"]
            else:
                weight_file = conf["weight_file"]
            inD         = conf.getint("input dim")
            outD        = conf.getint("output dim")
            olog        = conf.getboolean("olog")
            xvals       = np.load(conf["xvals"])
            wnfact      = float(conf["wnfact"])
            xlabel      = conf["xval label"]
            ylabel      = conf["yval label"]
            fmean       = conf["fmean"]
            fstdev      = conf["fstdev"]
            fmin        = conf["fmin"]
            fmax        = conf["fmax"]
            convlayers     = conf["convlayers"]
            if convlayers != "None":
                convlayers =  [int(num) for num in convlayers.split()]
                conv       = True
            else:
                convlayers = None
                conv       = False
            layers        =  [int(num) for num in conf["layers"].split()]

            # Plotting parameters
            if conf['pnames'] == '':
                pnames = None
            else:
                pnames = conf['pnames'].split()
            savefile = conf['savefile']
            if savefile != '':
                savefile = savefile + '_'

            # MCMC params
            if conf['flog'] != '' and conf['flog'] != 'None':
                if os.path.isabs(conf['flog']):
                    flog = conf['flog']
                else:
                    flog = outputdir + conf['flog']
            else:
                flog = None
            func     = conf["func"].split()
            evalfunc = importlib.import_module(func[1]).__getattribute__(func[0])
            pinit    = np.array([float(val) for val in conf["pinit"].split()])
            pmin     = np.array([float(val) for val in conf["pmin" ].split()])
            pmax     = np.array([float(val) for val in conf["pmax" ].split()])
            pstep    = np.array([float(val) for val in conf["pstep"].split()])
            niter    = int(conf.getfloat("niter"))
            burnin   = conf.getint("burnin")
            nchains  = conf.getint("nchains")
            thinning = conf.getint("thinning")

            # Check sizes
            if np.any(np.array([len(pinit), len(pstep), 
                                len(pmin),  len(pmax)  ]) != inD):
                print("One or more MCMC parameters (inital, min, max, step) ")
                print("do not match the dimensionality of the input for " + \
                      "the model.")
                print("Fix this and try again.")
                print('Input dimensionality:', inD)
                print('Lengths:')
                print('  pinit:', len(pinit))
                print('  pstep:', len(pstep))
                print('  pmin :', len(pmin))
                print('  pmax :', len(pmax))
                sys.exit()

            # Get stats about data for normalization/scaling
            if normalize:
                try:
                    mean   = np.load(inputdir+fmean)
                    stdev  = np.load(inputdir+fstdev)
                    x_mean = mean [:inD]
                    x_std  = stdev[:inD]
                    y_mean = mean [inD:]
                    y_std  = stdev[inD:]
                except:
                    print("HOMER requires the mean and standard deviation ")
                    print("of the training set used to train the ML model.")
                    print("These should be 1D arrays of the inputs followed " + \
                          "by the outputs.")
                    print("Update the path(s) and try again.")
                    sys.exit()
            else:
                x_mean = 0.
                x_std  = 1.
                y_mean = 0.
                y_std  = 1.
            if scale:
                try:
                    datmin    = np.load(inputdir + fmin)
                    datmax    = np.load(inputdir + fmax)
                    x_min     = datmin[:inD]
                    x_max     = datmax[:inD]
                    y_min     = datmin[inD:]
                    y_max     = datmax[inD:]
                    scalelims = [int(num) 
                                 for num in conf["scalelims"].split(',')]
                    # Check that the MCMC min/max are within the data set range
                    if np.any(x_min > pmin):
                        print("One or more minimum values for MCMC params " + \
                              "are less than the corresponding")
                        print("training data minimum.")
                        print("Fix this and try again.")
                        sys.exit()
                    if np.any(x_max < pmax):
                        print("One or more maximum values for MCMC params " + \
                              "are more than the corresponding")
                        print("training data maximum.")
                        print("Fix this and try again.")
                        sys.exit()
                    if normalize:
                        x_min = U.normalize(x_min, x_mean, x_std)
                        x_max = U.normalize(x_max, x_mean, x_std)
                        y_min = U.normalize(y_min, y_mean, y_std)
                        y_max = U.normalize(y_max, y_mean, y_std)
                except:
                    print("Error loading the training set min/max arrays.")
                    print("In the config file, scaling was indicated.")
                    print("Update the path(s) or change `scale` to False " + \
                          "and try again.")
                    sys.exit()
            else:
                x_min     =  0.
                x_max     =  1.
                y_min     =  0.
                y_max     =  1.
                scalelims = [0., 1.]

            if not onlyplot:
                # Instantiate model
                print('Building model...')
                nn = NN.NNModel(weight_file)

                # Run the MCMC
                if flog is not None:
                    logfile = open(flog, 'w')
                else:
                    logfile = None
                outp, bestp = mc3.mc.mcmc(data, uncert, func=evalfunc, 
                                          indparams=[nn, 
                                                     x_mean, x_std, 
                                                     y_mean, y_std, 
                                                     x_min, x_max, 
                                                     y_min, y_max, 
                                                     scalelims, 
                                                     xvals, wnfact,
                                                     starspec, factor, 
                                                     filters, filt2um, 
                                                     conv, olog],
                                          parnames=pnames, params=pinit, 
                                          pmin=pmin, pmax=pmax, stepsize=pstep,
                                          numit=niter, burnin=burnin, 
                                          nchains=nchains, 
                                          walk='snooker', hsize=2*nchains, 
                                          plots=False, leastsq=False, 
                                          log=logfile, 
                                          savefile=outputdir +             \
                                                   savefile  + 'output.npy')
                if flog is not None:
                    logfile.close()
                # NOTE! hsize has the hard-coded value to ensure that the 
                # parameter space is adequately seeded with models to speed up 
                # exploration
            else:
                print('Remaking plots...')
                # Load posterior, shape (nchains, nfree, niterperchain)
                outpc = np.load(outputdir+savefile+'output.npy')
                # Stack it to be (nfree, niter), remove burnin
                outp = outpc[0, :, burnin:]
                for c in range(1, nchains):
                    outp = np.hstack((outp, outpc[c, :, burnin:]))

            # Make plots
            P.trace(outp, parname=pnames, thinning=thinning, 
                    sep=np.size(outp[0]//nchains), 
                    savefile=outputdir+savefile+"MCMC_trace.png")
            P.histogram(outp, parname=pnames, thinning=thinning, 
                        savefile=outputdir+savefile+"MCMC_posterior.png")
            P.pairwise(outp, parname=pnames, thinning=thinning, 
                       savefile=outputdir+savefile+"MCMC_pairwise.png")

    return


if __name__ == "__main__":
    try:
        HOMER(*sys.argv[1:])
    except MemoryError:
        sys.stderr.write('\nERROR: Memory limit exceeded.')
        sys.exit(1)






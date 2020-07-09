#!/usr/bin/env python3
'''
Driver for HOMER
'''


import sys, os
import configparser
import importlib
import pickle
import numpy as np
import scipy.interpolate as si

import keras
from keras import backend as K

libdir = os.path.dirname(__file__) + '/lib'
sys.path.append(libdir)

import NN
import bestfit   as BF
import compost   as C
import utils     as U
import mcmcplots as P

mc3dir = os.path.dirname(__file__) + '/modules/MCcubed'
sys.path.append(mc3dir)
mccdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed'
sys.path.append(mccdir)
mcpdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed/plots'
sys.path.append(mcpdir)
import mcplots as mcp
mcmcdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed/mc'
sys.path.append(mcmcdir)
import MCcubed as mc3

try:
    import datasketches as ds
    print('Using the datasketches package to calculate ' + \
          'spectra quantiles, if requested.')
except:
    print('datasketches package is not available.  Will ' + \
          'not calculate spectra quantiles.')


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
            onlyplot   = conf.getboolean("onlyplot")
            compost    = conf.getboolean("compost")
            normalize  = conf.getboolean("normalize")
            scale      = conf.getboolean("scale")
            plot_PT    = conf.getboolean("plot_PT")
            quantiles  = conf.getboolean("quantiles")

            # Directories
            inputdir  = os.path.join(os.path.abspath(conf["inputdir" ]), '')
            outputdir = os.path.join(os.path.abspath(conf["outputdir"]), '')
            # Create the output directory if it does not exist
            U.make_dir(outputdir)

            # Data & model info
            if conf["data"][-4:] == '.npy':
                if os.path.isabs(conf["data"]):
                    data = np.load(conf["data"])
                else:
                    data = np.load(inputdir + conf["data"])
            else:
                data = np.array([float(num)                      \
                                 for num in conf["data"].split()])
            if conf["uncert"][-4:] == '.npy':
                if os.path.isabs(conf["uncert"]):
                    uncert = np.load(conf["uncert"])
                else:
                    uncert = np.load(inputdir + conf["uncert"])
            else:
                uncert = np.array([float(num)                        \
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

            if conf["factor"] != 'None':
                if conf["factor"][-4:] == '.npy':
                    if os.path.isabs(conf["factor"]):
                        factor = np.load(conf["factor"])
                    else:
                        factor = np.load(inputdir + conf["factor"])
                else:
                    try:
                        factor = float(conf["factor"])
                    except:
                        factor = 1.
            else:
                factor = None

            if conf["PTargs"] == 'None' or conf["PTargs"] == '':
                PTargs = []
            else:
                if conf["PTargs"][-4:] == '.txt':
                    if os.path.isabs(conf["PTargs"]):
                        PTargs = np.loadtxt(conf["PTargs"])
                    else:
                        PTargs = np.loadtxt(inputdir + conf["PTargs"])
                elif conf["PTargs"][-4:] == '.npy':
                    if os.path.isabs(conf["PTargs"]):
                        PTargs = np.load(conf["PTargs"])
                    else:
                        PTargs = np.load(inputdir + conf["PTargs"])
                else:
                    PTargs = [float(num) for num in conf["PTargs"].split()]

            if not os.path.isabs(conf["weight_file"]):
                weight_file = inputdir + conf["weight_file"]
            else:
                weight_file = conf["weight_file"]

            inD    = conf.getint("inD")
            outD   = conf.getint("outD")

            if conf["ilog"] in ["True", "False"]:
                ilog = conf.getboolean("ilog")
            elif conf["ilog"] in ["None", "none", ""]:
                ilog = False
            elif ',' in conf["ilog"]:
                ilog = [int(num) for num in conf["ilog"].split(',')]
            else:
                ilog = [int(num) for num in conf["ilog"].split()]

            if conf["olog"] in ["True", "False"]:
                olog = conf.getboolean("olog")
            elif conf["olog"] in ["None", "none", ""]:
                olog = False
            elif ',' in conf["olog"]:
                olog = [int(num) for num in conf["olog"].split(',')]
            else:
                olog = [int(num) for num in conf["olog"].split()]

            if os.path.isabs(conf["xvals"]):
                xvals = np.load(conf["xvals"])
            else:
                xvals = np.load(inputdir + conf["xvals"])
            wn     = conf.getboolean("wn")
            wnfact = float(conf["wnfact"])
            xlabel = conf["xlabel"]
            ylabel = conf["ylabel"]
            fmean  = conf["fmean"]
            fstdev = conf["fstdev"]
            fmin   = conf["fmin"]
            fmax   = conf["fmax"]
            if plot_PT:
                fpress = conf["fpress"]

            # Plotting parameters
            if conf['pnames'] == '':
                pnames = None
            else:
                pnames = conf['pnames'].split()
            if conf['postshift'] == '' or conf['postshift'] == 'None':
                postshift = None
            else:
                postshift = np.array([float(val) 
                                      for val in conf['postshift'].split()])
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

            f = conf["func"].split()
            if len(f) == 3:
                sys.path.append(f[2])
            func     = importlib.import_module(f[1]).__getattribute__(f[0])
            pinit    = np.array([float(val) for val in conf["pinit"].split()])
            pmin     = np.array([float(val) for val in conf["pmin" ].split()])
            pmax     = np.array([float(val) for val in conf["pmax" ].split()])
            pstep    = np.array([float(val) for val in conf["pstep"].split()])
            niter    = int(conf.getfloat("niter"))
            burnin   = conf.getint("burnin")
            nchains  = conf.getint("nchains")
            thinning = conf.getint("thinning")
            try:
                perc = np.array([float(val) for val in conf["perc"].split()])
            except:
                perc = np.array([0.6827, 0.9545, 0.9973])

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
                        print("Indices:", np.where(x_min > pmin)[0])
                        sys.exit()
                    if np.any(x_max < pmax):
                        print("One or more maximum values for MCMC params " + \
                              "are more than the corresponding")
                        print("training data maximum.")
                        print("Fix this and try again.")
                        print("Indices:", np.where(x_max < pmax)[0])
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

            if filters is not None:
                # Load filters
                filttran = []
                ifilt = np.zeros((len(filters), 2), dtype=int)
                meanwn = []
                for i in range(len(filters)):
                    datfilt = np.loadtxt(filters[i])
                    # Convert filter wavelenths to microns, then convert um -> cm-1
                    finterp = si.interp1d(10000. / (filt2um * datfilt[:,0]), 
                                          datfilt[:,1],
                                          bounds_error=False, fill_value=0)
                    # Interpolate and normalize
                    tranfilt = finterp(xvals)
                    tranfilt = tranfilt / np.trapz(tranfilt, xvals)
                    meanwn.append(np.sum(xvals*tranfilt)/sum(tranfilt))
                    # Find non-zero indices for faster integration
                    nonzero = np.where(tranfilt!=0)
                    ifilt[i, 0] = max(nonzero[0][ 0] - 1, 0)
                    ifilt[i, 1] = min(nonzero[0][-1] + 1, len(xvals)-1)
                    filttran.append(tranfilt[ifilt[i,0]:ifilt[i,1]]) # Store filter

                meanwn = np.asarray(meanwn)
            else:
                ifilt    = None
                filttran = None
                meanwn   = None

            ### Check if datasketches is available ###
            if ds and quantiles:
                # FINDME Hard-coded 1000 for accuracy. Change to config option?
                kll = ds.vector_of_kll_floats_sketches(1000, outD)
            else:
                kll = None

            # Save file names
            fsavefile  = outputdir + savefile + 'output.npy'
            fsavemodel = outputdir + savefile + 'output_model.npy'
            fsavesks   = outputdir + 'sketches.pickle'
            if not onlyplot:
                # Instantiate model
                print('\nBuilding model...\n')
                nn = NN.NNModel(weight_file)

                # Run the MCMC
                if flog is not None:
                    logfile = open(flog, 'w')
                else:
                    logfile = None
                count = np.array([0]) # to determine when to begin updating sketches
                outp, bestp = mc3.mc.mcmc(data, uncert, func=func, 
                                          indparams=[nn, 
                                                     x_mean, x_std, 
                                                     y_mean, y_std, 
                                                     x_min, x_max, 
                                                     y_min, y_max, 
                                                     scalelims, 
                                                     xvals*wnfact,
                                                     starspec, factor, 
                                                     filttran, ifilt, 
                                                     ilog, olog, 
                                                     kll, count, burnin],
                                          parnames=pnames, params=pinit, 
                                          pmin=pmin, pmax=pmax, stepsize=pstep,
                                          numit=niter, burnin=burnin, 
                                          nchains=nchains, 
                                          walk='snooker', hsize=4*nchains, 
                                          plots=False, leastsq=False, 
                                          log=logfile, 
                                          savefile =fsavefile,
                                          savemodel=fsavemodel)
                if flog is not None:
                    logfile.close()
                # NOTE! hsize has the hard-coded value to ensure that the 
                # parameter space is adequately seeded with models to speed up 
                # exploration

                # Serialize and save the sketches, in case needing to replot
                if kll is not None:
                    sers = kll.serialize()
                    pickle.dump(sers, open(fsavesks, 'wb'))
            else:
                print('Remaking plots...')
                if kll is not None:
                    try:
                        desers = pickle.load(open(fsavesks, 'rb'))
                        for i in range(len(desers)):
                            kll.deserialize(desers[i], i)
                    except:
                        print("No sketch file found.  Will not plot quantiles.")
                # Load posterior, shape (nchains, nfree, niterperchain)
                outpc = np.load(fsavefile)
                # Stack it to be (nfree, niter), remove burnin
                outp = outpc[0, :, burnin:]
                for c in range(1, nchains):
                    outp = np.hstack((outp, outpc[c, :, burnin:]))
                del outpc

            # Load the evaluated models & stack, excluding burnin
            evalmodels = np.load(fsavemodel)
            allmodel   = evalmodels[0,:,burnin:]
            for c in range(1, nchains):
                allmodel = np.hstack((allmodel, evalmodels[c, :, burnin:]))

            # Plot best-fit model
            print("\nPlotting best-fit model...\n")
            bestfit = BF.get_bestfit(allmodel, outp, flog, pstep>0)
            BF.plot_bestfit(outputdir, xvals, data, uncert, meanwn, ifilt, 
                            bestfit, xlabel, ylabel, kll, wn)

            # Shift posterior params, if needed (e.g., for units)
            if postshift is not None:
                outp += np.expand_dims(postshift, -1)

            # Make plots of posterior
            print('Making plots of the posterior...\n')
            pnames = np.asarray(pnames)
            mcp.trace(outp, parname=pnames[pstep>0], thinning=thinning, 
                      sep=np.size(outp[0]//nchains), 
                      savefile=outputdir+savefile+"MCMC_trace.png")
            mcp.histogram(outp, parname=pnames[pstep>0], thinning=thinning, 
                          savefile=outputdir+savefile+"MCMC_posterior.png")
            mcp.pairwise(outp, parname=pnames[pstep>0], thinning=thinning, 
                         savefile=outputdir+savefile+"MCMC_pairwise.png")

            # PT profiles
            if plot_PT:
                print("Plotting PT profiles...\n")
                pressure = np.loadtxt(inputdir + fpress, skiprows=1)[:,1]
                P.mcmc_pt(outp[:5], pressure, PTargs, 
                          savefile=outputdir+savefile+"MCMC_PT.png")

            # Format parameter names, and find maximum length
            parname = []
            pnlen   = 0
            for i in range(len(pnames)):
                if pstep[i] <= 0:
                    continue
                parname.append(pnames[i].replace('$', '').replace('\\', '').\
                                         replace('_', '').replace('^' , '').\
                                         replace('{', '').replace('}' , ''))
                pnlen = max(pnlen, len(parname[-1]))

            # Compare the posterior to another result
            if compost:
                print('Making comparison plots of posteriors...\n')
                compfile = conf["compfile"]
                compname = conf["compname"]
                if not os.path.isabs(conf["compsave"]):
                    compsave = outputdir + conf["compsave"]
                else:
                    compsave = conf["compsave"]
                if conf["compshift"] == '' or conf["compshift"] == 'None':
                    compshift = None
                else:
                    compshift = np.array([float(val) 
                                          for val in conf["compshift"].split()])
                # Load posterior and stack chains
                cpost  = np.load(compfile)
                cstack = cpost[0, :, burnin:]
                for c in np.arange(1, cpost.shape[0]):
                    cstack = np.hstack((cstack, cpost[c, :, burnin:]))
                if compshift is not None:
                    cstack += np.expand_dims(compshift, -1)
                # Make comparison plot
                C.comp_histogram(outp, cstack, 
                                 'HOMER', compname, 
                                 pnames, 
                                 savefile=compsave+'_hist.png')
                print('Bhattacharyya coefficients:')
                bhatchar = np.zeros(sum(pstep>0))
                for n in range(sum(pstep>0)):
                    rng   = min(outp[n].min(), cstack[n].min()), \
                            max(outp[n].max(), cstack[n].max())
                    hist1 = np.histogram(outp[n],   density=False, bins=60, 
                                         range=rng)[0]/outp[n].shape[0]
                    hist2 = np.histogram(cstack[n], density=False, bins=60, 
                                         range=rng)[0]/cstack[n].shape[0]
                    bhatchar[n] = np.sum(np.sqrt(hist1 * hist2))
                    print('  '+parname[n].ljust(pnlen, ' ') + ': ' + \
                          str(bhatchar[n]))
                    n += 1
                print('  '+'Mean'.ljust(pnlen, ' ')+':', np.mean(bhatchar))
                np.save(outputdir+'bhatchar.npy', bhatchar)
                if plot_PT:
                    C.comp_PT(pressure, outp[:5], cstack[:5], 'HOMER', compname,
                              PTargs, savefile=compsave+'_PT.png')
                C.comp_pairwise(outp, cstack, 
                                'HOMER', compname, 
                                pnames, 
                                savefile=compsave+'_pair.png')

    return


if __name__ == "__main__":
    try:
        HOMER(*sys.argv[1:])
    except MemoryError:
        sys.stderr.write('\nERROR: Memory limit exceeded.')
        sys.exit(1)






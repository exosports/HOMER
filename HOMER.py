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
import bestfit    as BF
import compost    as C
import credregion as CR
import utils      as U
import mcplots    as P

mc3dir = os.path.dirname(__file__) + '/modules/MCcubed'
sys.path.append(mc3dir)
mcdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed'
sys.path.append(mcdir)
mcmcdir = os.path.dirname(__file__) + '/modules/MCcubed/MCcubed/mc'
sys.path.append(mcmcdir)
import MCcubed as mc3
# NOTE: datasketches imported later, so that users are not forced to use it
#       see just before the MCMC is called


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
            credregion = conf.getboolean("credregion")
            compost    = conf.getboolean("compost")
            normalize  = conf.getboolean("normalize")
            scale      = conf.getboolean("scale")

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

            if conf["PTargs"] == 'None' or conf["PTargs"] == '':
                PTargs = []
            else:
                if   conf["PTargs"][-4:] == '.txt':
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

            inD    = conf.getint("input dim")
            outD   = conf.getint("output dim")
            ilog   = conf.getboolean("ilog")
            olog   = conf.getboolean("olog")
            xvals  = np.load(conf["xvals"])
            wnfact = float(conf["wnfact"])
            xlabel = conf["xval label"]
            ylabel = conf["yval label"]
            fmean  = conf["fmean"]
            fstdev = conf["fstdev"]
            fmin   = conf["fmin"]
            fmax   = conf["fmax"]
            fpress = conf["fpress"]

            convlayers     = conf["convlayers"]
            if convlayers == "None" or convlayers == '':
                convlayers = None
                conv       = False
            else:
                convlayers =  [int(num) for num in convlayers.split()]
                conv       = True

            layers =  [int(num) for num in conf["layers"].split()]

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

            ### Check if datasketches is available ###
            try:
                from datasketches import kll_floatarray_sketches
                print('Using the datasketches package to calculate ' + \
                      'spectra quantiles.')
                kll = kll_floatarray_sketches(10000, outD)
            except:
                print('datasketches package is not available.  Will ' + \
                      'not calculate spectra quantiles.')
                kll = None

            # Save file names
            fsavefile  = outputdir + savefile + 'output.npy'
            fsavemodel = outputdir + savefile + 'output_model.npy'
            fsavesks   = outputdir + 'sketches.pickle'
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
                                                     xvals*wnfact,
                                                     starspec, factor, 
                                                     filttran, ifilt, 
                                                     conv, ilog, olog, kll],
                                          parnames=pnames, params=pinit, 
                                          pmin=pmin, pmax=pmax, stepsize=pstep,
                                          numit=niter, burnin=burnin, 
                                          nchains=nchains, 
                                          walk='snooker', hsize=2*nchains, 
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
            if credregion:
                print('Calculating effective sample size...\n')
                fess = open(outputdir + 'ess.txt', 'w')
                speis, ess = CR.ess(outpc[:, :, burnin:])
                print('SPEIS:', speis)
                print('ESS  :', ess)
                fess.write('SPEIS:' + str(speis))
                fess.write('ESS  :' + str(ess))
                siggy = CR.sig(ess, 
                               p_est=perc)
                for i in range(len(perc)):
                    print('  ' + str(perc[i]) + u"\u00B1" + str(siggy[i]))
                    fess.write('  ' + str(perc[i]) + u"\u00B1" + str(siggy[i]))
                fess.close()

            # Stack it to be (nfree, niter), remove burnin
            outp = outpc[0, :, burnin:]
            for c in range(1, nchains):
                outp = np.hstack((outp, outpc[c, :, burnin:]))

            # Load the evaluated models & stack, excluding burnin
            evalmodels = np.load(fsavemodel)
            allmodel   = evalmodels[0,:,burnin:]
            for c in range(1, nchains):
                allmodel = np.hstack((allmodel, evalmodels[c, :, burnin:]))

            # Plot best-fit model
            bestfit = BF.get_bestfit(allmodel, outp, flog)
            BF.plot_bestfit(outputdir, xvals, data, uncert, meanwn, ifilt, bestfit, kll)

            # Shift posterior params, if needed (e.g., for units)
            if postshift is not None:
                outp += np.expand_dims(postshift, -1)

            # Make plots of posterior
            print('\nMaking plots of the posterior...\n')
            P.trace(outp, parname=pnames, thinning=thinning, 
                    sep=np.size(outp[0]//nchains), 
                    savefile=outputdir+savefile+"MCMC_trace.png")
            P.histogram(outp, parname=pnames, thinning=thinning, 
                        savefile=outputdir+savefile+"MCMC_posterior.png")
            P.pairwise(outp, parname=pnames, thinning=thinning, 
                       savefile=outputdir+savefile+"MCMC_pairwise.png")

            # PT profiles
            pressure = np.loadtxt(inputdir + fpress, skiprows=1)[:,1]
            P.mcmc_pt(outp[:5], pressure, PTargs, 
                      savefile=outputdir+savefile+"MCMC_PT.png")

            # Format parameter names, and find maximum length
            parname = []
            pnlen   = 0
            for i in range(len(pnames)):
                parname.append(pnames[i].replace('$', '').replace('\\', '').\
                                         replace('_', '').replace('^' , ''))
                pnlen = max(pnlen, len(parname[i]))
            # Compute credible regions
            if credregion:
                print('Calculating credible regions...\n')
                fcred = open(outputdir + 'credregion.txt', 'w')
                for n in range(outp.shape[0]):
                    pdf, xpdf, CRlo, CRhi = CR.credregion(outp[n], perc)
                    creg = [' U '.join(['({:10.4e}, {:10.4e})'.format(
                                        CRlo[j][k], CRhi[j][k])
                                        for k in range(len(CRlo[j]))])
                            for j in range(len(CRlo))]
                    print(parname[n] + " credible region:")
                    fcred.write(parname[n] + " credible region:")
                    for p in range(len(perc)):
                        print("  " + str(100*perc[p]) + "%: " + str(creg[p]))
                        fcred.write("  " + str(100*perc[p]) + "%: " + str(creg[p]))
                fcred.close()

            # Compare the posterior to another result
            if compost:
                print('\nMaking comparison plots of posteriors...')
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
                print('Plotting histograms of 1D posteriors...')
                C.comp_histogram(outp, cstack, 
                                 'HOMER', compname, 
                                 pnames, 
                                 savefile=compsave+'_hist.png')
                print('Bhattacharyya coefficients:')
                bhatchar = np.zeros(len(pnames))
                for i in range(len(pnames)):
                    rng   = min(outp[i].min(), cstack[i].min()), max(outp[i].max(), cstack[i].max())
                    hist1 = np.histogram(outp[i],   density=False, bins=60, range=rng)[0]/outp[i].shape[0]
                    hist2 = np.histogram(cstack[i], density=False, bins=60, range=rng)[0]/cstack[i].shape[0]
                    bhatchar[i] = np.sum(np.sqrt(hist1 * hist2))
                    print('  '+parname[i].ljust(pnlen, ' ')+': '+str(bhatchar[i]))
                print('  '+'Mean'.ljust(pnlen, ' ')+':', np.mean(bhatchar))
                np.save(outputdir+'bhatchar.npy', bhatchar)
                print('Plotting PT profiles...')
                C.comp_PT(pressure, outp[:5], cstack[:5], 'HOMER', compname, 
                          PTargs, savefile=compsave+'_PT.png')
                print('Plotting pairwise posteriors...')
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






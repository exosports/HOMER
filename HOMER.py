#!/usr/bin/env python3
'''
Driver for HOMER
'''


import sys, os
import configparser
import importlib
import copy
import pickle
import functools
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
import plotter   as P

moddir = os.path.dirname(__file__) + '/modules'
lisadir = moddir + '/LISA/'
sys.path.append(lisadir)
import LISA

mcpdir = os.path.join(lisadir, 'modules', 'MCcubed', 'MCcubed', 'plots')
sys.path.append(mcpdir)
import mcplots as mcp

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import datasketches as ds
    print('Using the datasketches package to calculate ' + \
          'spectra quantiles, if requested.')
except:
    ds = False
    print('datasketches package is not available.  Will ' + \
          'not calculate spectra quantiles.')


# Global variable for plotting T(p) posteriors, 
# since only Line et al. (2013) is supported currently
nPT = 5


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
            alg        = conf["alg"]
            onlyplot   = conf.getboolean("onlyplot")
            compost    = conf.getboolean("compost")
            normalize  = conf.getboolean("normalize")
            scale      = conf.getboolean("scale")
            plot_PT    = conf.getboolean("plot_PT")
            if "fext" in conf.keys():
                fext   = conf["fext"]
            else:
                fext   = ".png"
            quantiles  = conf.getboolean("quantiles")
            try:
                title  = conf.getboolean("title")
            except:
                title  = False
            if "ndec" in conf.keys():
                ndec = [int(val) for val in conf["ndec"].split()]
            else:
                ndec = None

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

            if "filters" not in conf.keys():
                filters = None
                filtconv = 1
            elif conf["filters"] == 'None':
                filters = None
                filtconv = 1
            else:
                filters = conf["filters"].split()
                filtconv = float(conf["filtconv"])

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

            if conf["ilog"] in ["True", "true", "T", "False", "false", "F"]:
                ilog = conf.getboolean("ilog")
            elif conf["ilog"] in ["None", "none", ""]:
                ilog = False
            elif conf["ilog"].isdigit():
                ilog = int(conf["ilog"])
            elif any(pun in conf["ilog"] for pun in [",", " ", "\n"]):
                if "," in conf["ilog"]:
                    ilog = [int(num) for num in conf["ilog"].split(',')]
                else:
                    ilog = [int(num) for num in conf["ilog"].split()]
                if any(num >= inD for num in ilog):
                    raise ValueError("One or more ilog indices exceed the " + \
                                     "specified number of inputs.")
            else:
                raise ValueError("ilog specification not understood.")

            if conf["olog"] in ["True", "true", "T", "False", "false", "F"]:
                olog = conf.getboolean("olog")
            elif conf["olog"] in ["None", "none", ""]:
                olog = False
            elif conf["olog"].isdigit():
                olog = int(conf["olog"])
            elif any(pun in conf["olog"] for pun in [",", " ", "\n"]):
                if "," in conf["olog"]:
                    olog = [int(num) for num in conf["olog"].split(',')]
                else:
                    olog = [int(num) for num in conf["olog"].split()]
                if any(num >= outD for num in olog):
                    raise ValueError("One or more olog indices exceed the " + \
                                     "specified number of outputs.")
            else:
                raise ValueError("olog specification not understood.")

            if os.path.isabs(conf["xvals"]):
                xvals = np.load(conf["xvals"])
            else:
                xvals = np.load(inputdir + conf["xvals"])
            if "wn" in conf.keys():
                wn = conf.getboolean("wn")
            else:
                wn = True
            if "wnfact" in conf.keys():
                wnfact = float(conf["wnfact"])
            else:
                wnfact = 1.
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

            if 'postshift' not in conf.keys():
                postshift = None
            elif conf['postshift'] in ['', 'None', 'none', 'False', 'false', 'F']:
                postshift = None
            elif 'norm' in conf['postshift']:
                postshift = conf['postshift']
            else:
                try:
                    postshift = np.array([float(val) 
                                          for val in conf['postshift'].split()])
                except:
                    raise ValueError("Invalid specification for postshift.")

            if "savefile" in conf.keys():
                savefile = conf['savefile']
                if savefile != '':
                    savefile = savefile + '_'
            else:
                savefile = ''

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
            elif len(f) > 3:
                raise ValueError("The 'func' parameter can have 3 elements "   \
                               + "at most. Given "+str(len(f))+" elements:\n" \
                               + str(f))
            func  = importlib.import_module(f[1]).__getattribute__(f[0])

            pinit = np.array([float(val) for val in conf["pinit"].split()])
            pmin  = np.array([float(val) for val in conf["pmin" ].split()])
            pmax  = np.array([float(val) for val in conf["pmax" ].split()])
            pstep = np.array([float(val) for val in conf["pstep"].split()])

            niter = int(conf.getfloat("niter"))

            if "burnin" in conf.keys():
                burnin = conf.getint("burnin")
            else:
                burnin = 0

            if "nchains" in conf.keys():
                nchains = conf.getint("nchains")
            else:
                nchains = 1

            if "thinning" not in conf.keys():
                thinning = 1
            elif conf["thinning"] in ["", "None", "none", "False", "false"]:
                thinning = 1
            else:
                thinning = conf.getint("thinning")
                if thinning < 1:
                    print("Nonphysical thinning value provided (<1). " + \
                          "Setting to 1.")
                    thinning = 1

            try:
                perc = np.array([float(val) for val in conf["perc"].split()])
            except:
                perc = np.array([0.6827, 0.9545, 0.9973])

            # Get the true parameters, if given
            if "truepars" not in conf.keys():
                truepars = None
            elif conf["truepars"] in ["", "None", "none", "False", "false"]:
                truepars = None
            else:
                if '.npy' in conf["truepars"]:
                    truepars = np.load(conf["truepars"])
                else:
                    truepars = np.array([float(val) 
                                         for val in conf["truepars"].split()])
                if len(truepars) != inD:
                    raise ValueError("Number of true parameter values "    + \
                                     "given does not match the number of " + \
                                     "inputs.")
                if ilog:
                    truepars[ilog] = np.log10(truepars[ilog])
                truepars = truepars[pstep>0]

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
                if wn:
                    xwn = xvals
                else:
                    xwn = wnfact / (filtconv * xvals)
                for i in range(len(filters)):
                    datfilt = np.loadtxt(filters[i])
                    if wn:
                        finterp = si.interp1d(datfilt[:,0], 
                                              datfilt[:,1],
                                              bounds_error=False, fill_value=0)
                    else:
                        # Convert filter x-values, then convert to inverse space
                        finterp = si.interp1d(wnfact / (filtconv * datfilt[:,0]), 
                                              datfilt[:,1],
                                              bounds_error=False, fill_value=0)
                    # Interpolate and normalize
                    tranfilt = finterp(xwn)
                    tranfilt = tranfilt / np.abs(np.trapz(tranfilt, xwn))
                    meanwn.append(np.sum(xwn*tranfilt)/sum(tranfilt))
                    # Find non-zero indices for faster integration
                    nonzero = np.where(tranfilt!=0)
                    ifilt[i, 0] = max(nonzero[0][ 0] - 1, 0)
                    ifilt[i, 1] = min(nonzero[0][-1] + 2, len(xwn)-1)
                    filttran.append(tranfilt[ifilt[i,0]:ifilt[i,1]]) # Store filter

                meanwave = np.asarray(meanwn)
                if not wn:
                    meanwave = 10000./meanwave
            else:
                ifilt    = None
                filttran = None
                meanwave = None
                xwn      = xvals

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
            fposterior = outputdir + 'output_posterior.npy'
            fbestp     = outputdir + 'output_bestp.npy'

            # Instantiate model
            print('\nBuilding model...\n')
            nn = NN.NNModel(weight_file)

            # Pack the parameters
            if alg in ['snooker', 'demc']:
                model = functools.partial(func, nn=nn, 
                                                ilog=ilog, olog=olog, 
                                                x_mean=x_mean, x_std=x_std, 
                                                y_mean=y_mean, y_std=y_std, 
                                                x_min=x_min, x_max=x_max, 
                                                y_min=y_min, y_max=y_max, 
                                                scalelims=scalelims, xvals=xwn,
                                                filters=filttran, ifilt=ifilt, 
                                                starspec=starspec, 
                                                factor=factor)
                count = np.array([0]) # to determine when to update sketches
                indparams = [nn, 
                             x_mean, x_std, y_mean, y_std, 
                             x_min, x_max, y_min, y_max, scalelims, 
                             xwn, starspec, factor, 
                             filttran, ifilt, ilog, olog, 
                             kll, count, burnin]
                params = {"data"      : data     , "uncert"     : uncert, 
                          "func"      : func     , "indparams"  : indparams, 
                          "pnames"    : pnames   , "pinit"      : pinit, 
                          "pmin"      : pmin     , "pmax"       : pmax, 
                          "pstep"     : pstep    , "niter"      : niter, 
                          "burnin"    : burnin   , "thinning"   : thinning, 
                          "nchains"   : nchains  , "hsize"      : 4*nchains, 
                          "savefile"  : savefile , "outputdir"  : outputdir, 
                          "fsavefile" : fsavefile, "fsavemodel" : fsavemodel, 
                          "flog"      : flog}

            elif alg in ['multinest', 'ultranest']:
                burnin  = 0
                nchains = 1
                # Set static variables for `func`
                model = functools.partial(func, nn=nn, inD=inD, 
                                                pstep=pstep, pinit=pinit, 
                                                ilog=ilog, olog=olog, 
                                                x_mean=x_mean, x_std=x_std, 
                                                y_mean=y_mean, y_std=y_std, 
                                                x_min=x_min, x_max=x_max, 
                                                y_min=y_min, y_max=y_max, 
                                                scalelims=scalelims, 
                                                xvals=xwn, 
                                                filters=filttran, 
                                                ifilt=ifilt, 
                                                starspec=starspec, 
                                                factor=factor)

                pr = importlib.import_module(f[1]).__getattribute__('prior')
                ll = importlib.import_module(f[1]).__getattribute__('loglikelihood')

                prior   = functools.partial(pr, pmin=pmin, pmax=pmax, 
                                                pstep=pstep)
                loglike = functools.partial(ll, data=data, uncert=uncert, 
                                                nn=nn, inD=inD, 
                                                pstep=pstep, pinit=pinit, 
                                                ilog=ilog, olog=olog, 
                                                x_mean=x_mean, x_std=x_std, 
                                                y_mean=y_mean, y_std=y_std, 
                                                x_min=x_min, x_max=x_max, 
                                                y_min=y_min, y_max=y_max, 
                                                scalelims=scalelims, 
                                                xvals=xwn, 
                                                filters=filttran, 
                                                ifilt=ifilt)
                prior.__name__ = 'prior'
                loglike.__name__ = 'loglike'
                params = {"prior"     : prior    , "loglike" : loglike, 
                          "pnames"    : pnames   , "pstep"   : pstep, 
                          "outputdir" : outputdir, "kll"     : kll, 
                          "model"     : model}

            if not onlyplot:
                # Call LISA
                outp, bestp = LISA.run(alg, params)

                # Save out the arrays
                np.save(fposterior, outp )
                np.save(fbestp    , bestp)

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
                outp  = np.load(fposterior)
                bestp = np.load(fbestp)

            bestfit = model(bestp)

            # Plot best-fit model
            print("\nPlotting best-fit model...\n")
            BF.plot_bestfit(outputdir, xvals, data, uncert, meanwave, ifilt, 
                            bestfit, xlabel, ylabel, kll=kll, 
                            bestpars=bestp, truepars=truepars, 
                            title=title, ndec=ndec, fext=fext)

            # Shift posterior params, if needed (e.g., for units)
            if postshift is not None:
                if type(postshift) == str:
                    if 'norm' in postshift:
                        # Get indices to normalize
                        ibeg = int(postshift.split('_')[-1].split('-')[0])
                        iend = int(postshift.split('_')[-1].split('-')[1]) + 1
                        # Adjust if there are static params
                        istatic = np.arange(len(pnames))[pstep==0]
                        for val in istatic:
                            if val < ibeg:
                                ibeg -= 1
                            if val < iend:
                                iend -= 1
                        # Adjust posterior
                        outp[ibeg:iend] = np.log10(10**outp[ibeg:iend] /       \
                                            np.sum(10**outp[ibeg:iend], axis=0))
                    else:
                        raise Exception("Unknown postshift specification.")
                else:
                    outp += np.expand_dims(postshift, -1)

            # Make plots of posterior
            print('Making plots of the posterior...\n')
            pnames = np.asarray(pnames)
            mcp.trace(outp, parname=pnames[pstep>0], thinning=thinning, 
                      sep=np.size(outp[0]//nchains), 
                      savefile=outputdir+savefile+"LISA_trace"+fext, 
                      truepars=truepars)
            mcp.histogram(outp, parname=pnames[pstep>0], thinning=thinning, 
                          savefile=outputdir+savefile+"LISA_posterior"+fext, 
                          truepars=truepars, credreg=True, ptitle=title)
            mcp.pairwise(outp, parname=pnames[pstep>0], thinning=thinning, 
                         savefile=outputdir+savefile+"LISA_pairwise"+fext, 
                         truepars=truepars, credreg=True, ptitle=title, 
                         ndec=ndec)

            # PT profiles
            if plot_PT:
                print("Plotting PT profiles...\n")
                pressure  = np.loadtxt(inputdir + fpress, skiprows=1)[:,1]
                presspost = np.zeros((nPT, outp.shape[-1]))
                ifixd = np.arange(nPT)[(pstep<=0)[:nPT]]
                istep = np.arange(nPT)[(pstep> 0)[:nPT]]
                #"none" expands axis, ensures it works if no params are fixed
                presspost[ifixd] = pinit[ifixd, None]
                presspost[istep] = outp [istep]
                P.pt_post(presspost, pressure, PTargs, 
                          savefile=outputdir+savefile+"LISA_PT"+fext)

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
                if "compburn" not in conf.keys():
                    compburn = 0
                else:
                    compburn = conf.getint("compburn")
                if not os.path.isabs(conf["compsave"]):
                    compsave = outputdir + conf["compsave"]
                else:
                    compsave = conf["compsave"]
                if "compshift" not in conf.keys():
                    compshift = None
                elif conf["compshift"] in ['','None','none','False','false']:
                    compshift = None
                else:
                    compshift = np.array([float(val) 
                                          for val in conf["compshift"].split()])
                # Load posterior and stack chains
                cpost  = np.load(compfile)
                cstack = cpost[0, :, compburn:]
                for c in np.arange(1, cpost.shape[0]):
                    cstack = np.hstack((cstack, cpost[c, :, compburn:]))
                if compshift is not None:
                    cstack += np.expand_dims(compshift, -1)
                # Make comparison plot
                C.comp_histogram(outp, cstack, 
                                 'HOMER', compname, 
                                 np.asarray(pnames)[pstep>0], 
                                 savefile=compsave+"_hist"+fext)
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
                    if 'cinit' not in conf.keys() and \
                       np.amin(np.arange(len(pinit))[pstep<=0]) < nPT:
                        print("To plot a comparison of T(p) posteriors with " +\
                              "fixed values, `cinit` must be\n"   +\
                              "specified in the configuration file.")
                    elif len(istep) == nPT:
                        # No fixed T(p) parameters
                        C.comp_PT(pressure, presspost, cstack[:nPT], 'HOMER', 
                                  compname, PTargs, savefile=compsave+"_PT"+fext)
                    else:
                        cinit = np.array([float(val) for val in conf["cinit"].split()])
                        cprespost = np.zeros((nPT, cstack.shape[-1]))
                        cprespost[ifixd] = cinit [ifixd]
                        cprespost[istep] = cstack[istep]
                        C.comp_PT(pressure, presspost, cprespost, 'HOMER', 
                                  compname, PTargs, savefile=compsave+"_PT"+fext)
                C.comp_pairwise(outp, cstack, 
                                'HOMER', compname, 
                                np.asarray(pnames)[pstep>0], 
                                savefile=compsave+"_pair"+fext)

    return


if __name__ == "__main__":
    try:
        HOMER(*sys.argv[1:])
    except MemoryError:
        sys.stderr.write('\nERROR: Memory limit exceeded.')
        sys.exit(1)






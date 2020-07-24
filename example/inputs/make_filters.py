#! /usr/bin/env python

import sys, os
import numpy as np

nbins = 46
xvals = np.load('xvals.npy')
xvals_bin = 0
for i in range(nbins):
    xvals_bin += xvals[i::nbins]
    if i==0:
        left = xvals[i::nbins]
    elif i==nbins-1:
        rght = xvals[i::nbins]

xvals_bin /= nbins
np.save('xvals_binned.npy', xvals_bin)

# Create the filters
for i in range(len(left)):
    with open('./filters/'+str(i+1).zfill(2)+'.dat', 'w') as foo:
        foo.write(str(left[i]-1e-10)+' 0.0\n')
        foo.write(str(left[i]) +' 1.0\n')
        foo.write(str((left[i]+rght[i])/2.) +' 1.0\n')
        foo.write(str(rght[i]) +' 1.0\n')
        foo.write(str(rght[i]+1e-10)+' 0.0\n')


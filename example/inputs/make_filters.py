#! /usr/bin/env python
"""
Script to produce a lower-resolution grid from high-res NN output.
"""
import sys, os
import numpy as np

# Target number of bins
nbins = 46
# Full-resolution grid of NN output
xvals = np.load('xvals.npy')
# Compute the value of each bin, as well as the left- and right-most values
xvals_bin = 0
for i in range(nbins):
    xvals_bin += xvals[i::nbins]
    if i==0:
        left = xvals[i::nbins]
    elif i==nbins-1:
        rght = xvals[i::nbins]

# Save the binned values
xvals_bin /= nbins
np.save('xvals_binned'+str(nbins)+'.npy', xvals_bin)

# Create the filters
for i in range(len(left)):
    with open('./filters/'+str(i+1).zfill(2)+'.dat', 'w') as foo:
        foo.write(str(left[i]-1e-10)+' 0.0\n')
        foo.write(str(left[i]) +' 1.0\n')
        foo.write(str((left[i]+rght[i])/2.) +' 1.0\n')
        foo.write(str(rght[i]) +' 1.0\n')
        foo.write(str(rght[i]+1e-10)+' 0.0\n')


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

import os
if __name__ == "__main__": 
    import sys;
    if sys.byteorder!='little':
        print('Only little endian machines currently supported! bye bye ....');
        quit();

    module_path = os.path.abspath('%s/../..'%os.path.dirname(os.path.realpath(__file__)));
    print(module_path);
    sys.path.append(module_path);
    sys.path.insert(0,module_path)

from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
from basis.preproc.palign import PeakAlign
params = {'mzmaxshift': 20, 'mzbinsize': 5, 'mzunits':'ppm'}
filepath = '/Users/kv/desktop/testMSIdata/testRow01.imzML'


def import_imzml(filepath,params):
    """
        Reads MSI peak picked data from the imzML file type 
        
        Parameters
        
            mzmaxshift: maximum peak shift in respective (ppm or Da) units
            mzbinsize:  bin size for histogram-resolution 
            mzunits:    ppm or Da units
    
    """ 

    imzfile       = ImzMLParser(filepath, parse_lib='ElementTree')
    n_intensities = sum(imzfile.intensityLengths)
    sp_indcs      = np.concatenate((np.array([0]),np.cumsum(imzfile.intensityLengths)))
    
    mz = np.zeros(n_intensities)
    sp = np.zeros(n_intensities)
    
    index = 0
    for idx, (x,y,z) in enumerate(imzfile.coordinates):
        imz, isp = imzfile.getspectrum(idx)
        sp[sp_indcs[index]:sp_indcs[index+1]] = isp
        mz[sp_indcs[index]:sp_indcs[index+1]] = imz
        index  = index + 1
        
    mz = mz[sp>0]
    sp = sp[sp>0]
    
    if (params['mzunits']=='Da') or (params['mzunits']=='ppm'):
             mzres = params['mzbinsize']
    else:
        print('Error: m/z need to be in Da or ppm units')
    
    # perform calculation of common mass to charge feature vector
    peakalign_obj = PeakAlign()
    cmz = peakalign_obj.get_reference(mz, mzres, params['mzmaxshift'], params['mzunits'])
    
    nmz = len(cmz)
    nsp = len(imzfile.intensityLengths)
    X   = np.zeros([nsp,nmz])
    xy  = np.zeros([nsp,3])
    
    # perform matching of individual m/z species to the common mass to charge (cmz) vector
    index = 0 
    if params['mzunits']=='ppm':
        cmz = peakalign_obj.to_ppm(np.array(cmz))
        
    for idx, (x,y,z) in enumerate(imzfile.coordinates):
        imz, isp = imzfile.getspectrum(idx)
        if params['mzunits']=='ppm':
           imz = peakalign_obj.to_ppm(np.array(imz))    
        refmzidcs, mzindcs = peakalign_obj.pmatch_nn(cmz,imz,params['mzmaxshift'])
        if len(refmzidcs)>0:
            X[index,refmzidcs] = np.array(isp)[mzindcs]
        xy[index,:]        = [x,y,z]
        index = index + 1 
        
    if params['mzunits']=='ppm':
        cmz = peakalign_obj.to_mz(cmz)
    
    rankx = np.argsort(np.sum(X,axis = 0))
    hdata = np.vstack((rankx,cmz))
    return X, xy, cmz, hdata
        
X, xy, mz, hdata = import_imzml(filepath,params)
  
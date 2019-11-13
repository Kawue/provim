# -*- coding: utf-8 -*-
"""
*******************************************
Variance Stabilizing Transformation Module
*******************************************

The module is designed to account for heteroscedastic noise structure 
characterized by increasing variance as a function of increased signal 
intensity. This procedure is essential to make sure that both small and large 
peaks have the same "technical" variance for downstream statistical analysis 
or visualization.    

run python.exe vst.py --help to get info about parameters of the module

Project:                        BASIS
License:                        BSD
Chief project investigator:     Dr. Kirill Veselkov
Lead developer:                 Dr. Kirill Veselkov 
 
References:

    [1] Veselkov KA, et al. (2011) Optimized preprocessing of ultra-
    performance liquid chromatography/mass spectrometry urinary metabolic 
    profiles for improved information recovery. Anal Chem 83(15):5864-5872.
    
    [2] Veselkov KA, et al. (2014), Chemo-informatic strategy for imaging mass 
    spectrometry-based hyperspectral profiling of lipid signatures in 
    colorectal cancer. PNAS, 111: 1216-122. 
 
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

import basis.io.manageh5db as mh5
import h5py
import time
import numpy as np
from scipy.stats.mstats import winsorize
from basis.utils.typechecker import is_number
from basis.utils.cmdline import OptionsHolder
from basis.procconfig import VST_options
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base
from basis.utils.timing import tic, toc

def do_vst(h5dbname='', method='', params='', istrain=1):
    
    """
    **Performs inter-sample normalization to account for overall intensity varation between tissue samples.**
     
    Args:
                
        h5dbname:  The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                     The variance stabilizing transformation needs to be applied after peak alignment, lock
                     mass correction and normalization steps.
                        
        
        method:      The choice of a variance stabilizing transformation method {``started-log`` (default)}
                     Additional methods can be added in a modular fashion. 
                    
        params:     The set of parameters for variance stabilizing transformation method. The ``offset`` parameter is added by prior to performing started-log transformation
                         
    """
    dataset_names = h5Base.get_traindata_names(h5dbname,istrain)
    if not dataset_names:
        return
    else:
        pathinh5 = h5Base.h5pathfinder(dataset_names[0])
    
    if istrain==1:
        windex = mh5.load_dataset(h5dbname, pathinh5 + 'windex')+1
        VSTObj = VST(index = windex, method = method, params = params)
    elif istrain==0:
        VSTObj = VSTObj()
        VSTObj.load_procobj(h5dbname,pathinh5)
        
    VSTObj.stabilize_h5(h5dbname,dataset_names)
   
    if istrain==1:
        mh5.save_dataset(h5dbname, pathinh5 + 'windex',windex)
        VSTObj.save_procobj(h5dbname,pathinh5)
        VSTObj.save_preproc2matlab(h5dbname,windex)
        
    return
                                 
    
class VST(h5Base):
    """
        **Container for inter-sample normalization class.**
        
        Attributes:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            method: The choice of variance stabilizing transformation
                    Additional methods can be added in a modular fashion. 
                        
            params: The set of parameters for variance stabilizing transformation. 
                              
            index: workflow index
            
            istrain: training or testing phase (1 or 0)
        
        """
    def __init__(self, index = '', method='', params='', mzrange=''):
        #  Upload base worfklow methods and attributes
        h5Base.__init__(self)
        self.description = 'Variance Stabilizing Transformation Settings'        
        self.method  = method
        self.index   = index
        self.params  = params
        self.istrain = 1
        self.do      = 'yes'
        
    def stabilize_h5(self,h5dbname, dataset_names):
        
        print('\n\n' '...Initializing variance stabizing transformation procedure... \n')
    
        h5proc = h5py.File(h5dbname,'a')
        i = 0
        for datasetid in dataset_names:
            try:
                X  = mh5.load_dataset(h5proc,datasetid+'/Sp')
                ncmz,nobs = X.shape
                
                # apply vst
                X  =  self.vst(X)
                
                # save dataset
                mh5.save_dataset(h5proc,datasetid+'/Sp',X)
                print('%s. %s: Successfully stabilized and deposited into --> %s' %(i + 1,datasetid,os.path.basename(h5dbname)))
            except:
                print('%s. %s:  Failed to be stabilized and deposited' %(i+1, datasetid))
                pass
            i = i + 1
               
        h5proc.close()
         
    
    def vst(self,X):
        """
        **Perform variance stabilizing transformation.**
    
        Args:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            method:      The choice of a variance stabilizing transformation method {``started-log`` (default)}
                         Additional methods can be added in a modular fashion. 
                        
            params:     The set of parameters for variance stabilizing transformation method. The ``offset`` parameter is added by prior to performing started-log transformation.               
            
         
        Returns:
        
            X: variance stabilized dataset.
        """
        #warnings.filterwarnings("ignore")
        # Added by K. Wuellems 08.04.2019
        if "winsorize" in self.params:
            winsorize_limits = self.params["winsorize"]
            winsorize_limits = [winsorize_limits[0], 1-winsorize_limits[1]]
            X = np.array(winsorize(X, winsorize_limits))

        methodselector = {'started-log': self.started_log, 'started-sqrt': self.started_sqrt, 'started-nonneg': self.started_nonneg}
        #####

        vstfunc  = methodselector.get(self.method);
        
        X  = vstfunc(X,self.params)          
        return X 
        
    def started_log(self, X, params):
        
        """
        **Performs started-log variance stabilizing transformation.**
    
        Args:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            params:     The set of parameters for variance stabilizing transformation method. The ``offset`` parameter is added by prior to performing started-log transformation.               
            
         
        Returns:
        
            X: variance stabilized dataset.
            
        """
        if 'offset' in params:
            offset = params['offset']
            if not is_number(params['offset']):
                # modified by K. Wuellems 08.04.2019
                if params['offset'] == "auto":
                    # Baseline of zero by log(1) = 0.
                    offset = np.amin(X)
                else:
                    #offset=1.
                    raise ValueError("Variance stabilizing offset has to be a number.")
        X = np.log((X-offset)+1)
        return X
    
    # Added by K. Wuellems 08.04.2019
    # Apply squareroot transform.
    def started_sqrt(self, X, params):
        if 'offset' in params:
            offset = params['offset']
            if not is_number(params['offset']):
                if params['offset'] == "auto":
                    # Baseline of zero by sqrt(0) = 0.
                    offset = np.amin(X)
                else:
                    #offset=1.
                    raise ValueError("Variance stabilizing offset has to be a number.")
        X = np.sqrt(X-offset)
        return X

    # Only apply a offset in case of negative values to have a baseline of zero.
    def started_nonneg(self, X, params):
        if 'offset' in params:
            print("For non neg transform the offset will be ignored and is chosen automatically.")
        # Baseline of zero with no transform.
        offset = np.amin(X)
        X = X-offset
        return X

    #####
            

if __name__ == "__main__": 
    tic()
    settings=OptionsHolder(__doc__, VST_options);
    settings.description='Variance Stabilizing Transformation Settings';
    settings.do='yes';
    print(settings.program_description);
    settings.parse_command_line_args();
    print(settings.format_parameters());
    print('\nStarting.....')
    
    do_vst(h5dbname = settings.parameters['h5dbname'], method=settings.parameters['method'], params=settings.parameters['params']);

    print('\nFinished on %s in'%(time.strftime("%a, %d %b %Y at %H:%M:%S")))   
    toc()
    print(settings.description_epilog);

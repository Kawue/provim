## -*- coding: utf-8 -*-
"""
*********************************
Intra-sample Normalization Module
*********************************

The module is designed to account for overall intensity (pixel to pixel) 
variation between spectra within individual datasets. This unwanted variation 
can be caused by a variety of reasons, including heterogeneous matrix 
deposition (e.g. MALDI) or differences in tissue thickness within a tissue 
sample.   

See also inter-sample normalization module to account for global intensity 
differences between samples.

run python.exe intranorm.py --help to get info about parameters of the module

  
References:

    [1] Veselkov KA, et al. (2011) Optimized preprocessing of ultra-
    performance liquid chromatography/mass spectrometry urinary metabolic 
    profiles for improved information recovery. Anal Chem 83(15):5864-5872.
    
    [2] Veselkov KA, et al. (2014), Chemo-informatic strategy for imaging mass 
    spectrometry-based hyperspectral profiling of lipid signatures in 
    colorectal cancer. 
    PNAS, 111: 1216-122. 
 
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
    sys.path.insert(0,module_path);

import h5py
import time
import numpy as np
import basis.io.manageh5db as mh5
from basis.procconfig import IntraNorm_options
from basis.utils.typechecker import is_string, is_number
from basis.utils.cmdline import OptionsHolder
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base
from basis.utils.timing import tic, toc

def do_normalize(h5dbname='', method='mfc', params = {'offset': 0, 'reference': 'median'}, 
                 mzrange=[0, 2000],  istrain=1):
    
    """
    **Performs intra-sample normalization module to account for overall intensity (pixel to pixel) variation between spectra within individual datasets.**
     
    See also inter-sample normalization to account for overall intensity varation between tissue samples. 
    
    Args:
                    
        dbprocpath:  The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                     The intra/inter-sample normalization procedures are applied after peak alignment and lock
                     mass correction steps.
                                
        method:     The choice of an intra-sample normalization method {``median fold change`` (default), ``mean`` or ``median``}
                    Additional methods can be added in a modular fashion. 
                    
        params:     The set of parameters for intra-sample normalization method. The median fold change normalization requires the refence dataset with respect to which the fold intensity changes of other datasets are
                    calculated, ``mean`` by default. The ``offset`` parameter disregards peak intensity smaller that its value. 
                    ``{'reference': 'mean', 'offset': 0}`` by default for median fold change 
                    
        mzrange:    [mzmin mzmax] - the mass-to-charge ratio (m/z) range desired to be processed. If not specified, the full mz range is used. 
         
    """
                                                              
    dataset_names = h5Base.get_traindata_names(h5dbname,istrain)
    if not dataset_names:
        return
    else:
        pathinh5 = h5Base.h5pathfinder(dataset_names[0])
    
    if istrain==1:
        windex = mh5.load_dataset(h5dbname, pathinh5 + 'windex')+1
        IntraNormObj = IntraNorm(index = windex, method = method, params = params, 
                                 mzrange = mzrange)
    elif istrain==0:
        IntraNormObj = IntraNorm()
        IntraNormObj.load_procobj(h5dbname,pathinh5)
    
    IntraNormObj.intranormalize_h5(h5dbname,dataset_names)
   
    if istrain==1:
        mh5.save_dataset(h5dbname, pathinh5 + 'windex',windex)
        IntraNormObj.save_procobj(h5dbname,pathinh5)
        IntraNormObj.save_preproc2matlab(h5dbname,windex)
        
    return


class IntraNorm(h5Base):
    
    """
        **Container for intra-sample normalization class.**
        
        Attributes:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            method: The choice of an inter-sample normalization method {``median fold change`` (default), ``mean`` or ``median``}
                    Additional methods can be added in a modular fashion. 
                        
            params: The set of parameters for inter-sample normalization method. The median fold change normalization requires the refence profile with respect to which the fold intensity changes are
                    calculated, ``mean`` by default. The ``offset`` parameter disregards peak intensity smaller that its value. 
                    ``{'reference': 'mean', 'offset': 0}`` by default for median fold change 
                        
            mzrange:    [mzmin mzmax] - The mass-to-charge ratio (m/z) range desired to be processed. If not specified, the full data range will be used. 
            
            index: workflow index
             
            mz: The mass to charge feature vector.
            
            istrain: training or testing phase (1 or 0)
        
        """
        
    def __init__(self, index = '', method='', params='', mzrange=''):
        #  Upload base worfklow methods and attributes
        h5Base.__init__(self)
        self.description = 'Intranormalization Settings'        
        self.mzrange = mzrange    
        self.method  = method
        self.index   = index
        self.params  = params
        self.istrain = 1
        self.do      = 'yes'
        
    
    def intranormalize_h5(self,h5dbname,datasets):
            
        """
        **Performs iterative intra-sample normalization using h5-database file.**
        
        Args:
            
            h5dbname: The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                  
            datasets: The names of datasets to be normalized

        """
        
        print('\n\n...Initializing intra-sample normalization procedure...\n ') 
   
        h5proc = h5py.File(h5dbname,'a')
        i = 0
        for datasetid in datasets:
            try:  
                # X [number of features, number of rows, number of columns] - opposite to matlab
                mz = mh5.load_dataset(h5proc, datasetid+'/mz')
                X  = mh5.load_dataset(h5proc, datasetid+'/Sp') 
                
                # perform intra-sample normalization            
                scf,refX  = self.intranormalize(X, mz)

                X = X/scf
                
                # re-write data-set in the hdf5 database file     
                mh5.save_dataset(h5proc, datasetid+'/Sp', X)
                
                print('%s. %s: Successfully normalized and deposited into --> %s' %(str(i+1),datasetid, os.path.basename(h5dbname)))
            except Exception as inst:
                print(inst);
                print('%s. %s:  Failed to be normalized and deposited' %(str(i+1),  datasetid))
                pass    
            i = i + 1
       
        h5proc.close()   
            
    def intranormalize(self,X, mz):
        
        """
        **Perform intra-sample normalization.**
        
        Args:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            method:      The choice of an intra-sample normalization method {``median fold change`` (default), ``mean`` or ``median``}
                         Additional methods can be added in a modular fashion. 
                        
            params:     The set of parameters for inter-sample normalization method. The median fold change normalization requires the refence profile with respect to which the fold intensity changes are
                        calculated, ``mean`` by default. The ``offset`` parameter disregards peak intensity smaller that its value. 
                        ``{'reference': 'mean', 'offset': 0}`` by default for median fold change 
                                
            mzrange:    [mzmin mzmax] - The mass-to-charge ratio (m/z) range desired to be processed. If not specified, the full data range will be used. 
            
            mz: The mass to charge feature vector.
    
        Returns:
            
            X: normalized dataset
        
        """              
        ## select appropriate normalization method    
        methodselector = {'mean': self.__mean, 
                          'median': self.__median, 
                          'mfc': self.__mfc};
                  
        normfunc  = methodselector.get(self.method);
        
        ## prepare data for normalization    
        Xn = X    
        if len(self.mzrange)==2:
                Xn = X[(mz>self.mzrange[0]) & (mz<self.mzrange[1]),:]
        nmz,nobs = Xn.shape
        
        offset = self.params['offset'];
        Xn[np.isnan(Xn)] = 0
        if is_number(offset):
            Xn[Xn<=offset]   = 0
        
        ## ignore mz features and spectra with all zeros or nans
        sumrowX   = np.nansum(Xn,axis=1)
        Xn        = Xn[sumrowX!=0,:]
        sumcolX   = np.nansum(Xn,axis=0)
        Xn        = Xn[:,sumcolX!=0]
        Xn[Xn==0] = np.nan

        ## get normalization factors for each spectrum  
        scf,redrefX = normfunc(Xn,self.params)
        
        scf = scf.reshape(1,len(scf))

        scf[np.isnan(scf)] = 1
        scf[scf==0] = 1
        scf = np.divide(scf,np.nanmedian(scf))
        
        ## estimate expected range of scaling factors and adjust the outlying ones
        if 'outliers' in self.params:
            if self.params['outliers']=='yes':
                logscf     = np.log(scf[scf>0]);
                stdlogscf  = 1.4826 * np.median(abs(logscf - np.median(logscf)))
                scf[scf<np.exp(-stdlogscf*3)] = np.exp(-stdlogscf*3)
                scf[scf>np.exp(+stdlogscf*3)] = np.exp(+stdlogscf*3)
        
        ## divide each spectrum with its estimated normalization factor        
        refX = np.zeros([nmz,1])        
        scX  = np.ones([nobs,1])
        # Seems to provide an error with _mean and _median. Fix by K.Wuellems 10.04.2019
        if len(redrefX) > 0:
            refX[sumrowX!=0] = redrefX
        else:
            refX = redrefX
        scX[sumcolX!=0] = scf.T
        scX = scX.T
        return scX, refX
    
    def __mean(self,X,params):
        
        """
        
        **Caclulates mean value per spectrum, **mean intra-sample normalization.**
        
        Args:
        
            X: MSI dataset (number of m/z features x number of spectra )
        
        Returns:
            scfactors: normalization factors.
        
        """

        scfactors = np.nanmean(X,axis=0)
        refX = []
        return scfactors, refX
    
    def __median(self,X,params):
        
        """
        
        **Caclulates median value per spectrum, **mean intra-sample normalization.**
        
        Args:
            X: MSI dataset (number of m/z features x number of spectra )
        
        Returns:
            scfactors: normalization factors
        
        """
                  
        scfactors = np.nanmean(X,axis=0)
        refX = []
        return scfactors, refX
    
    def __mfc(self,X,params):
        
        """
        
        **Caclulates median fold change value, **median fold change intra-sample normalization.**
        
        Args:
            X: MSI dataset (number of m/z features x number of spectra )
       
       Returns:
            scfactors: normalization factors
        
        """
        
        ref = params['reference']
        if is_string(ref) and ref=='mean':
            refX = np.nanmean(X,axis=1)
        elif is_string(ref) and ref =='median':
            refX = np.nanmedian(X,axis=1)
        else:
            refX=ref
        
        refX[refX==0]=np.nan
        refX = refX.reshape(len(refX),1)
        scfactors = np.nanmedian(X/refX,axis=0)
        return scfactors, refX

        
if __name__ == "__main__": 
    tic()
    settings=OptionsHolder(__doc__, IntraNorm_options);
    settings.description='Itranormalization Settings';
    settings.do='yes';
    print(settings.program_description);
    settings.parse_command_line_args();
    print(settings.format_parameters());
    print('\nStarting.....')
    
    do_normalize(h5dbname=settings.parameters['h5dbname'],\
                 method=settings.parameters['method'],\
                 params = settings.parameters['params'],\
                 mzrange=[settings.parameters['min_mz'], settings.parameters['max_mz']]);
    
    print('\nFinished on %s in'%(time.strftime("%a, %d %b %Y at %H:%M:%S")));   
    toc()  
    print(settings.description_epilog)
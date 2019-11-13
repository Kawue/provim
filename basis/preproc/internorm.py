# -*- coding: utf-8 -*-
"""
*********************************
Inter-sample Normalization Module
*********************************

The module is designed to account for overall intensity differences between 
MSI datasets of multiple tissue samples. This unwanted variation can be caused
by a variety of reasons, including  differences in sample preparation steps or 
tissue section thickness.   

See also intra-sample normalization module to account for overall intensity 
(pixel to pixel) variation between spectra within individual datasets.


run python.exe internorm.py --help to get info about parameters of the module


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
    sys.path.insert(0,module_path)

import h5py
import time
import numpy as np
import warnings
import basis.io.manageh5db as mh5
from basis.preproc.intranorm import IntraNorm
from basis.utils.cmdline import OptionsHolder
from basis.procconfig import InterNorm_options
from basis.utils.typechecker import is_number
from basis.utils.timing import tic, toc

def do_normalize(h5dbname='', method='mfc', params = {'offset': 0, 'reference': 'median'}, 
                 mzrange=[0, 2000], istrain=1):
    
    """
    **Performs inter-sample normalization to account for overall intensity varation between tissue samples.**
     
    Args:
                    
        h5dbname:  The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                     The intra/inter-sample normalization procedures are applied after peak alignment and lock
                     mass correction steps.
                        
        
        method:      The choice of an inter-sample normalization method {"MFC" for median fold change (default), ``mean`` or ``median``}
                     Additional methods can be added in a modular fashion. 
                    
        params:     The set of parameters for inter-sample normalization method. The median fold change normalization requires the refence dataset with respect to which the fold intensity changes of other datasets are
                    calculated, ``mean`` by default. The ``offset`` parameter disregards peak intensity smaller that its value. 
                    ``{'reference': 'mean', 'offset': 0}`` by default for median fold change 
                    
        
        mzrange:    [mzmin mzmax] - the mass-to-charge ratio (m/z) range desired to be processed. If not specified, the full mz range is used. 
         
    """
                    
    dataset_names = IntraNorm.get_traindata_names(h5dbname,istrain)
    if not dataset_names:
        return
    else:
        pathinh5 = IntraNorm.h5pathfinder(dataset_names[0])
    
    if istrain==1:
        cmz = mh5.load_dataset(pathinh5 + h5dbname,'cmz')
        windex = mh5.load_dataset(h5dbname, pathinh5 + 'windex') + 1 
        InterNormObj = InterNorm(index = windex, method = method, params = params, 
                                 mzrange = mzrange, mz = cmz)
    elif istrain==0:
        InterNormObj = InterNorm()
        InterNormObj.load_procobj(h5dbname,pathinh5)
    
    InterNormObj.internormalize_h5(h5dbname,dataset_names)
 
    if istrain==1:
        mh5.save_dataset(h5dbname, pathinh5 + 'windex',windex)
        InterNormObj.save_procobj(h5dbname,pathinh5)
        InterNormObj.save_preproc2matlab(h5dbname,windex)
    
    return

class InterNorm(IntraNorm):
    
    """
        **Container for inter-sample normalization class.**
        
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
        
    def __init__(self, index = '', method='', params='', mzrange='', mz=''):
        IntraNorm.__init__(self)
        self.description = 'Internormalization Settings'
        self.mzrange = mzrange    
        self.method  = method
        self.params  = params
        self.index   = index
        self.mz      = mz
        self.do      = 'yes'
        self.istrain = 1
    
    def internormalize_h5(self,h5dbname, datasets):
        
        """
        **Performs iterative inter-sample normalization using h5-database file.**
        
        Args:
            
            h5dbname: The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                  
            datasets: The names of datasets to be normalized

        """
        
        print('\n\n...Preparing data for inter-sample normalization procedure... \n')
        h5proc = h5py.File(h5dbname,'a')
        
        # derive normalization scaling factors per dataset in an iterative fashion
        refX = []
        passdatasets = []
        i = 0
        for datasetid in datasets:
            try:
                X  = mh5.load_dataset(h5proc,datasetid+'/Sp')
                refx = self.__get_scaling_factors(X,self.mz)
                refX.append(refx) 
                print('%s. %s: Prepared for normalization' %(str(i+1),datasetid))
                passdatasets.append(datasetid)
            except:
                print('%s. %s:  Failed to be prepared' %(str(i+1), datasetid))
                pass
            i = i + 1
        
        # arrange reference profiles
        refX = np.transpose(np.squeeze(np.array(refX)))
        
        # normalize datasets
        if self.method=='mean' or self.method=='median':
            scX = refX
            refX = []
        else:
            self.mzrange= []
            scX,refX = self.intranormalize(refX,self.mz)
        
        # maintain original data scale
        if self.istrain==1:
            self.gscale = np.nanmedian(scX)
            if len(refX)>0:
                self.params['reference'] = refX
        
        scX = scX/self.gscale
        scX = scX.squeeze()
        scX[np.isnan(scX)] = 1
        scX[scX==0] = 1
        
        # now apply normalization procedure in an iterative fashion, one sample at a time 
        print('\n\n ...Performing inter-sample normalization... \n')
        i = 0    
        for datasetid in passdatasets:
            try:
                X  = mh5.load_dataset(h5proc, datasetid+'/Sp') 
                X  = X/scX[i]
             
                mh5.save_dataset(h5proc, datasetid+'/Sp', X)
                
                print('%s. %s: Successfully normalized and deposited into --> %s' %(i+1, datasetid,os.path.basename(h5dbname)))
            except:
                print('%s. %s:  Failed to be normalized and deposited' %(i+1, datasetid))

            i = i+1        
        
        h5proc.close()
        return
    
    def __get_scaling_factors(self,X,mz):
        
        """
        **Caclulates scaling factors for inter-sample normalization procedure.**
        
        Args:
            
            X: MSI dataset (number of m/z features, number of rows, number of columns).
                  
            method:      The choice of an inter-sample normalization method {``median fold change`` (default), ``mean`` or ``median``}
                         Additional methods can be added in a modular fashion. 
                        
            params:     The set of parameters for inter-sample normalization method. The median fold change normalization requires the refence profile with respect to which the fold intensity changes are
                        calculated, ``mean`` by default. The ``offset`` parameter disregards peak intensity smaller that its value. 
                        ``{'reference': 'mean', 'offset': 0}`` by default for median fold change 
                        
            mzrange:    [mzmin mzmax] - The mass-to-charge ratio (m/z) range desired to be processed. If not specified, the full data range will be used. 
            
            mz: The mass to charge feature vector.
        
        Returns:
                 refX: reference profile or scaling factor for inter-sample normalization.
        """

        methodselector = {'mean': self.__get_refmean, 
                          'median': self.__get_refmedian, 
                          'mfc': self.__get_refmfc};
                  
        normfunc  = methodselector.get(self.method);
        
        ## prepare data for normalization    
        nmz,nobs = X.shape
        
        Xn = X    
        if len(self.mzrange)==2:
            Xn = X[(mz>self.mzrange[0]) & (mz<self.mzrange[1]),:]
       
        offset = self.params['offset'];
        if is_number(offset):
            Xn[Xn<=offset] = 0
        
        # remove all zero spectra
        Xn = Xn[:,np.nansum(Xn,axis=0)!=0]    
        Xn[Xn==0] = np.nan
            
        refX = normfunc(Xn,self.params)
        return refX
        
    def __get_refmean(self,X,params):
        
        """
        **Caclulates global median value across all spectra in a dataset, **mean inter-sample normalization.**
        
        Args:
            
            X: MSI dataset (number of spectra x number of m/z features)
                
        Returns:
            
            refX: reference profile
            
        """
        
        refx = np.nanmean(X)
        
        return refx
    
    def __get_refmedian(self,X,params):
        
        """
        
        **Caclulates global median value across all spectra in a dataset, **mean inter-sample normalization.**
        
        Args:
        
            X: MSI dataset (number of spectra x number of m/z features)
        
        Returns:
            
            refX: reference profile
            
        """
        
        refx = np.nanmedian(X)
        return refx
        
    def __get_refmfc(self,X,params):
        
        """
        
        **Caclulates median profile across all spectra in a dataset, **median fold change inter-sample normalization.**
        
        Args:
                  X: MSI dataset (number of spectra x number of m/z features)
                  
                  params: {'reference': 'mean'}, the choice of representative profile profile for median fold change normalization, 
                  'mean' by default. 
        
        Returns:
            
            refX: reference profile
        """
        
        ref = params['reference']
        #expect to see runtime warning of the mean/median of empty slice in case all column values are none
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            if ref=='mean':
                refx = np.nanmean(X,axis=1)
            elif ref =='median':
                refx = np.nanmedian(X,axis=1)
            
        refx[refx==0]=np.nan
        refx = refx.reshape(len(refx),1)
        return refx
        
    
if __name__ == "__main__": 
    tic()
    settings=OptionsHolder(__doc__, InterNorm_options);   
    settings.description='Iternormalization Settings';
    settings.do='yes';
    settings.gscale=[];
    print(settings.program_description);
    settings.parse_command_line_args();
    print(settings.format_parameters());
    print('\nStarting.....')

    #settings.parameters['h5dbname'] = '/Users/kv/desktop/test/pyproc_data__1928_22_03_2017.h5'
    do_normalize(h5dbname=settings.parameters['h5dbname'],\
                 method=settings.parameters['method'],\
                 params = settings.parameters['params'],\
                 mzrange=[settings.parameters['min_mz'], settings.parameters['max_mz']]);
    
    print('\nFinished on %s in'%(time.strftime("%a, %d %b %Y at %H:%M:%S")));   
    toc()  
    print(settings.description_epilog);
    

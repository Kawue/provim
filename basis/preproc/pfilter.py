# -*- coding: utf-8 -*-
"""
***********************************
Solvent/Matrix Peak Removal Module
***********************************

The module is designed to filter out matrix/solvent/contaminant related peaks from mass 
spectrometry imaging datasets via tailored cluster-driven strategy.  

run python.exe pfilter.py --help to get info about parameters of the module


References:

    [1] Veselkov KA, et al. (2014), Chemo-informatic strategy for imaging mass 
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
    sys.path.append(module_path)
    sys.path.insert(0,module_path);

import h5py
import time
import numpy as np
import basis.io.manageh5db as mh5
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base
from basis.utils.cmdline import  OptionsHolder
from basis.procconfig import PeakFilter_options
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from basis.utils.timing import tic, toc

def do_filtering(h5dbname='', method='', params = '', istrain = 1):
    """
    **Performs filtering of solvent/matrix related peaks by means of tailored cluster-driven strategy**
     
    Args:
                
        h5dbname:  The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                     The peak filtering needs to be applied after variance stabilizing transformation, lock
                     mass correction and normalization steps.
        
        method:      The choice of peak filteing methods. Additional methods can be added in a modular fashion. 
                    
        params:     The set of parameters for peak filtering.
                         
    """
    dataset_names = h5Base.get_traindata_names(h5dbname,istrain)
    if not dataset_names:
        return
    else:
        pathinh5 = h5Base.h5pathfinder(dataset_names[0])
    
    if istrain==1:
        windex = mh5.load_dataset(h5dbname, pathinh5 + '/windex')+1
        PeakClusterObj = PeakCluster(index = windex, method = method, params = params)
        PeakClusterObj.fit(h5dbname,dataset_names)
    elif istrain==0:
        print('\nThe solvent/matrix related peaks have already been identified based on training data...\n')
   
    if istrain==1:
        mh5.save_dataset(h5dbname, pathinh5 + '/windex',windex)
        PeakClusterObj.save_preproc2matlab(h5dbname,windex)
        PeakClusterObj.save_procobj(h5dbname,pathinh5)
        
    return
   

class PeakCluster(h5Base):
    """
    **Container for peak filtering class.**
    
    Attributes:
        
        method: The choice of peak filtering strategy.
                Additional methods can be added in a modular fashion. 
                    
        params: The set of parameters for peak filtering. 
                          
        index: workflow index
        
        labels: the grouping labels of m/z features differentianting solvent related peaks
        from object related peaks 
        
        istrain: training or testing phase (1 or 0)
    
    """

    def __init__(self, index='', method='', params=''):
        h5Base.__init__(self)
        self.description = 'Peak Filtering Settings'
        self.do       = 'yes'
        self.index    = index
        self.istrain  = 1
        self.method   = method
        self.params   = params
        self.labels   = np.array([])
    
    def fit(self, h5dbname = '', dataset_names = ''):
        
        """
        **Performs filtering of solvent/matrix related peaks via cluster-driven strategy.**
        
         Args:
            
            h5dbname: The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                  
            dataset_names: The names of datasets for peak filtering
        
        """    
           
        if 'nclusters' in self.params: 
            nclusters = int(self.params['nclusters'])
        else:
            nclusters = 2 
        
        print('\n\n' '...Initializing peak filtering procedure... ')
        print("\nPerforming iterative distance calculation...")
        
        # Calculate correlation matrix in an iterative fashion        
        self._iter_corr(h5dbname, dataset_names)
        if len(self.distmat)<=0:
            print("Clustering has failed...")
            return
            
        # Principal component analysis or multi-dimensional scaling for dimensionality reduction  
        self.distmat = self.distmat-np.mean(self.distmat,axis=1)
        ObjPCA = PCA(n_components=nclusters)
        ObjPCA.fit(self.distmat)
        PC_Scores = np.dot(self.distmat,ObjPCA.components_.T) 
        
        if self.method=='kmeans':      
            print("\nPerforming clustering...")
            Objkmeans = KMeans(n_clusters=nclusters).fit(PC_Scores)
            self.labels = Objkmeans.labels_
            print("\nCompleted!\n\n")
        
        return
                
    def _iter_corr(self, h5dbname, dataset_names):
        
        """
        **Caclulates distance matrix between data table features in a iterative fashion.**
        
        Args:
            
            h5dbname: The path to a hdf5-based msi database for storage and organization of pre-processed MSI data. 
                  
            dataset_names: The names of datasets for peak filtering
        
        """    
        h5proc = h5py.File(h5dbname,'a')
        cmz = mh5.load_dataset(h5proc,'cmz')
        nmz = len(cmz)
        self.distmat = np.zeros([nmz,nmz])
        
        i = 0
        for datasetid in dataset_names:
            try:
                # import data from the hdf5 database file
                Xi       = mh5.load_dataset(h5proc,datasetid+'/Sp')
                nmz,nXi  = Xi.shape
                stdX     = np.std(Xi,axis=1)
                indcs    = np.array(np.where(stdX!=0)).flatten()
                icorrmat = np.corrcoef(Xi[indcs,:])              
                icorrmat[np.isnan(icorrmat)] = 0
                self.distmat[np.ix_(indcs,indcs)] = self.distmat[np.ix_(indcs,indcs)] + icorrmat
                print('%s. %s: Successfully updated...' %(i+1,  datasetid))
            except:
                print('%s. %s: Failed to update...' %(i+1, datasetid))
                pass
            i = i + 1
        
        h5proc.close() 
        
        # Adjusted by K. Wuellems 20.02.2019
        if i==1:
            self.distmat = self.distmat
        elif i>1:
            self.distmat = self.distmat/i
        else:
            self.distmat = []
        
        return    
        
            
if __name__ == "__main__": 
    tic()
    settings=OptionsHolder(__doc__, PeakFilter_options);   
    settings.description='Peak Filtering Settings';
    settings.do='yes';
    print(settings.program_description);
    settings.parse_command_line_args();
    print(settings.format_parameters());
    print('\nStarting.....')
    #settings.parameters['h5dbname'] = '/Users/kirillveselkov/desktop/DESI_Art/pyproc_data__1928_22_03_2017.h5'
    #settings.parameters['h5dbname'] = '/Users/kirillveselkov/Desktop/test/dbproc9.h5'
    do_filtering(h5dbname=settings.parameters['h5dbname'],\
                 method=settings.parameters['method'],\
                 params = settings.parameters['params'])
                 
    print('\nFinished on %s in'%(time.strftime("%a, %d %b %Y at %H:%M:%S")));   
    toc()  
    print(settings.description_epilog)
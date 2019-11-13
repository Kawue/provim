# -*- coding: utf-8 -*-
"""
*********************
Peak alignment module
*********************

The module is designed to adjust for the inherent variation in instrumental 
measurements of m/z ratio and ion mobility drift time of molecular ion species 
across large-scale mass spectrometry data-sets

run python.exe palign.py --help to get info about parameters of the module

 
References:

    [1] Veselkov KA, Lindon JC, Ebbels TMD, Crockford D, Volynkin VV, Holmes E,
        Davies DB, Nicholson J, 2009, Recursive Segment-Wise Peak Alignment of 
        Biological H-1 NMR Spectra for Improved Metabolic Biomarker Recovery, 
        Analical Chemistry, Vol: 81, Pages (56-66).
    
    [2] Jeffries N, 2005 "Algorithms for alignment of mass spectrometry 
        proteomic data", Bioinformatics, 21(14), 3066-3073. 
 
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
from basis.utils.signalproc import smooth1D, get_threshold, median_threshold
from basis.utils.cmdline import OptionsHolder
from basis.procconfig import PeakAlign_options
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base
from scipy import interpolate
from basis.utils.typechecker import is_number
from basis.utils.timing import tic, toc

def do_alignment(h5rawdbname, h5dbname='', method='NN', 
                 params = {'mzmaxshift': 100, 'cmzbinsize': 10, 'mzunits':'ppm', 'lockmz': '885.5468'}, pathinh5=''):
    
    """
    **Performs alignment of the measured drift time or mz vectors of molecular ion peaks 
    across multiple MSI datasets. If not provided, the reference mz (rmz) or drift 
    time value of a given peak is computed via kernel based approach 
    to be the most common ("representative") across all datasets.** 
    
     
    Args:
            
        h5rawdbname: Path to a hdf5-based database file with multiple deposited msi datasets 
                        for peak alignment. Each MSI dataset assumes to contain peak picked 
                        MSI data with their spatial coordinates and mz (and optionally drift time)
                        feature vectors. 
                    
        h5dbname:  Path to a hdf5-based msi database for storage and organization of pre-processed data; 
                        if this db file exists, all pre-processing parameters will be extracted from it,
                        to make sure that the newly imported data are compatible with the ones stored in 
                        the processed db. The pre-processing workflow can be customized for newly 
                        created db instance.
                            
        method:   Method for adjusting measured drift time or mz vectors of molecular ion peaks 
                            across multiple MSI datasets. By default, the measured values are matched to the reference ones
                            using the 'nearest neighbour' approach. The reference profile estimated based on kernel density approach
                            to contain features with the highest number of peaks across all samples
                   
        params:  ``{'param1': 'value', 'param2', 'value'}`` Parameter set for the alignment method. The maximum
                            allowed peak shift across datasets needs to be specified for the default nearest neighbour 
                            apprroach (e.g. 0.1 in Da, by default).
                    
                    mzmaxshift - the max m/z drift allowed for adjustment between samples
                    cmzbinsize - the histogram resolution for common m/z vector estimation   
                    mzunits    - choice of units for peak alignments (e.g. 'Da' or 'ppm' for m/z feature vector)
                   
                    Note that the parameter values for reference vector computations or
                    peak alignments need to be specified in their respective units (e.g. max peak shift: 0.01 (Da) or
                    max peak shift: 100 (ppm)     
    """
    
    rawdataset_names = h5Base.get_dataset_names(h5rawdbname)
    if not rawdataset_names:
        return
    
    if h5Base.checkdbfile(h5dbname):
        PAlignObj = PeakAlign()
        PAlignObj.load_procobj(h5dbname,pathinh5)    
    else:
        h5dbname = h5Base.generate_h5filename(h5rawdbname,h5dbname) 
        PAlignObj = PeakAlign(method , params)
        PAlignObj.geth5_reference(h5rawdbname, rawdataset_names, 
                                  PAlignObj.params['mzunits'],pathinh5) 

        if len(PAlignObj.cmz)==0:
            print('ERROR: cmz vector has not been estimated. Terminating!')
            return
        
        # performs iterative alignment-one sample at a time 
    PAlignObj.matchh5_mz2cmz(h5rawdbname,rawdataset_names,h5dbname,pathinh5)
                    
    
    # save parameters in case of the training mode
    if PAlignObj.istrain==1:
        # legacy for matlab only (to be deleted), as reconstructed from pre-processing object
        mh5.save_dataset(h5dbname,pathinh5 + 'cmz',data=PAlignObj.cmz)
        mh5.save_dataset(h5dbname, pathinh5 + 'windex',1)
        PAlignObj.save_procobj(h5dbname,pathinh5)
        PAlignObj.save_preproc2matlab(h5dbname,1)


class PeakAlign(h5Base): 
    """    
    **The container containing the choice of methods and parameters for m/z drift correction**
         
    Attributes:
            
                  
        method:   Method for adjusting measured drift time or mz vectors of molecular ion peaks 
                            across multiple MSI datasets. 
             
        params:  the parameter set for the imported file type, ``{'HDI_fileext':'.txt', 'HDI_mzline': 3, 'HDI_delimiter': '\t'}`` 
        
        istrain: whether the dataset is used for training or testing

    """    
       
    def __init__(self, method ='', params=''):
        h5Base.__init__(self)
        self.description = 'Peak Alignment Settings'
        self.method  = method
        self.params  = params
        self.windex   = 1
        self.istrain  = 1
        if 'lockmz' in self.params:
            if len(self.params['lockmz'])>1:
                try:
                    self.params['lockmz'] = np.sort(np.array([float(j) for j in params['lockmz'].split(',')]))
                except:
                    try:
                        self.params['lockmz'] = np.array(float(self.params['lockmz']))
                    except:
                        #raise lockparseerror
                        print('Error: could not exctract user provided known lock masses')
                        self.params['lockmz'] = ''
                
        
    def matchh5_mz2cmz(self, h5rawdbname, rawdataset_names, h5dbname, pathinh5):
        
        """
        Performs matching of mz feature vector with the common one using h5-based database file.
        
        Args:
        
            h5rawdbname: h5-based database file name with raw peak picked msi data. 
            
            rawdataset_names: raw dataset names
                        
            h5dbname: h5-based database file name with processed peak picked msi data .
                 
        """     
        
        print('\n\n' '...Initializing m/z drift correction procedure procedure... ')
        h5raw = h5py.File(h5rawdbname,'r')
        
        h5proc = h5py.File(h5dbname, 'a')     
        ncmz = len(self.cmz)
         
        # alignment method selection 
        methodselector = {'NN': self.pmatch_nn}
        alignfunc  = methodselector.get(self.method);
        
        mzunits = self.params['mzunits']    
        if mzunits=='ppm':
            cmz = self.to_ppm(self.cmz)
        else:
            cmz = self.cmz
        
        i=-1
        for datasetid in rawdataset_names:
            i = i + 1
            try:
                print('%s. %s: Preparing for alignment...' %(str(i+1),datasetid))   
                mzraw = mh5.load_dataset(h5raw,pathinh5 + datasetid+'/mzraw')
                Xraw  = mh5.load_dataset(h5raw,pathinh5 + datasetid+'/Spraw') 
                xy    = mh5.load_dataset(h5raw,pathinh5 + datasetid+'/xy')  
                hdata = mh5.load_dataset(h5raw,pathinh5 + datasetid+'/hdata')
                
                if  len(self.params['lockmz'])>0 and is_number(self.params['lockmz'][0]):
                   mzraw = self.calibrate_mz(mzraw,self.params['lockmz'],
                                          self.params['mzmaxshift'],self.params['mzunits'])
                if mzunits=='ppm':
                    mzraw = self.to_ppm(mzraw)   
                refmzidcs, mzindcs = alignfunc(cmz,mzraw,self.params['mzmaxshift'])
               
                # create an array
                [nmz,nobs] = Xraw.shape
                if (nobs==0) :
                    print(len(str(i+1))*' ' +'  %s: Corrupt data' %os.path.basename(pathinh5 + datasetid))
                else:
                    Xal                  = np.zeros((ncmz,nobs),dtype='float')
                    Xal[refmzidcs,:]     = Xraw[mzindcs,:]
                    nmz,nheadlines       = hdata.shape
                    hdataal              = np.zeros((ncmz,nheadlines),dtype='float')
                    hdataal[refmzidcs,:] = hdata[mzindcs,:] 
                    
                    # save the datasets 
                    ginfo = h5proc.create_group(pathinh5 + datasetid)
                    mh5.save_dataset(ginfo,'Sp',data = Xal,chunksize=(ncmz,1),compression_opts=4)
                    mh5.save_dataset(ginfo,'sizeSp',data =[ncmz,nobs])                
                    mh5.save_dataset(ginfo,'xy',data = xy,chunksize=True)
                    mh5.save_dataset(ginfo,'hdata',data = hdataal,chunksize=True)
                    mh5.save_dataset(ginfo,'mz',data = self.cmz,chunksize=True)
                    mh5.save_dataset(ginfo,'istrain',data = self.istrain)
                    print(len(str(i+1))*' ' + '  %s: Successfully aligned and deposited into --> %s' %(pathinh5 + datasetid,os.path.basename(h5dbname)))
            except:
                print(len(str(i+1))*' ' + '  %sFailed to be aligned and deposited' %pathinh5 + datasetid)
                pass
                    
        h5raw.close()
        h5proc.close()
        return;
    
    def geth5_reference(self, h5rawdbname, rawdataset_names, units, pathinh5, h5readpath='/mzraw'):       
        
        """
        Calculates the reference feature vector for peak alignment using hdf5 database file.
        
        Args:
        
            h5rawdbname: h5-based database file name with raw peak picked msi data. 
            
            rawdataset_names: raw dataset names.
                        
            h5dbname: h5-based database file name with processed peak picked msi data.
                 
        """     
        
        print('\nCalculating common m/z feature vector ...' )  
        h5file = h5py.File(h5rawdbname,'r')
        mz = []    
        for datasetid in rawdataset_names:
           try:
               imz = h5file[pathinh5 + datasetid+h5readpath][()]
               if len(self.params['lockmz'])>0 and is_number(self.params['lockmz'][0]):
                   imz = self.calibrate_mz(imz,self.params['lockmz'],
                                          self.params['mzmaxshift'],self.params['mzunits'])
               mz.append(imz)
           except:
               pass
        h5file.close()
        
        # transform list into one dimensional array (i.e. vector)
        mz    = np.concatenate(mz)
        
        # transform list into one dimensional array (i.e. vector)
        if (units=='Da') or (units=='ppm'):
             mzres = self.params['cmzbinsize']
             
        self.cmz = self.get_reference(mz, mzres, self.params['mzmaxshift'], units)
        if len(self.params['lockmz'])>0 and is_number(self.params['lockmz'][0]):
            self.cmz = self.calibrate_mz(self.cmz,self.params['lockmz'],
                                          self.params['mzmaxshift'],self.params['mzunits'])
        print('Completed!' )  
    
    @staticmethod      
    def calibrate_mz(mz, lockmz, mzmaxshift, mzunits):
      
        """
        Performs external lock mass correction of m/z feature vector using a vector ("lockmz") of known masses.
        
        Args:
        
            lockmz: an array of known masses. 
                        
            mz: feature vector for alignment.
            
            maxshift: maximum allowed positional shift.  
                 
        Returns:
    
            refmzidcs: matched indices from refmz feature vector
            
            mzindcs: matached indices from mz feature vector
                 
        """     
        
        if mzunits=='Da':
            refmzidcs, mzindcs = PeakAlign.pmatch_nn(lockmz,mz,mzmaxshift)
            if len(refmzidcs)>0 and is_number(refmzidcs[0]):
                mzdev = np.median(lockmz[refmzidcs] - mz[mzindcs])
                mz = mz + mzdev
        elif mzunits=='ppm':
            lockppm = PeakAlign.to_ppm(lockmz)
            ppm = PeakAlign.to_ppm(mz)
            refmzidcs, mzindcs = PeakAlign.pmatch_nn(lockppm,ppm,mzmaxshift)
            if len(refmzidcs)>0 and is_number(refmzidcs[0]):
                ppmdev = np.median(lockppm[refmzidcs] - ppm[mzindcs])
                ppm = ppm + ppmdev
                mz = PeakAlign.to_mz(ppm)
        return mz
               
    @staticmethod
    def pmatch_nn(refmz,mz,maxshift):
        
        """
        Performs nearest neighbour matching of mz or css feature vector to the reference one.
        
        Args:
        
            refmz: reference mz feature vector. 
                        
            mz: feature vector for alignment.
            
            maxshift: maximum allowed positional shift.  
                 
        Returns:
    
            refmzidcs: matched indices from refmz feature vector
            
            mzindcs: matached indices from mz feature vector
                 
        """     
        try:    
            refmz   = refmz.flatten()
            mz      = mz.flatten()
            nvrbls  = len(refmz)
            mzindcs = np.round(np.interp(mz, refmz, np.arange(0., nvrbls)))
            mzindcs = (mzindcs.astype(int)).flatten()
            
            filtindcs = np.asarray(np.nonzero(np.abs(refmz[mzindcs]-mz)<=maxshift))
            filtindcs = (filtindcs.astype(int)).flatten()
            refmzidcs = np.unique(mzindcs[filtindcs])
            refmzidcs = np.asarray(refmzidcs)
            mzbins    = np.hstack([np.min(refmzidcs)-0.5,refmzidcs.flatten()+.5])
            freq      = np.histogram(mzindcs[filtindcs], bins=mzbins)
            mzrepidcs = refmzidcs[freq[0].astype(int)>1]
            
            mzfilt = mzindcs[filtindcs]
            mz = mz[filtindcs]
            uniqmzidx = (np.ones([1, len(mzfilt)])).flatten()
            for i in mzrepidcs:
                imzdx = ((np.asarray(np.nonzero(i == mzfilt))).astype(int)).flatten()
                minidx = (np.abs(mz[imzdx]-refmz[i])).argmin()
                uniqmzidx[imzdx] = 0.
                uniqmzidx[imzdx[minidx]] = 1.
            
            uniqmzidx = (np.asarray(np.nonzero(uniqmzidx==1.))).flatten()
            mzindcs = filtindcs[uniqmzidx]
        except:
            refmzidcs=[]
            mzindcs=[]        
        
        return refmzidcs, mzindcs
    
    
    @staticmethod
    def get_reference(mz, mzres, mzmaxshift, mzunits):
       
       """
        Computes reference feature vector with respect to which all datasets
        are aligned or matched by means of kernal density estimation.
        
        Args:
        
            dbfilepath: Path to a hdf5-based database file with multiple deposited msi datasets 
                        for peak alignment. 
                        
            datasetids: Paths to datasets in the db file.
            
            alignObj: Object containing peak alignment parameters and method's choices.  
              
            window: Frame length for sliding window, ``10`` data points by default.
             
            weighting: Weighting scheme for smoothing ``'tricubic'`` (default), ``'gaussian'`` or ``'linear'``.
                 
        Returns:
                 rmz: Reference mz or ccs vector for alignment.
                 
        """     
       debug_mode = 0
           
       if mzunits=='Da':
           rconst = .5
       elif mzunits=='ppm':
           rconst = 1
           mz = PeakAlign.to_ppm(mz)
       else:
           rconst = 1
           pass
                          
       mzmin = np.min(mz)-100*mzres
       mzmax = np.max(mz)+100*mzres
       nbins = np.round((mzmax-mzmin)/mzres)+1
       nbins = nbins.astype(int)
       
        # convert mz to bin indices for efficient histogram calculations
       idx     = np.round(np.divide((nbins-1)*(mz - mzmin), (mzmax-mzmin) ,dtype=float)+ rconst)
       idx     = idx.astype(int)
        
        # construct histogram {bin index, frequency} in an efficient way
       histvals = np.bincount(idx,minlength=nbins)         
       nbins    = len(histvals)
       histidx  = np.arange(1,nbins+1,1);       
        # convert histogram indices to original units
       histmz = np.divide((mzmax-mzmin)*(histidx-rconst),nbins-1,dtype=float)+mzmin
        
        # increased detection of the centroid peak accuraccy (upto 4 decimal points)
       if mzunits=='Da':
           iconst = np.max([mzres*1000,1])
       else:
           iconst = mzres
       
        # smooth the histogram
       if debug_mode==1:
           histvals_or = histvals
       
       histvals    = smooth1D(histmz,histvals); 
       histintidx  = np.arange(1,nbins*iconst+1,1);
       inthistmz   = np.divide((mzmax-mzmin)*(histintidx-rconst),len(histintidx)-1,dtype=float)+mzmin    
       
       # interpolation to improve centroid accuracy detection
       histintvals = interpolate.pchip_interpolate(histmz, histvals, inthistmz);
       histintvals[histintvals<0] = 0
       #thrval= get_threshold(histintvals, nbins='')
       thrval= median_threshold(histintvals)
       histintvals[histintvals<thrval] = 0
       
        # detect accurately centroids (add minor disturbance)   
       histintvals[histintvals>thrval] = 1000*histintvals[histintvals>thrval] + 10**(-6) * np.cumsum(np.random.uniform(size=np.sum(histintvals>0)))
     
       # detect accurately centroids
       # a = histintvals[2:]<histintvals[1:-1]
       # b = histintvals[1:-1]>histintvals[:-2]    
       # c = a.astype(int) + b.astype(int) 
       # maxidx = np.array(np.where(c==2))+1
       
       # more advanced strategy to detect accurately centroids
       maxidx = PeakAlign.__findpeaks(histintvals,np.ceil(mzmaxshift))
       refmz  = np.divide((mzmax-mzmin)*(maxidx-rconst),len(histintidx)-1,dtype=float)+mzmin 
       refmz  = refmz.flatten()
       
       # increased detection of the centroid peak accuraccy (upto 4 decimal points)
       if mzunits=='ppm':
           refmz = PeakAlign.to_mz(refmz)   
       
       if debug_mode==1:
            # visualize histogram if debug mode is on
          import matplotlib.pyplot as plt
          histvals= histvals/np.max(histvals)
          histvals = histvals*np.max(histvals_or)
          histintvals = interpolate.pchip_interpolate(histmz, histvals, inthistmz);      
          mzvis = histintvals[maxidx]
          mzvis = mzvis.flatten()
          
          if mzunits=='ppm':
              histmz    = PeakAlign.to_mz(histmz)
              inthistmz = PeakAlign.to_mz(inthistmz)
              
          plt.plot(histmz,histvals,'g')
          plt.plot(histmz,histvals_or,'bo')
          plt.plot(refmz,mzvis,'ro')
          plt.show()
          
       return refmz
    
    @staticmethod    
    def to_mz(value):
         return np.exp(value*1.0e-6)*1.00794;
    
    @staticmethod
    def to_ppm(value):
         return np.log(value/1.00794)*1.0e6;   
 
    @staticmethod
    def __findpeaks(sp, gap=3, int_thr = None):

        """
        Returns a vector of the local peak maxima (peaks) of the input signal vector
        
        Args:
            
                sp: input signal vector (e.g. spectral or chromatographic data)
                    
                gap: the minimum gap between peaks (in data points)
                
                int_thr: intensity threshold (the data are assumed to be smoothed)
        
        Returns:
                
                peakindcs: a vector of the local peak maxima indices
        """      
        # number of data points
        gap = int(gap)
        ndp = len(sp)      
        x = np.zeros(ndp+2*gap)      
        x[:gap] = sp[0]-1.e-6      
        x[-gap:] = sp[-1]-1.e-6      
        x[gap:gap+ndp] = sp      
        peak_candidate = np.zeros(ndp)      
        peak_candidate[:] = True
        
        for s in range(gap):      
            # staring
            start = gap - s - 1
            h_s = x[start : start + ndp]            
            # central
            central = gap
            h_c = x[central : central + ndp]            
            # ending
            end = gap + s + 1
            h_e = x[end : end + ndp]            
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_s, h_c > h_e))
        
        peakindcs = np.argwhere(peak_candidate).flatten()              
        if int_thr is not None:
            peakindcs = peakindcs[sp[peakindcs] > int_thr]
        
        return peakindcs    
     
if __name__ == "__main__": 
    tic()
    settings=OptionsHolder(__doc__, PeakAlign_options);
    settings.description='Peak Alignment';
    settings.do='yes';
    print(settings.program_description);
    settings.parse_command_line_args();
    print(settings.format_parameters());
    print('\nStarting.....')

    do_alignment(settings.parameters['h5rawdbname'], \
                 h5dbname = settings.parameters['h5dbname'], \
                 method = settings.parameters['method'], \
                 params = settings.parameters['params']);
    
    print('\nFinished on %s in'%(time.strftime("%a, %d %b %Y at %H:%M:%S")));   
    toc()  
    print(settings.description_epilog);
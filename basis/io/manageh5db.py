# -*- coding: utf-8 -*-
"""

************************
HDF5-database management
************************

The module includes a set of methods for organization, management and rapid retrieval of MSI data 
via HDF5-based chunked layouts

"""

import h5py
import sys
import os
from basis.utils.typechecker import is_string, iteritem 
import basis.io.getimage as gim
import numpy as np
import re
  
def save_dataset(dbfilepath, pathinh5, data, chunksize = '', compression_opts = ''):
    
    pathinh5 = re.sub('//','/',pathinh5)
    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file_group = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        h5file_group = dbfilepath
        isdbfile=0
    else:
        return
    
    try:        
        isdata = pathinh5 in h5file_group
    except:
        return
        
    if isdata ==True:
        fdata = h5file_group[pathinh5]
        fdata[...] = data
        return
        
    if (not chunksize) and (not compression_opts):
        h5file_group.create_dataset(pathinh5,data=data)
    elif (chunksize) and (compression_opts):
        h5file_group.create_dataset(pathinh5,data=data, chunks = chunksize, 
                                     compression = "gzip", compression_opts = compression_opts)
    elif (chunksize):
        h5file_group.create_dataset(pathinh5,data=data, chunks = chunksize )
    elif (compression_opts):
        h5file_group.create_dataset(pathinh5,data=data, chunks = True, 
                                     compression = "gzip", compression_opts = compression_opts)
        
    if isdbfile==1:
        h5file_group.close()
    return

def load_dataset(dbfilepath, pathinh5):
    
    pathinh5 = re.sub('//','/',pathinh5)
    
    dataset=[]    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file_group = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        h5file_group = dbfilepath
        isdbfile=0
    else:
        return dataset
    
    try:        
        isdata = pathinh5 in h5file_group
    except:
        return dataset
    
    if isdata ==True:
        dataset = h5file_group[pathinh5][()]
        
    if isdbfile==1:
        h5file_group.close()
    
    return dataset
        
def save_preproc_obj(dbfilepath, ProcObj, pathinh5 =''):
    
    """
    **Saves the pre-processing parameters of a module into the hdf5 database.**
    
    Args: 
        
        dbfilepath: the name and path to the hdf5-database file
        
        ProcObj: the pre-processing workflow object
        
        pathinh5: the path in the hdf5 file for object storage
        
    """ 
    
    h5objpath = pathinh5 + ProcObj.description
    h5objpath = re.sub('//','/',h5objpath)
    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file_group = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        h5file_group = dbfilepath
        isdbfile=0
    else:
        return

    try:    
        objvars = vars(ProcObj)
    except:
        return

    try:
        isgroup = h5objpath in h5file_group
    except:
        return
    
    if isgroup==False:
        h5file_group.create_group(h5objpath)
    else:
        print('%s object has already been saved into the database file' %h5objpath)
        return
    
    
    h5obj = h5file_group[h5objpath]
    for i_name in objvars.keys():
        subobj = objvars[i_name]
        if isinstance(subobj,dict):
            h5obj.create_group(i_name)
            h5subobj = h5obj[i_name]
            for j_name in subobj.keys():
                save_dataset(h5subobj,j_name,subobj[j_name])
        else:                
            save_dataset(h5obj,i_name,objvars[i_name])
    
    print('\n%s from pre-processing workflow have been saved to --> %s' %(h5objpath,str(dbfilepath)))
    
    if isdbfile==1:
        h5file_group.close()
    
def load_preproc_obj(dbfilepath,procid, pathinh5 =''):
    
    """
    
    **Loads the pre-processing parameters of a module from the hdf5 database.**
    
    Args: 
        
        dbfilepath: the name and path to the hdf5-database file
        
        procid: the module identifier
        
        pathinh5: the path in the hdf5 file for object storage
        
    """ 
    
    h5objpath = pathinh5 +procid
    h5objpath = re.sub('//','/',h5objpath)
    
    ProcObj = {}
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file_group = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        h5file_group = dbfilepath
        isdbfile=0
    else:
        return ProcObj
        
    try:
        isobj = h5objpath in h5file_group
    except:
        return ProcObj
        
    if isobj==False:
        return ProcObj
    # check whether this object is part of the preprocessing workflow
    h5obj = h5file_group[h5objpath]
    for i_name in h5obj.keys():
        if isinstance(h5obj[i_name],h5py.Group):
            h5subobj = h5obj[i_name]
            subProcObj = {}
            for j_name in h5subobj.keys():
                 subProcObj[j_name] = load_dataset(h5subobj,j_name)
            ProcObj[i_name] = subProcObj
        else:
            ProcObj[i_name] = load_dataset(h5obj,i_name)
              
    if isdbfile==1:
        h5file_group.close()

    return ProcObj

def get_dataset_names(dbfilepath, dbroot='', dataset_names=[]):
    """
    
    Recursively exctracts dataset names from hdf5 database
    
    """    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file = h5py.File(dbfilepath,'r')
        item   = h5file
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        item = dbfilepath
        isdbfile=0
    else:
        return dataset_names
        
    for key, val in iteritem(dict(item)):
        try: 
            subitem = dict(val)
            if ('Xraw' in subitem) or ('Spraw' in subitem) or ('X' in subitem) or ('Sp' in subitem):
                success = 1
            else:
                success = 0
        except:
            success = 0
        if success==1:
            dataset_names.append(val.name)
        elif isinstance(val,h5py.Group):
            dbroot = dbroot + val.name
            dataset_names = get_dataset_names(val,dbroot,dataset_names)
    if isdbfile==1:
        h5file.close()

    return dataset_names

def get_traindata_names(dbfilepath, dbroot='', dataset_names=[], istrain=1):
    """
    
    Recursively exctracts dataset names from hdf5 database
    
    """    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file = h5py.File(dbfilepath,'r')
        item   = h5file
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        item = dbfilepath
        isdbfile=0
    else:
        return dataset_names
        
    for key, val in iteritem(dict(item)):
        try: 
            subitem = dict(val)
            if ('istrain' in subitem) and ('Sp' in subitem):
                if load_dataset(item,val.name+'/istrain')==istrain:
                    success = 1
                else:
                    success = 0
            else:
                success = 0
        except:
            success = 0
        if success==1:
            dataset_names.append(val.name)
        elif isinstance(val,h5py.Group):
            dbroot = dbroot + val.name
            dataset_names = get_traindata_names(val,dbroot,dataset_names,istrain)
    if isdbfile==1:
        h5file.close()

    return dataset_names
 
 
def print_structure_h5db(dbfilepath, dbroot = '', offset='    ') :
    """Prints the HDF5 database structure"""
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file = h5py.File(dbfilepath,'r')
        item   = h5file
        isdbfile=1
    elif (isinstance(dbfilepath,h5py.File)) or (isinstance(dbfilepath,h5py.Group)): 
        item = dbfilepath
        isdbfile=0
    else:
        return 
    
    if  isinstance(item,h5py.File):
       print(item.file, '(File)', item.name)
 
    elif isinstance(item,h5py.Dataset):
        print('(Dataset)', item.name, '    len =', item.shape) #, g.dtype
 
    elif isinstance(item,h5py.Group):
        print('(Group)', item.name)
 
    else:
        print('Warning: The item type is unkown', item.name)
        sys.exit ( "execution is terminated" )
 
    if isinstance(item, h5py.File) or isinstance(item, h5py.Group):
        for key,val in dict(item).iteritems() :
            subitem = val
            print(offset, key) #,"   ", subg.name #, val, subg.len(), type(subg),
            dbroot = dbroot+'i'
            print_structure_h5db(subitem,  dbroot = dbroot, offset = '    ')
    
    if isdbfile==1:
       h5file.close()

def conv_dict2strlist(d):
    fields = d.keys()
    s = []    
    for field in fields:
        s.append(field)
        vals = d[field]
        if isinstance(vals,list):
            if len(vals)>2:
                vals = vals[:2]
        s.append(str(vals))
    h5slist = [n.encode("ascii", "ignore") for n in s]
    return h5slist


def save_preproc2matlab(dbfilepath,do,index, name, method, params):
    # this is a temporaly function to ensure that the preprocessing workflow is comparable with the one implemented in matlab
    pathinh5='/preproc'    
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif isinstance(dbfilepath,h5py.File): 
        h5file = dbfilepath
        isdbfile=0
    else:
        return
    try:    
        ginfo = h5file.create_group(pathinh5+'/'+name)
        ginfo.create_dataset('index',data=int(index))
        ginfo.create_dataset('methodnames',data=method)
        ginfo.create_dataset('do',data=int(do=='yes'))
        ginfo.create_dataset('selected',data=int(1))
        if isinstance(params,dict):
            params = conv_dict2strlist(params)
        ginfo.create_dataset('params',(len(params),1),'S10',params)
    except:
        pass
    if isdbfile==1:
        h5file.close()
    return      

def save_data2matlab(dbfilepath,israw=0):
    if is_string(dbfilepath) and (os.path.exists(dbfilepath)):
        h5file = h5py.File(dbfilepath,'a')
        isdbfile=1
    elif isinstance(dbfilepath,h5py.File): 
        h5file = dbfilepath
        isdbfile=0
    else:
        return
    datasets = np.unique(get_dataset_names(h5file,dataset_names=[]))    
    cmz  = load_dataset(h5file,'cmz')
    ncmz = len(cmz) 
    faileddatasetindcs = []
    cmzminlims = np.zeros((1,ncmz));
    cmzminlims[:] = np.inf
    cmzmaxlims = np.zeros((1,ncmz));
    mztics     = np.zeros((1,ncmz));
    print('\n\n...Configuring database file %s for matlab upload...\n'%os.path.basename(dbfilepath))
    
    i = 0
    for datasetid in datasets:
        try:
            # chunked data for fast retrieval of spectra            
            if israw == 0:            
                X  = load_dataset(h5file,datasetid+'/Sp')
            else: 
                X  = load_dataset(h5file,datasetid+'/Spraw')
                
            xy = load_dataset(h5file,datasetid+'/xy')
            
            xmin = np.nanmin(X,axis=1)
            cmzminlims = np.vstack([xmin[:],cmzminlims[:]])
            xmax = np.nanmax(X,axis=1)
            cmzmaxlims = np.vstack([xmax[:],cmzmaxlims[:]])
            mztics = mztics + np.nanmean(X,axis=1);
            faileddatasetindcs.extend([0])
            
            X,xy2D  = gim.conv2Dto3DX(X.T,xy.T)
            ncmz,nrows,ncols = X.shape 
            # chunked data for fast retrieval of images
            if israw == 0:         
                save_dataset(h5file,datasetid + '/X',data = X, chunksize=(1,nrows,ncols),compression_opts=4)
            else:
                save_dataset(h5file,datasetid + '/Xraw',data=X, chunksize=(1,nrows,ncols),compression_opts=4)
       
            save_dataset(h5file,datasetid + '/xy2D',data = np.transpose(xy2D))
            save_dataset(h5file,datasetid + '/sizeX',data = [ncols,nrows])
           
            print('%s. %s: Successfully configured for matlab basis upload --> %s' %(str(i+1),datasetid, os.path.basename(dbfilepath)))
        except:
            print('%s. %s:  Failed to be configured for matlab basis upload' %(str(i+1),  datasetid))
            pass
        i = i + 1
     
    cmzminlims = np.nanmin(cmzminlims,axis=0)
    cmzmaxlims = np.nanmax(cmzmaxlims,axis=0)
    cmzlimits = np.vstack([cmzminlims[:],cmzmaxlims[:]])
    
    save_dataset(h5file,'faileddatasetindcs',faileddatasetindcs)
    save_dataset(h5file,'cmzlimits',np.transpose(cmzlimits))
    save_dataset(h5file,'climits',[np.nanmin(cmzminlims),np.nanmax(cmzmaxlims)])
    save_dataset(h5file,'mztics',mztics)
    if isdbfile==1:
        h5file.close()
    return

    
if __name__ == "__main__":
    # python manageh5.py dbfilepath
    if len(sys.argv)==2:     
        arg_strs = str(sys.argv[1:])
        arg_strs = ''.join(arg_strs)
        #print kwargs    
        print_structure_h5db(str(arg_strs))
    sys.exit ( "End of import" )
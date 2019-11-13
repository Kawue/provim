import numpy as np
import pandas as pd
from sys import argv
import os
from os.path import isdir, dirname, basename, join
import fnmatch

def join_dataframes(df1, df2):
    intersect = set(df1.columns) & set(df2.columns)
    if intersect:
        df2 = df2.drop(columns=list(intersect))
    return pd.concat([df1, df2], axis=1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('4th Argument expects Boolean.')

def read_h5_files(path):
    script_dct = {
        "matrix_preprocessing.py": "*_simplified.h5",
        "matrix_postprocessing.py": "*_simplified.h5",
        "workflow_peakpicking.py": "*_matrixremoved.h5",
        "msi_dimension_reducer.py": "*_autopicked.h5",
        "automated_matrix_detection.py": "*_simplified.h5",
        "automated_matrix_detection_test.py": "*_simplified.h5"
    }
    h5_files = []
    fnames = []

    if "*" in path:
        identifier = basename(path)
    else:
        identifier = script_dct[basename(argv[0])]
        
    if isdir(path):
        dirpath = path
        dirflag = True
        print("Directory was given. Every HDF5 file in the given directory will be used.")
    else:
        dirpath = dirname(path)
        dirflag = False
    for r, ds, fs in os.walk(dirpath):
        for f in fs:
            if dirflag:
                if ".h5" in f:
                    h5_files.append(pd.read_hdf(join(r,f)))
                    fnames.append(f.split(".")[0])
            else:
                if fnmatch.fnmatchcase(f, identifier):
                    h5_files.append(pd.read_hdf(join(r,f)))
                    fnames.append(f.split(".")[0])
    #print("Directory was given. Every HDF5 file in the given directory will be used.")
    #else:
    #    h5_files.append(pd.read_hdf(argv[1]))
    #    fnames.append(basename(argv[1]).split(".")[0])

    if len(h5_files) == 0:
        raise ValueError("No HDF5 data sets found!")
    
    for h5_fileX in h5_files:
        for h5_fileY  in h5_files:
            if (h5_fileX.columns != h5_fileY.columns).any():
                raise ValueError("mz values of datasets are not normalized!")
    
    return h5_files, fnames

if __name__ == "__main__":
    df1 = pd.read_hdf(argv[1])
    df2 = pd.read_hdf(argv[2])
    join_dataframes(df1, df2).to_hdf(os.path.join(argv[3], argv[4]), key=argv[-1], complib="blosc", complevel=9)



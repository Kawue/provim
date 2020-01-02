import numpy as np
import pandas as pd
from sys import argv
import os
from os.path import isfile, isdir, dirname, basename, join, split
import fnmatch
import h5py

'''
# What the heck is this for? Maybe meant to be s.symmetric_difference(t) (s ^ t) so that no nan columns are created
def join_dataframes(df1, df2):
    intersect = set(df1.columns) & set(df2.columns)
    if intersect:
        df2 = df2.drop(columns=list(intersect))
    return pd.concat([df1, df2], axis=1)
'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('4th Argument expects Boolean.')

def read_h5_files(pathlist):
    script_dct = {
        "matrix_preprocessing.py": ["*processed_simplified.h5", "*_cleaned.h5"],
        "automated_matrix_detection.py": ["*_simplified.h5", "*_cleaned.h5"],
        "interactive_matrix_detection.py": ["*_simplified.h5", "*_cleaned.h5"],
        "matrix_postprocessing.py": ["*processed_simplified.h5", "*_cleaned.h5"],
        "workflow_peakpicking.py": ["*_matrixremoved.h5", "*_tumor.h5"],
        "msi_dimension_reducer.py": ["*_simplified.h5", "*_matrixremoved.h5", "*_autopicked.h5"],
        "msi_image_writer.py": ["*_simplified.h5", "*_matrixremoved.h5", "*_autopicked.h5"]
    }
    h5_files = []
    fnames = []
    paths = []

    for path in pathlist:
        if "*" in path:
            identifiers = [basename(path)]
        else:
            identifiers = script_dct[basename(argv[0])]
        print(os.path.split(path))
        if isfile(path):
            dirpath = dirname(path)
            dirflag = False
        elif isdir(path) or "*" in path:
            if "*" in path:
                dirpath = os.path.split(path)[0]
            else:
                dirpath = path
            dirflag = True
            print("Directory was given. Every matching HDF5 file in the given directory will be used.")
        else:
            raise ValueError("Something went terribly wrong. (Exit 1)")
        
        for root, dirs, files in os.walk(dirpath):
            for f in files:
                if dirflag:
                    for identifier in identifiers:
                        if fnmatch.fnmatchcase(f, identifier):
                            if len(h5py.File(join(root,f)).keys()) < 2:
                                h5_files.append(pd.read_hdf(join(root,f)))
                                fnames.append(f.split(".")[0])
                                paths.append(root)
                else:
                    if split(path)[1] == f:
                        if len(h5py.File(join(root,f)).keys()) < 2:
                            h5_files.append(pd.read_hdf(join(root,f)))
                            fnames.append(f.split(".")[0])
                            paths.append(root)
            #else:
            #    raise ValueError("Given File does not exist.")

    if len(h5_files) == 0:
        raise ValueError("No HDF5 data sets found!")

    if dirflag:
        to_remove = []
        #to_remove2 = []
        for idx, fname in enumerate(fnames):
            f = fname.split(".h5")[0]
            #f_list2 = [x for i, x in enumerate(fnames) if f in x and paths[idx] == paths[i]]

            f_list = [os.path.join(paths[i], x) for i, x in enumerate(fnames) if f == "_".join(x.split("_")[:-1]) and paths[idx] == paths[i]]
            #print()
            #print(f_list)
            #print(f_list2)
            #print()
            if len(f_list) > 0:
                if basename(argv[0]) == "matrix_postprocessing.py":
                    remove_idx = (idx-1) % len(fnames)
                else:
                    print(idx)
                    print(f_list)
                    remove_idx = idx
                to_remove.append(remove_idx)
                #to_remove2.append(fnames.index(fname))
        
        #print()
        #print("##########")
        #print(to_remove)
        #print(len(fnames))
        #print("##########")
        #print()
        #oldtest = fnames[:]
        
        fnames = [x for i, x in enumerate(fnames) if i not in to_remove]
        h5_files = [x for i, x in enumerate(h5_files) if i not in to_remove]
        paths = [x for i, x in enumerate(paths) if i not in to_remove]

        #print()
        #print("##########")
        #print([os.path.join(paths[i], x) for i, x in enumerate(fnames)])
        #print(len(fnames))
        #print()
        #print(set(oldtest) - set(fnames))
        #print("##########")
        #print()
    
    #for h5_fileX in h5_files:
    #    for h5_fileY  in h5_files:
    #        if (h5_fileX.columns != h5_fileY.columns).any():
    #            raise ValueError("mz values of datasets are not normalized!")

    return h5_files, fnames, paths

'''
if __name__ == "__main__":
    df1 = pd.read_hdf(argv[1])
    df2 = pd.read_hdf(argv[2])
    join_dataframes(df1, df2).to_hdf(os.path.join(argv[3], argv[4]), key=argv[-1], complib="blosc", complevel=9)
'''
import numpy as np
import pandas as pd
import h5py
from sys import argv
from pyimzml.ImzMLWriter import ImzMLWriter
from os.path import join, basename#, dirname
import os
import sys
import argparse                                                
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base

class PybasisDataSimplifier:
    def __init__(self, filepath):
        self.h5file = h5py.File(filepath, "r")
        #self.savepath = savepath
        self.dset_keys = h5Base.get_dataset_names(filepath)
        self.dframes = {}
        for dset_key in self.dset_keys:
            self.dframes[dset_key] = self.create_dframe(dset_key)
        if len(self.dset_keys) > 1:
            if not "Internormalization Settings" in self.h5file.keys():
                print("Multiple data sets are stored but no internormalization was applied. This can lead to downstream analysis problems!")
        self.h5file.close()


    def create_dframe(self, key):
        h5groups = self.h5file[key]
        grid_x = np.array(h5groups["xy"])[0].astype(int)
        grid_y = np.array(h5groups["xy"])[1].astype(int)
        mzs = np.array(h5groups["mz"]) # mzraw for unprocessed data
        data = np.array(h5groups["Sp"]).T # Spraw for unprocessed data
        indices = pd.MultiIndex.from_arrays([grid_x, grid_y], names=("grid_x", "grid_y"))
        dframe = pd.DataFrame(data, index=indices, columns=mzs)
        dframe = dframe.assign(dataset=key[1:]).set_index("dataset", append=True)
        return dframe

    def cut_dframe(self, dframe, limits):
        #left = dframe.columns[np.argmin(np.array(np.abs(dframe.columns - limits[0])))]
        #right = dframe.columns[np.argmin(np.array(np.abs(dframe.columns - limits[1])))]
        left = np.argmin(np.array(np.abs(dframe.columns - limits[0])))
        right = np.argmin(np.array(np.abs(dframe.columns - limits[1])))
        dframe = dframe.iloc[:, left:right+1]
        return dframe

    
    def write_simplified_h5(self, dframe, savepath, name):
        dframe.to_hdf(join(savepath, name + "_processed_simplified" + ".h5"), key=name, complib="blosc", complevel=9)


    def write_imzml(self, dframe, savepath, name):
        with ImzMLWriter(join(savepath, name + "_processed_simplified")) as writer:
            for coords, series in dframe.iterrows():
                mzs = np.array(series.index)
                intensities = series.values
                writer.addSpectrum(mzs, intensities, (coords[0], coords[1]))
        #writer.close()

    def merge_dframes(self, dframes):
        if not isinstance(dframes, dict):
            raise ValueError("Please provide DataFrames as dicts with key as name and DataFrame as value.")

        merged_dframe = pd.DataFrame()

        for name, dframe in dframes.items():
            dframe = dframe.assign(dataset=name[1:]).set_index("dataset", append=True)
            merged_dframe = merged_dframe.append(dframe)
        return merged_dframe

def print_statistics(dframe, name):
    mean_spec = dframe.mean(axis=0)
    print("Mean spectrum intensity statistics for data set: %s" % name)
    print("1p Intensity: %f" % (np.percentile(mean_spec, 1)))
    print("5p Intensity: %f" % (np.percentile(mean_spec, 5)))
    print("25p Intensity: %f" % (np.percentile(mean_spec, 25)))
    print("Mean Intensity: %f" % (np.mean(mean_spec)))
    print("75p Intensity: %f" % (np.percentile(mean_spec, 75)))
    print("95p Intensity: %f" % (np.percentile(mean_spec, 95)))
    print("99p Intensity: %f" % (np.percentile(mean_spec, 99)))
    print("Max Intensity: %f" % (np.amax(mean_spec)))
    print("m/z Range: (%f, %f)" % (dframe.columns[0], dframe.columns[-1]))
    print("Number of m/z channels: %i" % (len(dframe.columns)))
    print("Number of pixels: %i" % (len(dframe.index)))
    gy = dframe.index.get_level_values("grid_y").astype(int).max()+1
    gx = dframe.index.get_level_values("grid_x").astype(int).max()+1
    offset_gy = dframe.index.get_level_values("grid_y").astype(int).min()
    offset_gx = dframe.index.get_level_values("grid_x").astype(int).min()
    print("m/z channel image size (%i, %i)" % (gy, gx))
    print("m/z channel image offset (%i, %i)" % (offset_gy, offset_gx))


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to the processed pyBASIS file.")
    parser.add_argument("-s", "--savepath", type=str, required=False, help="Folder to save the output files. (Default equals filepath.)")
    parser.add_argument("-ms", "--mass_range_start", type=float, help="Mass Range Start. Omit to use the whole measure range.")
    parser.add_argument("-me", "--mass_range_end", type=float, help="Mass Range End. Omit to use the whole measure range.")
    parser.add_argument("--write_merge", action='store_true', help="Writes an additional HDF5, imzML and ibd file that merges all selected data sets.")
    args = parser.parse_args()
    
    filepath = args.filepath
    
    if args.savepath is not None:
        savepath = args.savepath
    else:
        savepath = os.path.dirname(filepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    try:
       mz_limits = (float(args.mass_range_start), float(args.mass_range_end))
    except:
        mz_limits = None

    try:
        write_merge = args.write_merge
    except:
        write_merge = False
        raise Warning("Third argument was omitted. Therefore no merged h5 and imzML will be created.")

    Simplifier = PybasisDataSimplifier(filepath)

    for key, dframe in Simplifier.dframes.items():
        key = key[1:]
        if mz_limits is not None:
            dframe = Simplifier.cut_dframe(dframe, mz_limits)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        Simplifier.write_simplified_h5(dframe, savepath, key)
        Simplifier.write_imzml(dframe, savepath, key)
        print_statistics(dframe, key)

    if len(Simplifier.dframes) > 1 and write_merge:
        merged_name = "-".join([key[1:] for key in Simplifier.dframes.keys()])
        merged_dframe = Simplifier.merge_dframes(Simplifier.dframes)
        if mz_limits is not None:
            merged_dframe = Simplifier.cut_dframe(merged_dframe, mz_limits)
        Simplifier.write_simplified_h5(merged_dframe, savepath, merged_name)
        Simplifier.write_imzml(merged_dframe, savepath, merged_name)
        print_statistics(merged_dframe, merged_name)

    # remove pyBASIS HDF file
    #os.remove(argv[1])
    #os.remove(argv[1].replace("_processed.h5", "_raw.h5"))





import numpy as np
import pandas as pd
from sys import argv
from os.path import join, basename, isdir
import os
from scipy.stats.mstats import winsorize
import matplotlib
import matplotlib.pyplot as plt
from easypicker import Easypicker
from msi_image_writer import MsiImageWriter
from interactive_peak_threshold import InteractivePeakPickingThresholder
from msi_utils import read_h5_files
import argparse


# Pick peaks, deisotope them and write them as HDF5 file
def pick_deisotope(dframe, t, iso_range, winsorize, transform, fnames, fname, savepath, interactive, meanframe=None):
    def create_dframe_on_merged(dframes, Picker):
        picked_dframes = []
        for idx, df in enumerate(dframes):
            picked_df = pd.DataFrame([], index=df.index)
            mz_pairs = zip(Picker.deiso_peaks_dict["left"], Picker.deiso_peaks_dict["right"])
            for left, right in mz_pairs:
                interval = Picker.mzs[left:right+1]
                interval = [i for i in interval if i in df.columns]
                #picked_df[np.median(interval)] = df[interval].sum(axis=1)
                picked_df[np.median(interval)] = df[interval].max(axis=1)
            print(f"Peak Number of Dataset {fnames[idx]} equals {picked_df.shape[1]}")
            picked_dframes.append(picked_df)
        return picked_dframes

    
    if interactive:
        normalize = True
        if meanframe is None:
            PeakPickingThresholder = InteractivePeakPickingThresholder(dframe, winsorize, normalize, fname, savepath)
        else:
            PeakPickingThresholder = InteractivePeakPickingThresholder(meanframe, winsorize, normalize, fname, savepath)
        PeakPickingThresholder.plot()
        Picker = PeakPickingThresholder.run(None)
        if meanframe is None:
            picked_dframes = [Picker.create_dframe(deisotoped=True)]
        else:
            picked_dframes = create_dframe_on_merged(dframe, Picker)
    else:
        if meanframe is None:
            Picker = Easypicker(dframe, winsorize)
        else:
            Picker = Easypicker(meanframe, winsorize)
        Picker.find_peaks(t)
        Picker.deisotope((iso_range[0], iso_range[1]))
        if meanframe is None:
            picked_dframes = [Picker.create_dframe(deisotoped=True)]
        else:
            picked_dframes = create_dframe_on_merged(dframe, Picker)


    for idx, picked_dframe in enumerate(picked_dframes):
        if transform == "sqrt":
            picked_dframes[idx] = picked_dframe.applymap(lambda x: np.sqrt(x) if x > 0 else 0)
        elif transform == "log":
            picked_dframes[idx] = picked_dframe.applymap(lambda x: np.log(x + 1))
        elif transform == "nonneg":
            picked_dframes[idx] = picked_dframe.applymap(lambda x: 0 if x < 0 else x)
        elif transform == "none":
            pass
        else:
            raise ValueError("Bug in transfor argument!")
    #print(fnames)
    for idx, picked_dframe in enumerate(picked_dframes):
        if meanframe is None:
            wording = "_autopicked"
            picked_dframe.to_hdf(join(savepath, fname + wording + ".h5"), key=fname+wording, complib="blosc", complevel=9)
            if quality_control_flag:
                quality_control_routine(picked_dframe, fname, savepath, Picker)
        else:
            wording = "_autopicked_on_merge"
            picked_dframe.to_hdf(join(savepath, fnames[idx] + wording + ".h5"), key=fnames[idx]+wording, complib="blosc", complevel=9)
            if quality_control_flag:
                quality_control_routine(picked_dframe, fnames[idx], savepath, Picker)
        

        

    # Deprecated
    #if len(set(picked_dframe.index.get_level_values("dataset"))) > 1:
    #    picked_dframe.to_hdf(join(savepath,"merged_autopicked.h5"), key="merged_autopicked", complib="blosc", complevel=9)

'''
def merge_dframes(dframes):
    merged_dframe = pd.DataFrame()
    for idx, dframe in enumerate(dframes):
        #dframe = dframe.assign(dataset=fnames[idx]).set_index("dataset", append=True)
        merged_dframe = merged_dframe.append(dframe)
    return merged_dframe
'''
def merged_mean(dframes):
    mzs = sorted(list(set(np.array([dframe.columns.tolist() for dframe in dframes]).flatten())))
    intensities = []
    for mz in mzs:
        intensities.append(np.mean(np.concatenate([dframe[mz].values for dframe in dframes if mz in dframe.columns])))
    return pd.DataFrame([intensities], columns=mzs)
    

def quality_control_routine(dframe, fname, savepath, Picker):
    # dframe should be picked at this point
    subsavepath = join(savepath, fname)

    #for dset in set(dframe.index.get_level_values("dataset")):
    #    subframe = dframe.iloc[dframe.index.get_level_values("dataset") == dset]
    # Write mz images
    #    ImageWriter = MsiImageWriter(subframe, join(savepath, "images", dset))
    #    ImageWriter.write_msi_imgs()

    # Plot mean spectrum and picked peaks
    plt.figure(figsize=(16,9))
    plt.title(fname)
    plt.plot(Picker.mzs, Picker.mean_spec, label="Spectrum")
    plt.plot(Picker.peaks_mzs, Picker.mean_spec[Picker.peaks_idx], "yo", label="Picked")
    plt.plot(Picker.deiso_peaks_mzs, Picker.mean_spec[Picker.deiso_peaks_idx], "r*", label="Deisotoped")
    plt.hlines(Picker.peaks_dict["rel_height"], Picker.mzs[Picker.peaks_dict["left"]], Picker.mzs[Picker.peaks_dict["right"]])
    plt.legend()
    plt.savefig(subsavepath, bbox_inches='tight')
    plt.close()

    # Write csv of picked peaks
    mean_spectrum = np.mean(dframe, axis=0)
    mean_spectrum.index = np.round(mean_spectrum.index, 3)
    mean_spectrum.to_csv(subsavepath + ".csv", sep=",", index_label=["mz"], header = ["intensity"])



########## Begin Workflow ##########
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--readpath", type=str, required=True, nargs='+', help="Path to h5 files.")
parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save output.")
parser.add_argument("-w", "--winsorize", type=int, required=False, default=5, help="Maximum peak intensity. The w'th highest peak will be used as upper limit. Default is 5.")
parser.add_argument("--pick_on_merge", required=False, action='store_true', help="True to pick on union of all data sets, False otherwise. Default is False.")
parser.add_argument("-t", "--transformation", type=str, required=False, choices=["sqrt", "log", "none"], default="log", help="Spectrum transformation (sqrt, log, none). Default is log.")
parser.add_argument("--interactive", required=False, action='store_true', default=False, help="Start interactive tool for treshold adjustment. If True, deisotoping and threshold parameters are ignored and taken from tool settings.")
parser.add_argument("-d", "--deisotoping", type=float, nargs='+', required=False, default=[0.85, 1.15], help="Dalton range tuple to search for isotopes. Use 0 0 to deactivate. Default is 0.85 1.15.")
parser.add_argument("-p", "--threshold", type=float, required=False, default=[0], nargs="+", help="Intensity threshold for peak picking. If a folder with multiple data sets is selected either one or as many thresholds as data sets have to be provided.")
parser.add_argument("-q", "--quality_control", required=False, action='store_true', help="Saves a mean spectrum plot for quality control purposes. Default is True.")
args=parser.parse_args()

readpath = args.readpath
savepath = args.savepath

h5_files, fnames, paths = read_h5_files(readpath)

for i, dframe in enumerate(h5_files):
    dframe_norm = (dframe - dframe.min().min()) / (dframe.max().max()-dframe.min().min())
    h5_files[i] = dframe_norm

    #print(dframe_norm.min().min())
    #print(dframe_norm.max().max()) 

def set_savepath(path, idx, paths=paths):
    if path:
        savepath = path
    else:
        savepath = paths[idx]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath

quality_control_flag = args.quality_control
winsorize = args.winsorize

pick_on_merged = args.pick_on_merge
if pick_on_merged == False:
    print("pick_on_merge argument is False. If the HDF5 stores more than one data set picking will be done one every single one independent!")

transform = args.transformation

interactive = args.interactive

iso_range = args.deisotoping
if len(iso_range) != 2:
    raise ValueError("-d / --deisotoping must be a tuple of two values. First value is min range, second is max range.")

t = args.threshold
if len(t) == 0:
    if not interactive:
        raise ValueError("If --interactive is disabled thresholds must be provided!")

if len(t) > 1:
    if len(fnames) != len(t):
        raise Exception("For multiple data sets the number of thresholds must be one or match the number of data sets.")


if pick_on_merged:
    if len(h5_files) == 1:
        dframe = h5_files[0]
        if len(set(dframe.index.get_level_values("dataset"))) < 2:
            print("It seems like the HDF5 file has only one data set stored.")
        else:
            dframe_ids = list(set(dframe.index.get_level_values("dataset")))
            h5_files = [dframe.loc[dframe.index.get_level_values("dataset") == dframe_id] for dframe_id in dframe_ids]
            fnames = dframe_ids
    meanframe = merged_mean(h5_files)
    pick_deisotope(h5_files, t[0], iso_range, winsorize, transform, fnames, "merged", set_savepath(savepath, 0), interactive, meanframe)
else:
    if len(h5_files) == 1:
        dframe = h5_files[0]
        dframe_ids = list(set(dframe.index.get_level_values("dataset")))
        fnames = dframe_ids
        if len(dframe_ids) > 1:
            for idx, dframe_id in enumerate(dframe_ids):
                selected_dframe = dframe.loc[dframe.index.get_level_values("dataset") == dframe_id]
                pick_deisotope(selected_dframe, t[idx], iso_range, winsorize, transform, fnames[idx], fnames, set_savepath(savepath, idx), interactive)
        else:
            pick_deisotope(dframe, t[0], iso_range, winsorize, transform, fnames, fnames[0], set_savepath(savepath, 0), interactive)
    else:
        for idx, dframe in enumerate(h5_files):
            if interactive:
                pick_deisotope(dframe, t[0], iso_range, winsorize, transform, fnames, fnames[idx], set_savepath(savepath, idx), interactive)
            else:
                pick_deisotope(dframe, t[idx], iso_range, winsorize, transform, fnames, fnames[idx], set_savepath(savepath, idx), interactive)
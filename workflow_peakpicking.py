import numpy as np
import pandas as pd
from sys import argv
from os.path import join, basename, isdir
import os
from scipy.stats.mstats import winsorize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from easypicker import Easypicker
from msi_image_writer import MsiImageWriter
from interactive_peak_threshold import InteractivePeakPickingThresholder
from msi_utils import str2bool, read_h5_files
import argparse


# Pick peaks, deisotope them and write them as HDF5 file
def pick_deisotope(dframe, t, iso_range, winsorize, transform, fname, savepath, interactive):
    if interactive:
        normalize = True
        PeakPickingThresholder = InteractivePeakPickingThresholder(dframe, winsorize, normalize, fname, savepath)
        PeakPickingThresholder.plot()
        Picker = PeakPickingThresholder.run(None)
        picked_dframe = Picker.create_dframe(deisotoped=True)
    else:
        Picker = Easypicker(dframe, winsorize)
        Picker.find_peaks(t)
        Picker.deisotope(iso_range)
        picked_dframe = Picker.create_dframe(deisotoped=True)

    if transform == "sqrt":
        picked_dframe = picked_dframe.applymap(lambda x: np.sqrt(x) if x > 0 else 0)
    elif transform == "log":
        picked_dframe = picked_dframe.applymap(lambda x: np.log(x) if x > 1 else 0)
    elif transform == "none":
        pass
    else:
        raise ValueError("Bug in transfor argument!")
    
    for dset in set(picked_dframe.index.get_level_values("dataset")):
        subframe = picked_dframe.iloc[picked_dframe.index.get_level_values("dataset") == dset]
        subframe.to_hdf(join(savepath, dset + "_autopicked.h5"), key=dset+"_autopicked", complib="blosc", complevel=9)

    if len(set(picked_dframe.index.get_level_values("dataset"))) > 1:
        picked_dframe.to_hdf(join(savepath,"merged_autopicked.h5"), key="merged_autopicked", complib="blosc", complevel=9)


    # Hier gut aufgehoben?
    # Hat den Vorteil, dass Picker verfügbar ist und quality control für jedes dframe angewendet wird, das erzeugt wird.
    # TODO: Hier könnte noch ein Seiteneffekt mit merged sein ?!
    if quality_control_flag:
        quality_control_routine(picked_dframe, fname, savepath, Picker)


def merge_dframes(dframes):
    merged_dframe = pd.DataFrame()
    for idx, dframe in enumerate(dframes):
        #dframe = dframe.assign(dataset=fnames[idx]).set_index("dataset", append=True)
        merged_dframe = merged_dframe.append(dframe)
    return merged_dframe


def quality_control_routine(dframe, fname, savepath, Picker):
    # dframe should be picked at this point
    subsavepath = join(savepath, fname)

    for dset in set(dframe.index.get_level_values("dataset")):
        subframe = dframe.iloc[dframe.index.get_level_values("dataset") == dset]
        # Write mz images
        ImageWriter = MsiImageWriter(subframe, join(savepath, "images", dset))
        ImageWriter.write_msi_imgs()

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
parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to h5 files.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
parser.add_argument("-w", "--winsorize", type=int, required=False, default=5, help="Maximum peak intensity. The w'th highest peak will be used as upper limit. Default is 5.")
parser.add_argument("-m", "--pickOnMerge", type=str2bool, required=False, default=False, help="True to pick on union of all data sets, False otherwise. Default is False.")
parser.add_argument("-t", "--transformation", type=str, required=False, choices=["sqrt", "log", "none"], default="log", help="Spectrum transformation (sqrt, log, none). Default is log.")
parser.add_argument("-i", "--interactive", type=bool, required=False, default=False, help="Start interactive tool for treshold adjustment. If True, deisotoping and threshold parameters are ignored and taken from tool settings.")
parser.add_argument("-d", "--deisotoping", type=float, required=False, default=1.15, help="Dalton range to search for isotopes. Use 0 to deactivate. Default is 1.15.")
parser.add_argument("-p", "--threshold", type=float, required=True, nargs="+", help="Intensity threshold for peak picking. If a folder with multiple data sets is selected either one or as many thresholds as data sets have to be provided.")
parser.add_argument("-q", "--quality_control", required=False, type=str2bool, default=True, help="Saves a mean spectrum plot for quality control purposes. Default is True.")

args=parser.parse_args()

h5_files, fnames = read_h5_files(args.filepath)
savepath = args.savepath
quality_control_flag = args.quality_control
winsorize = args.winsorize

pick_on_merged = args.pickOnMerge
if pick_on_merged == False:
    print("pickOnMerge argument is False. If the HDF5 stores more than one data set picking will be done one every single one independent!")

transform = args.transformation

iso_range = args.deisotoping
t = args.threshold
if len(t) > 1:
    if len(fnames) != len(t):
        raise Exception("For multiple data sets the number of thresholds must be one or match the number of data sets.")
interactive = args.interactive

for i, h5_fileX in enumerate(h5_files):
    for j, h5_fileY  in enumerate(h5_files):
        try:
            if (h5_fileX.columns != h5_fileY.columns).any():
                print("Unequal mz values in data set pair (%i, %i)"%(i,j))
                raise ValueError("mz values of datasets are not normalized!")
        except Exception as e:
            print("Unequal mz values in data set pair (%i, %i)"%(i,j))
            raise ValueError(e)

if pick_on_merged:
    if len(h5_files) == 1:
        dframe = h5_files[0]
        if len(set(dframe.index.get_level_values("dataset"))) < 2:
            print("It seems like the HDF5 file has only one data set stored.")
    else:
        dframe = merge_dframes(h5_files)
    pick_deisotope(dframe, t[0], iso_range, winsorize, transform, "merged", savepath, interactive)
else:
    if len(h5_files) == 1:
        dframe = h5_files[0]
        dframe_ids = list(set(dframe.index.get_level_values("dataset")))
        if len(dframe_ids) > 1:
            for idx, dframe_id in enumerate(dframe_ids):
                selected_dframe = dframe.loc[dframe.index.get_level_values("dataset") == dframe_id]
                pick_deisotope(selected_dframe, t[idx], iso_range, winsorize, transform, dframe_id, savepath, interactive)
        else:
            pick_deisotope(dframe, t[0], iso_range, winsorize, transform, fnames[0], savepath, interactive)
    else:
        for idx, dframe in enumerate(h5_files):
            pick_deisotope(dframe, t[idx], iso_range, winsorize, transform, fnames[idx], savepath, interactive)
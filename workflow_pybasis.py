import sys
from sys import argv
from os.path import join
from basis.io import importmsi, exportmsi
from basis.preproc import palign, intranorm, internorm, vst, pfilter
from basis.utils.msmanager import H5BaseMSIWorkflow as h5Base
from msi_utils import str2bool
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dirpath", type=str, required=True, help="Path to folder with imzML files.")
parser.add_argument("-n", "--name", type=str, required=False, default="basis.h5", help="Name of the pyBASIS file. Default is 'basis.h5'.")
parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save pybasis HDF5 output.")
parser.add_argument("-i", "--instrument", type=str, required=False, choices=["tof", "orbitrap"], default="orbitrap", help="MSI insturment. Default is Orbitrap. Only determines some default parameters and can be used for other instruments as well.")

parser.add_argument("--multisample", action='store_true', help="Use if multiple samples are processed to include internormalization, False otherwise.")

parser.add_argument("--intranorm_method", type=str, required=False, choices=["mfc", "mean", "median"], default="mfc", help="Method for intra normalization. Default is Median Fold Change.")
parser.add_argument("--intranorm_offset", type=float, required=False, default=0.0, help="Disregard peak intensity smaller that this value. Default is 0.")
parser.add_argument("--intranorm_outliers", type=str, required=False, choices=["yes", "no"], default="yes", help="If 'yes' adjust outlying values after range estimation. Default is 'yes'.")

parser.add_argument("--internorm_method", type=str, required=False, choices=["mfc", "mean", "median"], default="mfc", help="Method for inter normalization. Default is Median Fold Change.")
parser.add_argument("--internorm_offset", type=float, required=False, default=0.0, help="Disregard peak intensity smaller that this value. Default is 0.")
parser.add_argument("--internorm_outliers", type=str, required=False, choices=["yes", "no"], default="no", help="If 'yes' adjust outlying values after range estimation. Default is 'no'.")

args = parser.parse_args()

datafolder = args.dirpath
savefolder = args.savepath
if args.savepath:
    savefolder = args.savepath
else:
    savefolder = datafolder


h5name = args.name
h5name_raw = join(savefolder, h5name.split(".")[0] + "_raw_pybasis." + h5name.split(".")[-1])
h5name_processed = join(savefolder, h5name.split(".")[0] + "_processed_pybasis." + h5name.split(".")[-1])
instrument = "orbitrap"
filetype = "imzML"
fileext = ".imzML"
multisample = args.multisample

if args.instrument == "tof":
    # import parameters
    mzunits = "Da"
    mzmaxshift = 0.1
    cmzbinsize = 0.01

if args.instrument == "orbitrap":
    # import parameters
    mzunits = "Da"
    mzmaxshift = 0.1
    cmzbinsize = 0.001

# palign parameters
palign_method = "NN"
lock_mz = ""

# intranorm parameters
intranorm_method = args.intranorm_method
intranorm_reference = "mean"
intranorm_offset = args.intranorm_offset
intranorm_outliers = args.intranorm_outliers # Adjust outlying values after range estimation
intranorm_min_mz = 0.0
intranorm_max_mz = 50000.0 # Just a high number to consider always the whole spectrum.

# internorm parameters
internorm_method = args.internorm_method
internorm_reference = "mean"
internorm_offset = args.internorm_offset
internorm_outliers = args.internorm_outliers
internorm_min_mz = 0.0
internorm_max_mz = 50000.0 # Just a high number to consider always the whole spectrum.


importmsi.do_import(datafolder, h5name_raw, filetype=filetype, params={"mzunits": mzunits, "mzmaxshift": mzmaxshift, "cmzbinsize": cmzbinsize, "fileext": fileext})
print("\nimzML file imported.\n")

palign.do_alignment(h5name_raw, h5dbname=h5name_processed, method=palign_method, params={"mzunits": mzunits, "mzmaxshift": mzmaxshift, "cmzbinsize": cmzbinsize, "lockmz": lock_mz})
print("\nPeak alignment completed.\n")

intranorm.do_normalize(h5dbname=h5name_processed, method=intranorm_method, params={"offset": intranorm_offset, "reference": intranorm_reference}, mzrange=[intranorm_min_mz, intranorm_max_mz])
print("\nIntra data set normalization completed.\n")

if multisample:
    internorm.do_normalize(h5dbname=h5name_processed, method=internorm_method, params={"offset": internorm_offset, "reference": internorm_reference}, mzrange=[internorm_min_mz, internorm_max_mz])





# variance stabilization transformation parameters
#vst_method = "started-nonneg"
#vst_offset = "auto"
#vst_winsorize = [0.01, 1]
# matrix peak removal parameters
#mpr_method = "kmeans"
#mpr_nclusters = 2

#vst.do_vst(h5dbname=h5name_processed, method=vst_method, params={"offset": vst_offset, "winsorize": vst_winsorize})
#print("\nVariance stabilization transformation completed.\n")

#pfilter.do_filtering(h5dbname=h5name_processed, method=mpr_method, params={"nclusters": mpr_nclusters})
#print("\nMatrix peak removal completed.\n")

#l1 = np.where(np.array(h5file["Peak Filtering Settings"]["labels"]) == 0)[0]
#l2 = np.where(np.array(h5file["Peak Filtering Settings"]["labels"]) == 1)[0]
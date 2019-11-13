import numpy as np
import pandas as pd
from sys import argv
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to h5 files.")
parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
parser.add_argument("-w", "--winsorize", type=int, required=False, default=5, help="Maximum peak intensity. The w'th highest peak will be used as upper limit. Default is 5.")
args=parser.parse_args()

filepath = args.filepath
dframe = pd.read_hdf(filepath)
savepath = args.savepath
winsorize_limit = args.winsorize
name = os.path.basename(filepath).split(".")[0]

data = dframe.values
t = np.sort(data, axis=None)[-winsorize_limit]

data[data > t] = t

dframe[:] = data

dframe.to_hdf(os.path.join(savepath, name)+"_winsorized-%i.h5"%
	winsorize_limit), key=name+"-winsorized-%i"%winsorize_limit, complib="blosc", complevel=9)
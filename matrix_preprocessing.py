import numpy as np
import argparse
from sys import argv
from os.path import join
import os
from msi_dimension_reducer import UMAP, LSA
from msi_utils import read_h5_files
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from msi_utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filepath", required=True, type=str, help="Path to _processed_simplified.h5 file.")
    parser.add_argument("-s", "--savepath", required=True, type=str, help="Path to save the output files.")
    parser.add_argument("-q", "--quality_control", required=False, type=str2bool, help="Saves a mean spectrum plot for quality control purposes.")
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath

    h5_files, fnames = read_h5_files(filepath)

    for idx, h5_file in enumerate(h5_files):
        print(fnames[idx])
        print(savepath)
        subpath = join(savepath, fnames[idx])
        print(subpath)
        print("---")
        if not os.path.isdir(subpath):
            os.makedirs(subpath)

        if args.quality_control:
            plt.figure(figsize=(16,9))
            plt.title("Full Spectrum " + fnames[idx])
            plt.plot(h5_file.columns, h5_file.mean(axis=0))
            plt.savefig(join(subpath, "Full Spectrum " + fnames[idx] + ".png"), bbox_inches='tight')
            plt.close()

        if h5_file.shape[0] > 1000:
            # This could also be PCA
            lsa = LSA(h5_file.values, n_components=1000)
            pre_embedded = lsa.perform()
            #umap = UMAP(pre_embedded, n_components=2, min_dist=0.2)
            umap = UMAP(pre_embedded, n_components=2, min_dist=0.05, n_neighbors=200, metric="cosine")
            embedding = umap.perform()
            np.save(join(savepath, fnames[idx] + "_embedding.npy"), embedding)
        else:
            #umap = UMAP(h5_file.values, n_components=2, min_dist=0.2)
            umap = UMAP(h5_file.values, n_components=2, min_dist=0.05, n_neighbors=200, metric="cosine")
            embedding = umap.perform()
            np.save(join(savepath, fnames[idx] + "_embedding.npy"), embedding)
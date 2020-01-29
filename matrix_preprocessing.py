import numpy as np
import argparse
from sys import argv
import os
from msi_dimension_reducer import UMAP, LSA
from msi_utils import read_h5_files
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--readpath", required=True, type=str, nargs='+', help="Path to _processed_simplified.h5 files (Folder or File/s).")
    parser.add_argument("-s", "--savepath", required=False, type=str, default=False, help="Path to save the output files. (Default equals readpath.)")
    parser.add_argument("--quality_control", required=False, action='store_true', help="Saves a mean spectrum plot for quality control purposes.")
    args = parser.parse_args()
    readpath = args.readpath
    savepath = args.savepath

    h5_files, fnames, paths = read_h5_files(readpath)

    def set_savepath(path, idx, paths=paths):
        if path:
            savepath = path
        else:
            savepath = paths[idx]
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        return savepath

    for idx, h5_file in enumerate(h5_files):
        if not os.path.isdir(set_savepath(savepath, idx)):
            os.makedirs(set_savepath(savepath, idx))

        if args.quality_control:
            plt.figure(figsize=(16,9))
            plt.title("Full Spectrum " + fnames[idx])
            plt.plot(h5_file.columns, h5_file.mean(axis=0))
            if not os.path.exists(os.path.join(set_savepath(savepath, idx), "quality-control")):
                os.makedirs(os.path.join(set_savepath(savepath, idx), "quality-control"))
            plt.savefig(os.path.join(set_savepath(savepath, idx), "quality-control", "Full Spectrum " + fnames[idx] + ".png"), bbox_inches='tight')
            plt.close()

        if h5_file.shape[1] > 1000:
            # This could also be PCA
            lsa = LSA(h5_file.values, n_components=1000)
            pre_embedded = lsa.perform()
            #umap = UMAP(pre_embedded, n_components=2, min_dist=0.2)
            umap = UMAP(pre_embedded, n_components=2, min_dist=0.05, n_neighbors=200, metric="cosine")
            embedding = umap.perform()
            np.save(os.path.join(set_savepath(savepath, idx), fnames[idx] + "_embedding.npy"), embedding)
        else:
            #umap = UMAP(h5_file.values, n_components=2, min_dist=0.2)
            umap = UMAP(h5_file.values, n_components=2, min_dist=0.05, n_neighbors=200, metric="cosine")
            embedding = umap.perform()
            np.save(os.path.join(set_savepath(savepath, idx), fnames[idx] + "_embedding.npy"), embedding)
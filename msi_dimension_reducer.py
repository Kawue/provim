import numpy as np
import pandas as pd
import sklearn.decomposition as skd
import sklearn.manifold as skm
import umap as uumap
from sys import argv
import matplotlib.pyplot as plt
from os.path import basename, join
from collections import OrderedDict
import os
import pandas as pd
from msi_image_writer import MsiImageWriter
from msi_utils import read_h5_files, str2bool
import argparse

class DimensionReducer:

    def __init__(self, data, n_components):
        self.data = data
        self.n_components = n_components



class PCA(DimensionReducer):
    def __init__(self, data, n_components, predict_components=False, whiten=False):
        super().__init__(data, n_components)
        self.whiten = whiten
        if predict_components:
            self.n_components = "mle"

    def perform(self):
        pca = skd.PCA(n_components=self.n_components, whiten=self.whiten)
        transform = pca.fit_transform(self.data)
        return transform



class NMF(DimensionReducer):
    def __init__(self, data, n_components, init=None, random_state=None):
        super().__init__(data, n_components)
        if init not in [None, "random"]:
            raise ValueError("init parameter is restricted to None or 'random'.")
        
        self.init = init
        self.random_state = random_state

        if self.n_components > min(data.shape):
            self.init = "random"
            print("The high number of n_components forced the parameter random_state to be set to 'random'.")
            if random_state is None:
                self.random_state = 0
            else:
                self.random_state = random_state

    def perform(self):
        nmf = skd.NMF(n_components=self.n_components, init=self.init, random_state=self.random_state)
        transform = nmf.fit_transform(self.data)
        return transform



class LDA(DimensionReducer):
    def __init__(self, data, n_components, random_state=0):
        super().__init__(data, n_components)
        self.random_state = random_state

    def perform(self):
        lda = skd.LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)
        transform = lda.fit_transform(self.data)
        return transform



class TSNE(DimensionReducer):
    def __init__(self, data, n_components, init="pca", metric="euclidean", random_state=0):
        super().__init__(data, n_components)
        if init not in ["pca", "random"]:
            raise ValueError("init parameter has to be 'pca' or 'random'.")
        if metric not in ["euclidean", "cosine", "correlation", "manhattan", "precomputed"]:
            raise ValueError("metric parameter is restricted to 'euclidean', 'cosine', 'correlation', 'manhattan' or 'precomputed'")
        if metric == "precomputed":
            print("With metric chosen as 'precomputed' data is expected to be a distance matrix!")
            if self.data.shape[0] != self.data.shape[1]:
                raise ValueError("data cannot be a distance matrix as dim[0] != dim[1].")
        self.init = init
        self.metric = metric
        self.random_state = random_state
        if self.n_components > 3:
            self.method = "exact"
        else:
            self.method = "barnes_hut"

    def perform(self):
        tsne = skm.TSNE(n_components=self.n_components, init=self.init, random_state=self.random_state, metric=self.metric, method=self.method)
        transform = tsne.fit_transform(self.data)
        return transform



class UMAP(DimensionReducer):
    def __init__(self, data, n_components, metric="euclidean", n_neighbors=15, min_dist=0.1):
        super().__init__(data, n_components)
        if metric not in ["euclidean", "cosine", "correlation", "manhattan", "precomputed"]:
            raise ValueError("metric parameter is restricted to 'euclidean', 'cosine', 'correlation', 'manhattan' or 'precomputed'")
        if metric == "precomputed":
            print("With metric chosen as 'precomputed' data is expected to be a distance matrix!")
            if self.data.shape[0] != self.data.shape[1]:
                raise ValueError("data cannot be a distance matrix as dim[0] != dim[1].")
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def perform(self):
        umap = uumap.UMAP(n_components=self.n_components, metric=self.metric, n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        transform = umap.fit_transform(self.data)
        return transform



class ICA(DimensionReducer):
    def __init__(self, data, n_components, random_state=0):
        super().__init__(data, n_components)
        self.random_state = random_state
 
    def perform(self):
        ica = skd.FastICA(n_components=self.n_components, random_state=self.random_state)
        transform = ica.fit_transform(self.data)
        return transform



class KPCA(DimensionReducer):
    def __init__(self, data, n_components, kernel="rbf", random_state=0):
        super().__init__(data, n_components)
        if kernel == "linear":
            print("kernel parameter for Kernel PCA was chosen to be 'linear'. The result will be equal to standard PCA.")
        if kernel not in ["linear", "poly", "rbf", "sigmoid", "cosine"]:
            raise ValueError("kernel parameter is restricted to 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'")
        self.kernel = kernel
        self.random_state = random_state

    def perform(self):
        kpca = skd.KernelPCA(n_components=self.n_components, kernel=self.kernel)
        transform = kpca.fit_transform(self.data)
        return transform 



class LSA(DimensionReducer):
    def __init__(self, data, n_components, random_state=0):
        super().__init__(data, n_components)
        self.random_state = random_state

    def perform(self):
        lsa = skd.TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        transform = lsa.fit_transform(self.data)
        return transform



class LLE(DimensionReducer):
    def __init__(self, data, n_components, n_neighbors=5, random_state=0):
        super().__init__(data, n_components)
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def perform(self):
        lle = skm.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components, random_state=self.random_state)
        transform = lle.fit_transform(self.data)
        return transform


class MDS(DimensionReducer):
    def __init__(self, data, n_components, dissimilarity='euclidean', random_state=0):
        super().__init__(data, n_components)
        if dissimilarity not in ["euclidean", "precomputed"]:
            raise ValueError("dissimilarity parameter is restricted to 'euclidean' or 'precomputed'")
        if dissimilarity == "precomputed":
            print("With dissimilarity chosen as 'precomputed' data is expected to be a dissimilarity matrix!")
            if self.data.shape[0] != self.data.shape[1]:
                raise ValueError("data cannot be a distance matrix as dim[0] != dim[1].")
        self.dissimilarity = dissimilarity
        self.random_state = random_state

    def perform(self):
        mds = skm.MDS(n_components=self.n_components, random_state=self.random_state)
        transform = mds.fit_transform(self.data)
        return transform



class Isomap(DimensionReducer):
    def __init__(self, data, n_components, n_neighbors=5):
        super().__init__(data, n_components)
        self.n_neighbors = n_neighbors

    def perform(self):
        isomap = skm.Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components)
        transform = isomap.fit_transform(self.data)
        return transform



class SpectralEmbedding(DimensionReducer):
    def __init__(self, data, n_components, affinity="nearest_neighbors", random_state=0, n_neighbors=None):
        super().__init__(data, n_components)
        if affinity not in ["nearest_neighbors", "rbf"]:
            raise ValueError("affinity parameter is restricted to 'nearest_neighbors' or 'rbf'")
        if affinity != "nearest_neighbors" and n_neighbors is not None:
            raise ValueError("n_neighbors parameter is only usable with affinity set to 'nearest_neighbors'.")
        self.affinity = affinity
        self.random_state = random_state
        self.n_neighbors = n_neighbors


    def perform(self):
        spem = skm.SpectralEmbedding(n_components=self.n_components, affinity=self.affinity, random_state=self.random_state, n_neighbors=self.n_neighbors)
        transform = spem.fit_transform(self.data)
        return transform



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to h5 files.")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
    parser.add_argument("-m", "--method", type=str, required=True, choices=["pca", "nmf", "lda", "tsne", "umap", "ica", "kpca", "lsa", "lle", "mds", "isomap", "spectralembedding"] help="Path to save output.")
    parser.add_argument("-n", "--ncomponents", type=int, required=True, help="Number of dimensions to reduce to.")
    parser.add_argument("--save_plots", type=str2bool, required=False, default=True, help="Save scatterplots and component images of the dimension reduction. Default True.")
    parser.add_argument("--misc_keywords", type=str, required=False, default=[] nargs="+", help="Further parameter names. Use ONLY if you know what the code does! Default settings are provided!")
    parser.add_argument("--misc_params", required=False, default=[] nargs="+", help="Further parameters. Order must match '--misc_keywords'. Use ONLY if you know what the code does! Default settings are provided!")
    parser.add_argument("--grine_output", type=float, required=False, default=[] nargs="+", help="Provides a dimension reduction output file for GRINE.")
    args=parser.parse_args()

    filepath = args.filepath
    savepath = args.savepath
    method = args.method.lower()
    n_components = args.ncomponents
    
    def catch(param):
        try:
            return float(param)
        except:
            return param
            
    misc = {args.misc_keywords[i]: catch(args.misc_params[i]) for i in range(len(args.misc_keywords)}

    scatteralpha = 0.5
    
    method_dict = {
        "pca": PCA,
        "nmf": NMF,
        "lda": LDA,
        "tsne": TSNE,
        "umap": UMAP,
        "ica": ICA,
        "kpca": KPCA,
        "lsa": LSA,
        "lle": LLE,
        "mds": MDS,
        "isomap": Isomap,
        "spectralembedding": SpectralEmbedding
        }

    h5_files, fnames = read_h5_files(filepath)

    if len(h5_files) > 1:
        merged_dframe = pd.DataFrame()
        for idx, dframe in enumerate(h5_files):
            merged_dframe = merged_dframe.append(dframe)
    else:
        merged_dframe = h5_files[0]

    if len(h5_files) > 20:
        print("Currently different colors for up to 20 data sets are supported. If you process more than 20 at a time colors will repeat for different data sets!")

    DR = method_dict[method](merged_dframe.values, n_components, **misc)
    embeddings = DR.perform()
    subsavepath = join(savepath, "dimreduce")
    if not os.path.exists(subsavepath):
        os.makedirs(subsavepath)
    np.save(join(subsavepath, method + "_embeddings"), embeddings)

    tab20 = plt.cm.tab20(np.linspace(0, 1, len(h5_files)))
    dset_names = list(OrderedDict.fromkeys(merged_dframe.index.get_level_values("dataset")))
    colors_dict = {dset: tab20[idx] for idx, dset in enumerate(dset_names)}
    dset_colors = [colors_dict[dset] for dset in merged_dframe.index.get_level_values("dataset")]
    
    if save_plots:
        for i in range(embeddings.shape[1]):
            for j in range(i+1, embeddings.shape[1]):
                subsubsavepath = join(subsavepath, method)
                if not os.path.isdir(subsubsavepath):
                    os.makedirs(subsubsavepath)
                plt.figure(figsize=(16,9))
                plt.title("%s of %s"%(method.upper(), basename(filepath).split(".")[0]))
                for idx, dset_name in enumerate(dset_names):
                    embedding = embeddings[np.where(merged_dframe.index.get_level_values("dataset") == dset_name)]
                    plt.scatter(embedding[:,i], embedding[:,j], c=[colors_dict[dset_name]], label=dset_name, alpha=scatteralpha)
                plt.legend()
                plt.xlabel("Component %i"%(i+1))
                plt.ylabel("Component %i"%(j+1))
                plt.savefig(join(subsubsavepath, "C%i-vs-C%i.png"%(i+1,j+1)), bbox_inches="tight")
                plt.close()

        for idx, dset_name in enumerate(dset_names):
            dframe = merged_dframe.iloc[merged_dframe.index.get_level_values("dataset") == dset_name]
            Writer = MsiImageWriter(dframe, join(subsavepath, fnames[idx]))
            dset_embedding = embeddings[np.where(merged_dframe.index.get_level_values("dataset") == dset_name)]
            Writer.write_dimvis(dimreduce_transform=dset_embedding, n_components=n_components, rgb_indices=[0,1,2], method_name=method)
            
            
    if args.grine_output:
        pass
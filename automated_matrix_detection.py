import numpy as np
import pandas as pd
from os.path import dirname, join
from sys import argv
import matplotlib.pyplot as plt
import sklearn.decomposition as skd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from collections import Counter
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.cluster import DBSCAN, KMeans
from scipy.signal import argrelmin
import umap
#from unidip import UniDip
from msi_utils import read_h5_files
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import argparse


class AutomaticDataCleaner:
    def __init__(self, dframe, embedding, name, savepath, n_neighbors, radius):
        self.dframe = dframe
        self.embedding = embedding
        self.name = name
        self.savepath = savepath
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.selected_cc = np.array([])
        
    
    def fit(self):
        self.knn()
        

    def knn(self):
        matrix_found_flag = None
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, radius=self.radius)
        nn.fit(self.embedding)

        knn = nn.kneighbors_graph()
        knn_cc = connected_components(knn, directed=False)
        print("Connected Components kNN: " + str(knn_cc))
        c = ["g","m","y","orange","k", "pink", "violet"]
        plt.figure()
        plt.plot(self.embedding[:,0], self.embedding[:,1], "ro")
        nr_components = np.amax(knn_cc[1])
        if nr_components < len(c):
            for i in range(nr_components + 1):
                idx = np.where(knn_cc[1]==i)
                plt.plot(self.embedding[idx,0], self.embedding[idx,1], color=c[i], marker="o")
                plt.savefig(join(self.savepath, self.name+"_embedding_connected_components_knn.png"))
            plt.close()
        plt.close()

        rnn = nn.radius_neighbors_graph()
        rnn_cc = connected_components(rnn, directed=False)
        print("Connected Components Radius: " + str(rnn_cc))
        plt.figure()
        plt.plot(self.embedding[:,0], self.embedding[:,1], "ro")
        nr_components = max(rnn_cc[1])
        if nr_components < len(c):
            for i in range(nr_components+1):
                idx = np.where(rnn_cc[1]==i)
                plt.plot(self.embedding[idx,0], self.embedding[idx,1], color=c[i], marker="o")
                plt.savefig(join(self.savepath, self.name+"_connected_components_radius.png"))
            plt.close()
        plt.close()

        # If any cc has only two components, take it.
        # If both have two components check if they are equal, otherwise ...
        # If both have more than two components take the one with less.
        # If the number of components is more than two check if they are equal, otherwise ...
        
        if knn_cc[0] < 2 and rnn_cc[0] < 2:
            print("No Matrix Found")
            matrix_found_flag = False
        elif knn_cc[0] == 2 or rnn_cc[0] == 2:
            matrix_found_flag = True
            if knn_cc[0] == rnn_cc[0]:
                if (knn_cc[1] == rnn_cc[1]).all():
                    self.selected_cc = knn_cc[1]
                else:
                    knn_idx = np.argmin([len(np.where(knn_cc[1] == i)[0]) for i in sorted(list(set(knn_cc[1])))])
                    rnn_idx = np.argmin([len(np.where(rnn_cc[1] == i)[0]) for i in sorted(list(set(rnn_cc[1])))])

                    knn = knn_cc[1] == knn_idx
                    rnn = rnn_cc[1] == rnn_idx

                    self.selected_cc = knn * rnn
            else:
                if knn_cc[0] == 2:
                    self.selected_cc = knn_cc[1]
                else:
                    self.selected_cc = rnn_cc[1]
        else:
            matrix_found_flag = True
            if knn_cc[0] != rnn_cc[0]:
                if knn_cc[0] < rnn_cc[0]:
                    self.selected_cc = knn_cc[1]
                else:
                    self.selected_cc = rnn_cc[1]
            else:
                knn_idx = np.argmax([len(np.where(knn_cc[1] == i)[0]) for i in sorted(list(set(knn_cc[1])))])
                rnn_idx = np.argmax([len(np.where(rnn_cc[1] == i)[0]) for i in sorted(list(set(rnn_cc[1])))])

                knn = knn_cc[1] != knn_idx
                rnn = rnn_cc[1] != rnn_idx

                self.selected_cc = knn * rnn

        print("Connected Components Joint Decision: " + str(self.selected_cc))
        if matrix_found_flag:
            plt.figure()
            plt.plot(self.embedding[:,0], self.embedding[:,1], "ro")
            self.selected_cc = self.selected_cc.astype(int)
            nr_components = max(self.selected_cc)
            if nr_components < len(c):
                for i in range(nr_components+1):
                    idx = np.where(self.selected_cc == i)
                    plt.plot(self.embedding[idx,0], self.embedding[idx,1], color=c[i], marker="o")
                    plt.savefig(join(self.savepath, self.name+"_connected_components_final.png"))
                    plt.close()
            else:
                print("Plot cannot be made, since the actual number of components %i is larger than the number of encoding colors %i."%(nr_components, len(c)))
            plt.close()

    
    def run_clean_remove(self):
        clearframe = self.dframe.iloc[np.where(self.selected_cc == False)[0]]
        matrixframe = self.dframe.iloc[np.where(self.selected_cc == True)[0]]
        clearframe.to_hdf(join(self.savepath, self.name + "_cleaned" + ".h5"), key=self.name + "_nomatrix", complib="blosc", complevel=9)
        matrixframe.to_hdf(join(self.savepath, self.name + "_matrix" + ".h5"), key=self.name + "_matrix", complib="blosc", complevel=9)
        pseudoframe = pd.DataFrame()
        pseudoframe.to_hdf(join(self.savepath, self.name + "_artifacts" + ".h5"), key=self.name + "_artifacts", complib="blosc", complevel=9)
        np.save(join(self.savepath, self.name + "_embedding_cleaned"), self.embedding[np.where(self.selected_cc == False)])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--readpath", type=str, required=True, nargs='+', help="Path to _processed_simplified.h5 file.")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save output.")
    #parser.add_argument("-e", "--embedding", type=str, required=True, help="Path to embedding file (.npy).", default=None)
    parser.add_argument("--n_neighbors", type=int, required=False, help="Number of neighbors to span a n_neighbors graph on the dimension reduction. Default is 15.", default=15)
    parser.add_argument("--n_radius", type=int, required=False, help="Distance radius to span a n_neighbors graph on the dimension reduction. Default is 10.", default=10)
  
    args = parser.parse_args()
    
    readpath = args.readpath
    savepath = args.savepath
    n_neighbors = args.n_neighbors
    n_radius = args.n_radius

    h5_files, fnames, paths = read_h5_files(readpath)

    def set_savepath(path, idx, paths=paths):
        if path:
            savepath = path
        else:
            savepath = paths[idx]
        return savepath

    for idx, h5_file in enumerate(h5_files):
        embedding = np.load(join(paths[idx], fnames[idx]+"_embedding.npy"))
        embedding = embedding**2 * np.sign(embedding)
        cleaner = AutomaticDataCleaner(h5_file, embedding, fnames[idx], set_savepath(savepath, idx), n_neighbors, n_radius)
        cleaner.fit()
        cleaner.run_clean_remove()
    plt.close("all")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
''' 
    #Old Approaches

    def __init__(self, dframe, embedding, name, savepath, n_neighbors, radius):
        self.dframe = dframe
        self.embedding = embedding
        self.name = name
        self.savepath = savepath
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.selected_cc = None
        #print("Data Set Shape: " + str(dframe.shape))
        #self.dists = euclidean_distances(self.embedding)
        #n = 1
        #self.distsflat = self.dists[np.triu_indices_from(self.dists, k=1)]**n
        #print("Number of Distance Values: %i"%(len(self.distsflat)))
        #plt.figure()
        #plt.title("Hist")
        #self.distnumber, self.distbins, _ = plt.hist(self.distsflat, self.dframe.shape[0],alpha=.3,label='data')
        #plt.savefig(join(self.savepath, name+"_histogramm.png"))
        #if self.distsflat.size > 1000000:
        #    self.distsflat = np.random.choice(self.distsflat, size = 1000000, replace=False)
        #self.window_const = int(len(self.distbins)*0.01)
        #print("Bins: %f"%(len(self.distbins)))
    
    def fit(self):
        self.knn()
        #self.dbscan_centroids()
        #self.distance_pdfs()
        #self.ud()
        #self.distance_binning()
    
    def distance_pdfs(self):
        self.density = gaussian_kde(self.distsflat)
        self.densityintensity = self.density(self.distbins)

        plt.figure()
        plt.title("Density")
        plt.plot(self.distbins, self.densityintensity, "b-", label="standard", alpha=0.5)
        plt.savefig(join(self.savepath, self.name+"_density.png"))

        #self.linspaceA = np.linspace(self.densityintensity[np.argmax(self.densityintensity)], self.densityintensity[-1], len(self.densityintensity[np.argmax(self.densityintensity):]))
        #self.kinkA = self.find_kink(self.distbins[np.argmax(self.densityintensity):], self.densityintensity[np.argmax(self.densityintensity):], self.linspaceA)
        #self.linspaceA = np.linspace(self.distnumber[np.argmax(self.distnumber)], self.distnumber[-1], len(self.distnumber[np.argmax(self.distnumber):]))
        #self.kinkB = self.find_kink(self.distbins[np.argmax(self.distnumber):-1], self.distnumber[np.argmax(self.distnumber):], self.linspaceA)
        #self.linspaceA = np.linspace(self.densityintensity[np.argmax(self.densityintensity)], self.densityintensity[-1], len(self.densityintensity[np.argmax(self.densityintensity):]))
        #self.kinkC = self.hproj(self.distbins[np.argmax(self.densityintensity):], self.densityintensity[np.argmax(self.densityintensity):], self.linspaceA)
        #self.linspaceA = np.linspace(self.distnumber[np.argmax(self.distnumber)], self.distnumber[-1], len(self.distnumber[np.argmax(self.distnumber):]))
        #self.kinkD = self.hproj(self.distbins[np.argmax(self.distnumber):-1], self.distnumber[np.argmax(self.distnumber):], self.linspaceA)

    
    def distance_binning(self):   
        self.threshold = self.kinkA

        loc = np.where(self.dists > self.threshold)
        counter = Counter(list(loc[0]) + list(loc[1]))
        self.countervals = np.msort(list(counter.values()))[::-1]
        self.linspace = np.linspace(self.countervals[0], self.countervals[-1], len(self.countervals))
        self.xaxis = np.array(list(range(len(self.linspace))))

        self.kink = self.find_kink(self.xaxis, self.countervals, self.linspace)

        plt.figure()
        plt.title("Counter")
        plt.bar(self.xaxis, self.countervals)
        plt.axvline(x=self.xaxis[self.kink], color="green")
        plt.savefig(join(self.savepath, self.name+"_counts.png"))

        tuples = [(pxidx, count) for pxidx, count in counter.items()]
        tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
        self.clean_idx = [pxidx for pxidx, count in tuples[:self.kink]]

        plt.figure()
        plt.title("embedding")
        plt.plot(self.embedding[:,0], self.embedding[:,1], "ro")
        plt.plot(self.embedding[self.clean_idx,0], self.embedding[self.clean_idx,1], "b.")
        plt.savefig(join(self.savepath, self.name+"_embedding_kink_selection.png"))


    def find_kink(self, xaxis, countervals, linspace):      
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
        def project(r0, point, line):
            dot = np.dot((point - r0), line)
            norm = np.dot(line, line)
            proj = r0 + ((dot/norm) * line)
            return proj

        plt.figure()
        plt.plot(xaxis, countervals)
        plt.plot(xaxis, linspace)
        #plt.axis('equal')

        projlengths = []

        for i in range(0, len(linspace)):
            line = np.array([xaxis[-1], linspace[-1]]) - np.array([xaxis[0], linspace[0]])
            point = np.array([xaxis[i], countervals[i]])
            r0 = np.array([xaxis[i], linspace[i]])
            proj = project(r0, point, line)
            projvec = proj - point
            projlength = np.linalg.norm(projvec)
            projlengths.append(projlength)
            if np.around(np.dot(line,projvec), 5) != 0:
                print("BUG!")
        
        max_idx = np.argmax(projlengths)
        print("Projection Length: %f"%(np.amax(projlengths)))
        
        line = np.array([xaxis[0], linspace[0]]) - np.array([xaxis[-1], linspace[-1]])
        point = np.array([xaxis[max_idx], countervals[max_idx]])
        r0 = np.array([xaxis[max_idx], linspace[max_idx]])
        proj = project(r0, point, line)
        
        plt.plot(point[0], point[1], "ko")
        plt.plot(proj[0], proj[1], "go")
        plt.plot([point[0], proj[0]], [point[1], proj[1]], "r-")
        plt.close()
        #plt.savefig(join(self.savepath, self.name + "_kinkpoint_detection.png"))
        return xaxis[max_idx]


    def hproj(self, xaxis, crossdists, linspace):
        projlengths = []

        for i in range(0, len(linspace)):
            projlengths.append(np.abs(crossdists[i] - linspace[i]))

        max_idx = np.argmax(projlengths)
        print("Projection Length: %f"%(np.amax(projlengths)))
        plt.figure()
        plt.plot(xaxis, crossdists)
        plt.plot(xaxis, linspace)
        plt.plot(xaxis[max_idx], crossdists[max_idx], "k^")
        plt.plot(xaxis[max_idx], linspace[max_idx], "g^")
        plt.plot([xaxis[max_idx], xaxis[max_idx]], [crossdists[max_idx], linspace[max_idx]], "k--")
        plt.close()
        return xaxis[max_idx]



    def dbscan_centroids(self):
        self.dbscan = DBSCAN().fit_predict(self.embedding)

        self.centroids = []
        for i in range(0, np.amax(self.dbscan) + 1):
            self.centroids.append(self.embedding[np.where(self.dbscan == i)].mean(axis = 0))
        plt.figure()
        plt.title("dbscan centroids")
        plt.plot(self.embedding[:, 0], self.embedding[:, 1], "ro")
        self.centroids = np.array(self.centroids)
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], "bx")

        for i in range(0, np.amax(self.dbscan) + 1):
            plt.figure()
            plt.title("dbscan %i"%i)
            plt.plot(self.embedding[:, 0], self.embedding[:, 1], "ro")
            plt.plot(self.embedding[np.where(self.dbscan == i), 0], self.embedding[np.where(self.dbscan == i), 1], "b*")


    def ud(self):
        udip = UniDip(self.distsflat, debug=False)
        intervals = udip.run()
        intervals = UniDip(self.distbins, is_hist=True, debug=False)
        intervals = udip.run()
'''
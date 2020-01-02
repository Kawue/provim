import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib import path
import os
import argparse
from msi_utils import read_h5_files     

class InteractiveDataCleaner:
    def __init__(self, dframe, embedding, name, savepath):
        self.name = name
        self.savepath = savepath
        self.dframe = dframe
        self.gx = np.array(dframe.index.get_level_values("grid_x").astype(int))
        self.gy = np.array(dframe.index.get_level_values("grid_y").astype(int))
        self.embedding = embedding
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.ax1_xmin = np.amin(self.embedding[:,0]) - 1
        self.ax1_ymin = np.amin(self.embedding[:,1]) - 1
        self.ax1_xmax = np.amax(self.embedding[:,0]) + 1
        self.ax1_ymax = np.amax(self.embedding[:,1]) + 1
        self.toggle = "reset"
        self.toggle_props = {
            "reset": {'color': 'black', 'linewidth': 2, 'alpha': 0.8},
            "clean": {'color': 'yellow', 'linewidth': 2, 'alpha': 0.8},
            "matrix": {'color': 'green', 'linewidth': 2, 'alpha': 0.8},
            "roi": {'color': 'orange', 'linewidth': 2, 'alpha': 0.8}
            }
        self.idx = {
            "reset": [],
            "clean": [],
            "matrix": [],
            "roi": [],
            "remain": []
            }
        self.roiname = "next-time-type-a-name-you-idiot"


    def create_baseplot(self):
        self.fig = plt.figure()
        self.fig.suptitle(self.name)
        self.ax1 = plt.axes([0.12, 0.2, 0.4, 0.75])
        self.ax1.set_title("UMAP Embedding")
        self.ax1.set_xlabel("Dimension 1")
        self.ax1.set_ylabel("Dimension 2")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.ax1.set_aspect('equal')    
        
        self.ax2 = plt.axes([0.55, 0.2, 0.4, 0.75])
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img, vmax=1)
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        plt.subplots_adjust(bottom=0.1)
        
        self.ax3 = plt.axes([0.55, 0.05, 0.12, 0.065])
        self.cleanbutton = Button(self.ax3, "Artifacts")
        self.cleanbutton.on_clicked(self.clean)

        self.ax4 = plt.axes([0.68, 0.05, 0.12, 0.065])
        self.removebutton = Button(self.ax4, "Matrix")
        self.removebutton.on_clicked(self.remove)

        self.ax5 = plt.axes([0.81, 0.05, 0.12, 0.065])
        self.runbutton = Button(self.ax5, "Run")
        self.runbutton.on_clicked(self.run_clean_remove)

        self.ax6 = plt.axes([0.05, 0.05, 0.12, 0.065])
        self.roibutton = Button(self.ax6, "ROI")
        self.roibutton.on_clicked(self.roi)

        self.ax7 = plt.axes([0.68, 0.15, 0.12, 0.065])
        self.resetbutton = Button(self.ax7, "Reset")
        self.resetbutton.on_clicked(self.reset)
        
        self.ax9 = plt.axes([0.18, 0.05, 0.12, 0.065])
        self.exportnamefield = TextBox(self.ax9, "", initial="")
        self.exportnamefield.on_submit(self.set_roiname)
        self.ax9.text(0, 1.15, "Type Name")
        
        self.ax8 = plt.axes([0.31, 0.05, 0.12, 0.065])
        self.exportbutton = Button(self.ax8, "Export")
        self.exportbutton.on_clicked(self.export_roi)   
        


    def calc_img(self):
        self.img[:] = 0
        self.img[(self.gy[self.idx[self.toggle]], self.gx[self.idx[self.toggle]])] = 1
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.imshow(self.img, vmax=1)


    def onselect(self, verts):
        p = path.Path(verts)
        self.idx[self.toggle] = p.contains_points(self.pts.get_offsets())
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.ax1.scatter(self.embedding[self.idx["reset"], 0], self.embedding[self.idx["reset"], 1], s=6, c=self.toggle_props["reset"]["color"])
        self.ax1.scatter(self.embedding[self.idx["roi"], 0], self.embedding[self.idx["roi"], 1], s=6, c=self.toggle_props["roi"]["color"])
        self.ax1.scatter(self.embedding[self.idx["clean"], 0], self.embedding[self.idx["clean"], 1], s=6, c=self.toggle_props["clean"]["color"])
        self.ax1.scatter(self.embedding[self.idx["matrix"], 0], self.embedding[self.idx["matrix"], 1], s=6, c=self.toggle_props["matrix"]["color"])
        self.calc_img()
        self.fig.canvas.draw_idle()


    def plot(self):
        self.create_baseplot()
        plt.show()


    def clean(self, event):
        print("clean")
        self.toggle = "clean"
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.idx["reset"] = []
        

    def remove(self, event):
        print("matrix")
        self.toggle = "matrix"
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.idx["reset"] = []

    def roi(self, event):
        print("roi")
        self.toggle = "roi"
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.idx["reset"] = []

    def reset(self, event):
        print("reset")
        self.toggle = "reset"
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.idx["roi"] = []
        self.idx["clean"] = []
        self.idx["matrix"] = []


    def run_clean_remove(self, event):
        if len(self.idx["clean"]) == 0:
            self.idx["clean"] = [False] * len(embedding)
        if len(self.idx["matrix"]) == 0:
            self.idx["matrix"] = [False] * len(embedding)
        self.idx["remain"] = [not x for x in np.array(self.idx["clean"]) + np.array(self.idx["matrix"])]
        self.clearframe = self.dframe[self.idx["clean"]]
        self.matrixframe = self.dframe[self.idx["matrix"]]
        self.dframe = self.dframe[self.idx["remain"]]
        self.clearframe.to_hdf(os.path.join(self.savepath, self.name + "_artifacts" + ".h5"), key=self.name + "_artifacts", complib="blosc", complevel=9)
        self.matrixframe.to_hdf(os.path.join(self.savepath, self.name + "_matrix" + ".h5"), key=self.name + "_matrix", complib="blosc", complevel=9)
        self.dframe.to_hdf(os.path.join(self.savepath, self.name + "_cleaned" + ".h5"), key=self.name + "_cleaned", complib="blosc", complevel=9)
        self.adjust_variables()
        self.adjust_plots()
        np.save(os.path.join(self.savepath, self.name + "_embedding_cleaned"), self.embedding)


    def set_roiname(self, text):
        self.roiname = text

    def export_roi(self, event):
        np.save(os.path.join(self.savepath, self.roiname), self.idx["roi"])


    def adjust_variables(self):
        self.embedding = self.embedding[self.idx["remain"]]
        self.gx = self.gx[self.idx["remain"]]
        self.gy = self.gy[self.idx["remain"]]
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.idx["roi"] = []
        self.idx["clean"]  = []
        self.idx["matrix"] = []
        self.idx["remain"] = []


    def adjust_plots(self):
        self.ax1_xmin = np.amin(self.embedding[:,0]) - 1
        self.ax1_ymin = np.amin(self.embedding[:,1]) - 1
        self.ax1_xmax = np.amax(self.embedding[:,0]) + 1
        self.ax1_ymax = np.amax(self.embedding[:,1]) + 1
        self.ax1.cla()
        self.ax1.set_title("UMAP Embedding")
        self.ax1.set_xlabel("Dimension 1")
        self.ax1.set_ylabel("Dimension 2")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img, vmax=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--readpath", type=str, required=True, nargs='+', help="Path to _processed_simplified.h5 file.")
    #parser.add_argument("-e", "--embedding", type=str, required=True, help="Path to _embedding.npy file.")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save output.")
    args = parser.parse_args()
    
    readpath = args.readpath
    savepath = args.savepath

    h5_files, fnames, paths = read_h5_files(readpath)
    
    def set_savepath(path, idx, paths=paths):
        if path:
            savepath = path
        else:
            savepath = paths[idx]
        return savepath

    for idx, h5_file in enumerate(h5_files):
        embedding = np.load(os.path.join(paths[idx], fnames[idx]+"_embedding.npy"))
        print(fnames[idx])
        plot = InteractiveDataCleaner(h5_file, embedding, fnames[idx], set_savepath(savepath, idx))
        plot.plot()
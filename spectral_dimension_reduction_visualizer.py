import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib import path
import matplotlib as mpl
import os
import argparse

class SpectralDimensionReductionVisualizer:
    def __init__(self, dframe, embedding, name, savepath):
        mpl.use("TkAgg")
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
        self.toggle = "roi"
        self.toggle_props = {
            "roi": {'color': 'orange', 'linewidth': 2, 'alpha': 0.8}
            }
        self.idx = {
            "roi": []
            }


    def create_baseplot(self):
        self.fig = plt.figure()
        self.fig.suptitle(self.name)
        self.ax1 = plt.axes([0.12, 0.2, 0.4, 0.75])
        self.ax1.set_title("Embedding")
        self.ax1.set_xlabel("Dimension 1")
        self.ax1.set_ylabel("Dimension 2")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue", alpha=0.7)
        self.ax1.set_aspect('auto', 'box')
        #self.ax1.set_aspect('equal') 
        
        self.ax2 = plt.axes([0.55, 0.2, 0.4, 0.75])
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img, vmax=1)
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        plt.subplots_adjust(bottom=0.1)

        self.ax3 = plt.axes([0.68, 0.15, 0.12, 0.065])
        self.resetbutton = Button(self.ax3, "Reset")
        self.resetbutton.on_clicked(self.reset)


    def calc_img(self):
        self.img[:] = 0
        self.img[(self.gy, self.gx)] = 1
        self.img[(self.gy[self.idx[self.toggle]], self.gx[self.idx[self.toggle]])] = 2
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img)


    def onselect(self, verts):
        p = path.Path(verts)
        self.idx[self.toggle] = p.contains_points(self.pts.get_offsets())
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue", alpha=0.7)
        self.ax1.scatter(self.embedding[self.idx["roi"], 0], self.embedding[self.idx["roi"], 1], s=6, c=self.toggle_props["roi"]["color"], alpha=0.7)
        self.calc_img()
        self.fig.canvas.draw_idle()


    def plot(self):
        self.create_baseplot()
        plt.show()

    def reset(self, event):
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.pts.remove()
        self.adjust_variables()
        self.adjust_plots()


    def adjust_variables(self):
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.idx["roi"] = []


    def adjust_plots(self):
        self.ax1_xmin = np.amin(self.embedding[:,0]) - 1
        self.ax1_ymin = np.amin(self.embedding[:,1]) - 1
        self.ax1_xmax = np.amax(self.embedding[:,0]) + 1
        self.ax1_ymax = np.amax(self.embedding[:,1]) + 1
        self.ax1.cla()
        self.ax1.set_title("Embedding")
        self.ax1.set_xlabel("Dimension 1")
        self.ax1.set_ylabel("Dimension 2")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue", alpha=0.7)
        
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datapath", type=str, required=True, help="Path to .h5 file.")
    parser.add_argument("-e", "--embeddingpath", type=str, required=True, help="Path to embedding .npy file.")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save output.")
    args = parser.parse_args()
    
    datapath = args.datapath
    embeddingpath = args.embeddingpath
    savepath = args.savepath

    dframe = pd.read_hdf(datapath)
    
    def set_savepath(path, readpath=os.path.dirname(datapath)):
        if path:
            savepath = path
        else:
            savepath = readpath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        return savepath

    name = dframe.index.get_level_values("dataset")[0]
    embedding = np.load(embeddingpath)
    plot = SpectralDimensionReductionVisualizer(dframe, embedding, name, set_savepath(savepath))
    plot.plot()
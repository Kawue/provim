import numpy as np
import pandas as pd
from sys import argv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib import path
from os.path import join, basename
import os
from easypicker import Easypicker
import time
from msi_utils import str2bool
import argparse

class InteractivePeakPickingThresholder:
    def __init__(self, dframe, aggregation, winsorize, normalize, name, savepath):
        mpl.use('TkAgg')
        self.dframe = dframe
        self.name = name
        self.savepath = savepath
        self.data = dframe.values
        self.winsorize = winsorize
        self.normalize = normalize
        self.aggregation = aggregation
        if self.aggregation == "mean":
            self.aggr_spec = np.mean(self.data, axis=0)
        elif self.aggregation == "median":
            self.aggr_spec = np.median(self.data, axis=0)
        elif self.aggregation == "max":
            self.aggr_spec = np.max(self.data, axis=0)
        else:
            raise ValueError("Parameter Error (aggregation) in InteractivePeakPickingThresholder.")
        #self.aggr_spec = np.array([np.log(x) if x > 1 else 0 for x in np.mean(self.data, axis=0)])
        if self.winsorize > 0:
            if type(self.winsorize) != int:
                raise ValueError("winsorize has to be of type int!")
            self.winsorize_limit = sorted(self.aggr_spec)[-self.winsorize]
            self.aggr_spec[self.aggr_spec > self.winsorize_limit] = self.winsorize_limit
        if self.normalize:
            mi = self.aggr_spec.min()
            ma = self.aggr_spec.max()
            self.aggr_spec = (self.aggr_spec - mi) / (ma - mi)
        self.mzs = dframe.columns
        self.active = True
        self.txtoffset = 0.01
        self.peak_nb = 0
        self.deiso_range_min = 0.8
        self.deiso_range_max = 1.2
        self.deiso_nb = 0
        self.Picker = None

    def create_baseplot(self):
        self.fig = plt.figure()
        self.fig.suptitle(self.name)
        self.ax1 = plt.axes([0.1, 0.15, 0.75, 0.7])
        self.ax1.set_title("Mean Spectrum")
        self.ax1.set_xlabel("Mass Channel")
        self.ax1.set_ylabel("Intensity")
        self.lineplot = self.ax1.plot(self.mzs, self.aggr_spec, c="blue", zorder=1)
        self.line = self.ax1.axhline(y=1, color="red", alpha=0)
        self.peakdots = self.ax1.scatter([], [], s=10, c="red", zorder=2)
        self.deisodots = self.ax1.scatter([], [], s=10, c="red", zorder=3)

        self.ax2 = plt.axes([0.87, 0.15, 0.1, 0.7])
        self.ax2.set_ylim(*self.ax1.get_ylim())
        self.ax2.set_axis_off()
        self.linetxt = self.ax2.text(0,0,"")

        

        self.ax3 = plt.axes([0.48, 0.01, 0.1, 0.055])
        self.ax3.text(-1.3, 0.4, "Deiso Range")
        self.minnamefield = TextBox(self.ax3, "Min", initial=str(self.deiso_range_min))
        self.minnamefield.on_submit(self.set_deiso_range_min)

        self.ax4 = plt.axes([0.64, 0.01, 0.1, 0.055])
        self.maxnamefield = TextBox(self.ax4, "Max", initial=str(self.deiso_range_max))
        self.maxnamefield.on_submit(self.set_deiso_range_max)

        self.ax5 = plt.axes([0.75, 0.01, 0.1, 0.055])
        self.deisobutton = Button(self.ax5, "Deiso")
        self.deisobutton.on_clicked(self.deiso)

        self.ax6 = plt.axes([0.86, 0.01, 0.1, 0.055])
        self.runbutton = Button(self.ax6, "Run")
        self.runbutton.on_clicked(self.run)
        
        self.peaktxt = self.ax3.text(-3.7, 0.9, "Number of Peaks: %i"%(self.peak_nb))
        self.deisotxt = self.ax3.text(-3.7, 0.3, "Number of Deiso: %i"%(self.deiso_nb))
      
        def onMouseMove(event):
            if event.inaxes == self.ax1:
                if 0.0 < event.ydata < 1.0 and self.mzs[0] < event.xdata < self.mzs[-1]:
                    self.line.remove()
                    self.linetxt.remove()
                    self.line = self.ax1.axhline(y=event.ydata, color="green")
                    self.linetxt = self.ax2.text(0, event.ydata-self.txtoffset, round(event.ydata,5))
                    event.canvas.draw()

        def onMouseClick(event):
            # LEFT=1, MIDDLE=2, RIGHT=3, BACK=8, FORWARD=9
            if event.inaxes == self.ax1 and event.button == 3:
                if 0.0 < event.ydata < 1.0 and self.mzs[0] < event.xdata < self.mzs[-1]:
                    self.peakdots.remove()
                    self.deisodots.remove()
                    self.pick(event.ydata)
                    self.peakdots = self.ax1.scatter(self.picked_mzs, self.picked_intens, s=50, c="red", zorder=2)
                    self.deisodots = self.ax1.scatter([], [], s=10, c="red", zorder=3)
                    self.peaktxt.set_text("Number of Peaks: %i"%(self.peak_nb))
                    self.deisotxt.set_text("Number of Deiso:  %i"%(self.deiso_nb))
                    event.canvas.draw()

        self.fig.canvas.mpl_connect("motion_notify_event", onMouseMove)
        self.fig.canvas.mpl_connect("button_press_event", onMouseClick)


    def plot(self):
        self.create_baseplot()
        plt.show()


    def pick(self, t):
        self.Picker = Easypicker(self.dframe, self.aggregation, self.winsorize)
        self.Picker.find_peaks(t)
        self.picked_mzs = self.Picker.peaks_mzs
        self.picked_intens = self.Picker.aggr_spec[self.Picker.peaks_idx]
        self.peak_nb = len(self.picked_mzs)
        self.deiso_nb = 0

    def deiso(self, event):
        if self.Picker:
            self.Picker.deisotope((self.deiso_range_min, self.deiso_range_max))
            self.deiso_mzs = self.Picker.deiso_peaks_mzs
            self.deiso_intens = self.Picker.aggr_spec[self.Picker.deiso_peaks_idx]
            self.deiso_nb = len(self.deiso_mzs)
            self.deisotxt.set_text("Number of Deiso:  %i"%(self.deiso_nb))
            self.deisodots = self.ax1.scatter(self.deiso_mzs, self.deiso_intens, s=15, c="orange", zorder=3)

    def set_deiso_range_min(self, text):
        try:
            self.deiso_range_min = float(text)
        except:
            self.minnamefield.set_val(str(self.deiso_range_min))
            print("Type number in deisorange!")
    
    def set_deiso_range_max(self, text):
        try:
            self.deiso_range_max = float(text)
        except:
            self.maxnamefield.set_val(str(self.deiso_range_max))
            print("Type number in deisorange!")

    
    def run(self, event):
        if self.peak_nb > 0 and self.deiso_nb > 0:
            plt.close("all")
            return self.Picker
        else:
            print("Select Peaks and Deisotope them!")
            warntxt = self.ax1.text(0.8, 0.5, "Select Peaks and Deisotope them!", fontsize=24)
            event.canvas.draw()
            time.sleep(2)
            warntxt.remove()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to _processed_simplified.h5 file.")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
    parser.add_argument("-w", "--winsorize", type=int, required=False, default=5, help="Maximum peak intensity. The w'th highest peak will be used as upper limit. Default is 5.")
    parser.add_argument("-n", "--normalize", type=str2bool, required=True, default=True, help="Whether to normalize the Spectrum into [0,1] or not. Default is True.")
    args = parser.parse_args()
    
    filepath = args.filepath
    savepath = args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    name = os.path.basename(filepath).split(".")[0]
    h5_file = pd.read_hdf(filepath)
    winsorize = args.winsorize
    normalize = args.normalize

    plot = InteractivePeakPickingThresholder(h5_file, winsorize, normalize, name, savepath)
    plot.plot()
    
    picked_dframe = plot.Picker.create_dframe(deisotoped=True)
    picked_dframe.to_hdf(savepath, key=os.path.basename(savepath).split(".")[0], complib="blosc", complevel=9)
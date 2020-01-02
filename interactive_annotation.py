import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib import path
from os.path import join, basename
import os
import argparse
import matplotlib.image as mpimg

#Testaufruf:
#python /homes/mfeser/gf-projekt/ims-sideprojects/processing/interactive_annotation.py -f BI_180718_Glio_5_converted/BI_180718_Glio_5_processed_simplified.h5 -e BI_180718_Glio_5_converted/BI_180718_Glio_5_processed_simplified_embedding.npy

class InteractiveDataAnnotation:
    def __init__(self, dframe, embedding, lightfields, name, savepath):
        self.name = name
        self.savepath = savepath
        self.dframe = dframe
        self.result = self.create_dframe()
        self.gx = np.array(dframe.index.get_level_values("grid_x").astype(int))
        self.gy = np.array(dframe.index.get_level_values("grid_y").astype(int))
        #changed the orientation of the coords matrix for coherence
        self.coords = np.array(list(zip(self.gy, self.gx)))
        self.embedding = embedding
        self.lightfields = lightfields
        self.ctr = 0
        self.lightfield = mpimg.imread(self.lightfields[self.ctr])
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        for c in self.coords:
            self.img[c[0]-1][c[1]-1]+=1
        self.ax1_xmin = np.amin(self.embedding[:,0]) - 1
        self.ax1_ymin = np.amin(self.embedding[:,1]) - 1
        self.ax1_xmax = np.amax(self.embedding[:,0]) + 1
        self.ax1_ymax = np.amax(self.embedding[:,1]) + 1
        self.toggle = "reset"
        self.toggle_props = {
            "select": {'color': 'red', 'linewidth': 2, 'alpha': 0.8},
            "reset": {'color': 'black', 'linewidth': 2, 'alpha': 0.8},
            "examine": {'color': 'red', 'linewidth': 2, 'alpha': 0.8}
            }
        self.idx = {
            "select": [],
            "reset": [],
            "examine": [],
            }
        self.exportname = "next-time-try-to-name-your-data"
        with open(join(self.savepath,self.name[:-21]+"_annotated.csv"),'w') as fd:
            fd.write("x\ty\ttissue_name\n")
        fd.close()

    def create_baseplot(self):
        self.fig = plt.figure()
        self.ax1 = plt.axes([0.12, 0.5, 0.2, 0.375])
        self.ax1.set_title("UMAP Embedding")
        self.ax1.set_xlabel("Dimension 1")
        self.ax1.set_ylabel("Dimension 2")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")


        self.ax2 = plt.axes([0.55, 0.2, 0.4, 0.75])
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img, vmax=2)
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        plt.subplots_adjust(bottom=0.1)

        self.ax3 = plt.axes([0.68,0.13,0.12,0.065])
        self.exportnamefield = TextBox(self.ax3,"", initial="")
        self.exportnamefield.on_submit(self.set_exportname)
        self.ax3.text(0,1.15,"Tissue Name")
        
        self.ax4 = plt.axes([0.55, 0.05, 0.12, 0.065])
        self.selectbutton = Button(self.ax4, "Select")
        self.selectbutton.on_clicked(self.select)

        self.ax6 = plt.axes([0.68, 0.05, 0.12, 0.065])
        self.annotatebutton = Button(self.ax6, "Annotate")
        self.annotatebutton.on_clicked(self.annotate)

        self.ax5 = plt.axes([0.81, 0.05, 0.12, 0.065])
        self.resetbutton = Button(self.ax5, "Reset")
        self.resetbutton.on_clicked(self.reset)  

        self.ax9 = plt.axes([0.81, 0.13, 0.12, 0.065])
        self.exportbutton = Button(self.ax9, "Export")
        self.exportbutton.on_clicked(self.export)  

        self.ax6 = plt.axes([0.12, 0, 0.2, 0.375])
        self.ax6.axis("off")
        self.ax6.set_title(self.lightfields[self.ctr])
        self.ax6.imshow(mpimg.imread(self.lightfields[self.ctr]), vmax=1)

        self.ax7 = plt.axes([0.33, 0.12, 0.12, 0.065])
        self.nextbutton = Button(self.ax7, "Next")
        self.nextbutton.on_clicked(self.next)
        
        self.ax8 = plt.axes([0.33, 0.05, 0.12, 0.065])
        self.prevbutton = Button(self.ax8, "Previous")
        self.prevbutton.on_clicked(self.prev)

        self.ax10 = plt.axes([0.55, 0.13, 0.12, 0.065])
        self.examinebutton = Button(self.ax10, "Examine")
        self.examinebutton.on_clicked(self.examine)

    def calc_img(self):
        self.img[:] = 0
        for c in self.coords:
            self.img[c[0]-1][c[1]-1]+=1
        self.img[(self.gy[self.idx[self.toggle]], self.gx[self.idx[self.toggle]])] += 1
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.imshow(self.img, vmax=2)
        #self.ax11.imshow(self.staticimg, vmax=1, alpha=0.5)


    def onselect(self, verts):
        p = path.Path(verts)
        self.idx[self.toggle] = p.contains_points(self.pts.get_offsets())
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.ax1.scatter(self.embedding[self.idx["reset"], 0], self.embedding[self.idx["reset"], 1], s=6, c=self.toggle_props["reset"]["color"])
        self.ax1.scatter(self.embedding[self.idx["select"], 0], self.embedding[self.idx["select"], 1], s=6, c=self.toggle_props["select"]["color"])
        self.calc_img()
        self.fig.canvas.draw_idle()

    def onselect2(self, verts):
        new_verts = []
        for v in verts:
            a = list(v)
            a.reverse()
            c=tuple(a)
            new_verts.append(c)
        p = path.Path(new_verts)
        self.idx[self.toggle] = p.contains_points(self.coords)
        self.img[:] = 0
        self.calc_img()
        self.ax1.scatter(self.embedding[self.idx["examine"], 0], self.embedding[self.idx["examine"], 1], s=6, c=self.toggle_props["examine"]["color"])
        self.fig.canvas.draw_idle()

    def plot(self):
        self.create_baseplot()
        plt.show()
   
    def examine(self, event):
        for c in self.coords:
            self.img[c[0]-1][c[1]-1]=1
        self.ax2.imshow(self.img, vmax=2)
        self.toggle = "examine"
        self.lsso = None
        self.lsso2 = LassoSelector(ax=self.ax2, onselect=self.onselect2, lineprops=self.toggle_props[self.toggle])
        print("Examine")
    
    def prev(self, event):
        self.ctr-=1
        self.ctr%=len(self.lightfields)
        self.ax6.set_title(self.lightfields[self.ctr])
        self.ax6.imshow(mpimg.imread(self.lightfields[self.ctr]), vmax=1)
        print("loading previous lightfield picture")

    def next(self, event):
        self.ctr+=1
        self.ctr%=len(self.lightfields)
        self.ax6.set_title(self.lightfields[self.ctr])
        self.ax6.imshow(mpimg.imread(self.lightfields[self.ctr]), vmax=1)
        print("loading next lightfield picture")

    def select(self, event):
        print("select")
        self.toggle = "select"
        self.lsso2 = None
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.idx["reset"] = []

    def reset(self, event):
        print("reset")
        self.toggle = "reset"
        self.lsso2 = None
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        for c in self.coords:
            self.img[c[0]-1][c[1]-1]=1
        self.ax2.imshow(self.img, vmax=2)
        self.idx["reset"] = []
        self.idx["select"] = []
        self.idx["examine"] = []
    
    def set_exportname(self, text):
        self.exportname = text

    def set_filename(self, text):
        self.filename = text

    def create_dframe(self):
        h5groups = self.dframe
        mzs = h5groups.columns
        return pd.DataFrame(columns=mzs, index=pd.MultiIndex(levels=[[],[],[]],
                             codes=[[],[],[]],
                             names=['grid_x', 'grid_y', 'dataset']))

    def export(self, event):
        print(self.result)
        self.result.to_hdf(join(self.savepath, self.name + "_" + self.exportname + ".h5"), key=self.name, complib="blosc", complevel=9)
        print("export")

    def annotate(self, event):
        if self.toggle!="examine":
            indices = self.dframe.index
            l = indices.tolist()
            ctr=0
            for elem in l:
                x = int(elem[0])
                y = int(elem[1])
                if self.img[y][x]==1:
                    row = self.dframe.xs(elem)
                    row.name = elem
                    self.result = self.result.append(row)
                    ctr+=1   
            print(ctr,"selected pixels were annotated")
            self.toggle = "reset"
            self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
            self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
            self.ax2.imshow(self.img, vmax=1)
            self.pts.remove()
            self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        else: 
            print("Annotation is disabled during examination")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--h5file", type=str, required=True, help="Path to _processed_simplified.h5 File")
    parser.add_argument("-e", "--embedding", type=str, required=True, help="Path to _embedding.npy File")
    parser.add_argument("-l", "--lightfield", type=str, required=False, default=".", help="Path to folder with .tif")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=".", help="Path to save Output")
    args = parser.parse_args()
    
    h5_file = pd.read_hdf(args.h5file)
    embedding = np.load(args.embedding)
    lightfield=args.lightfield
    name = basename(args.h5file).split(".")[0]
    savepath = args.savepath

    lightfields = []
    for file in os.listdir(lightfield):
        if file.endswith(".tif"):
            lightfields.append(join(lightfield,file))
    lightfields.reverse()

    print(" ")
    print(name)
    plot = InteractiveDataAnnotation(h5_file, embedding, lightfields, name, savepath)
    plot.plot()

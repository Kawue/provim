import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
from sys import argv
import pandas as pd
import argparse
from msi_utils import read_h5_files

class MsiImageWriter:
    def __init__(self, dframe, savepath, scaling="single", cmap=plt.cm.viridis, colorscale_boundary=(0,100)):
        self.dframe = dframe
        self.savepath = savepath

        if self.savepath is not None:
            if not os.path.isdir(self.savepath):
                os.makedirs(self.savepath)

        try:
            if len(set(self.dframe.index.get_level_values("dataset"))) > 1:
                raise ValueError("You provided a merged data set. Please provide a single data set.")
        except:
            print("No 'dataset' index encoded. A single data set is assumed.")

        self.grid_x = np.array(self.dframe.index.get_level_values("grid_x")).astype(int)
        self.grid_y = np.array(self.dframe.index.get_level_values("grid_y")).astype(int)
        self.height = self.grid_y.max() + 1
        self.width = self.grid_x.max() + 1

        self.colormap = plt.cm.ScalarMappable(plt.Normalize(), cmap=cmap)

        self.colorscale_boundary = colorscale_boundary

        if scaling == "all":
            self.colormap.set_clim(np.percentile(self.dframe, self.colorscale_boundary))
            self.scaling = scaling
        elif scaling == "single":
            self.scaling = scaling
            #self.dframe.apply(lambda x: (x - min(x))
        else:
            raise Exception('Scaling has to be "all" or "single"')


    def write_msi_imgs(self, fname):
        if not os.path.exists(os.path.join(self.savepath, fname+"-images")):
            os.makedirs(os.path.join(self.savepath, fname+"-images"))
        for mz, intens in self.dframe.iteritems():
            if self.scaling == "single":
                self.colormap.set_clim(np.percentile(intens, self.colorscale_boundary))
            img = self._create_empty_img(True)
            img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(np.array(intens))
            plt.imsave(os.path.join(self.savepath, fname+"-images", str(np.round(mz, 3)) + ".png"), img)

    
    def write_msi_clusters(self, labels):
        for label in set(labels):
            clusterpath = os.path.join(self.savepath, "C%i"%label)
            if self.savepath is not None:
                if not os.path.isdir(clusterpath):
                    os.makedirs(clusterpath)
            for mz, intens in self.dframe.iloc[:, np.where(labels==label)[0]].iteritems():
                if self.scaling == "single":
                    self.colormap.set_clim(np.percentile(intens, self.colorscale_boundary))
                img = self._create_empty_img(True)
                img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(np.array(intens))
                plt.imsave(os.path.join(clusterpath, str(np.round(mz, 3)) + ".png"), img)


    def write_dimvis_rgb(self, red_ch, green_ch, blue_ch, method_name):
        # Use matplotlibs reverse gray colormap to scale intensities in colorscale boundary
        tmp_cm = plt.cm.ScalarMappable(plt.Normalize(), cmap=plt.cm.Greys_r)
        tmp_cm.set_clim(np.percentile(red_ch, self.colorscale_boundary))
        r_intens = tmp_cm.to_rgba(red_ch)[:, 0]
        tmp_cm.set_clim(np.percentile(green_ch, self.colorscale_boundary))
        g_intens = self.colormap.to_rgba(green_ch)[:, 0]
        tmp_cm.set_clim(np.percentile(blue_ch, self.colorscale_boundary))
        b_intens = self.colormap.to_rgba(blue_ch)[:, 0]
        
        rgb_img = np.zeros((self.height, self.width, 3))
        rgb_img[(self.grid_y, self.grid_x, 0)] = r_intens
        rgb_img[(self.grid_y, self.grid_x, 1)] = g_intens
        rgb_img[(self.grid_y, self.grid_x, 2)] = b_intens

        self.colormap.set_clim(np.percentile(red_ch, self.colorscale_boundary))
        r_img = np.zeros((self.height, self.width))
        r_img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(red_ch)[:, 0]
        
        self.colormap.set_clim(np.percentile(green_ch, self.colorscale_boundary))
        g_img = np.zeros((self.height, self.width))
        g_img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(green_ch)[:, 0]
        
        self.colormap.set_clim(np.percentile(blue_ch, self.colorscale_boundary))
        b_img = np.zeros((self.height, self.width))
        b_img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(blue_ch)[:, 0]

        plt.imsave(os.path.join(self.savepath, method_name + "RGB.png"), rgb_img)
        plt.imsave(os.path.join(self.savepath, method_name + "R.png"), r_img)
        plt.imsave(os.path.join(self.savepath, method_name + "G.png"), g_img)
        plt.imsave(os.path.join(self.savepath, method_name + "B.png"), b_img)


    def write_dimvis_components(self, dimreduce_transform, n_components, method_name):
        for idx, intens in enumerate(dimreduce_transform[:, 0:n_components].T, start=1):
            img = self._create_empty_img(True)
            self.colormap.set_clim(np.percentile(intens, self.colorscale_boundary))
            img[(self.grid_y, self.grid_x)] = self.colormap.to_rgba(intens)
            plt.imsave(os.path.join(self.savepath, method_name + "_component_" + str(idx) + ".png"), img)


    def write_dimvis(self, dimreduce_transform, n_components, rgb_indices, method_name):
        if len(rgb_indices) != 3:
            raise ValueError("Tuple of list of size three is needed to produce an RGB image from three components.")
        self.write_dimvis_rgb(
            dimreduce_transform[:, rgb_indices[0]],
            dimreduce_transform[:, rgb_indices[1]],
            dimreduce_transform[:, rgb_indices[2]],
            method_name)
        self.write_dimvis_components(dimreduce_transform, n_components, method_name)


    def image_pruner(self):
        img = self._create_empty_img(False)
        img[(self.grid_y, self.grid_x)] = 1
        #plt.imshow(img)
        #plt.show()
        lbl = label(img)
        props = regionprops(lbl)
        if len(props) != 1:
            print("More than one measured region was found. Consider to apply matrix_remover first!")
        #self.sample_box = props[np.argmax([prop.area for prop in props])].bbox
        #sample_grid_x = self.dframe.index.get_level_values["grid_x"].isin(range(self.sample_box[1], self.sample_box[3]))
        #sample_grid_y = self.dframe.index.get_level_values["grid_y"].isin(range(self.sample_box[0], self.sample_box[2]))
        #self.dframe = self.dframe[sample_grid_x * sample_grid_y]
        self.dframe.rename(lambda n: (n-min(self.grid_x)).astype(int), level="grid_x", inplace=True)
        self.dframe.rename(lambda n: (n-min(self.grid_y)).astype(int), level="grid_y", inplace=True)
        self.grid_x = self.grid_x - min(self.grid_x)
        self.grid_y = self.grid_y - min(self.grid_y)
        self.height = self.grid_y.max() + 1
        self.width = self.grid_x.max() + 1


    def matrix_remover(self):
        img = self._create_empty_img(False)
        img[(self.grid_y, self.grid_x)] = 1
        lbl = label(img)
        props = regionprops(lbl)
        #plt.imshow(img)
        #plt.show()
        if len(props) > 2:
            print("More than two regions were found. The algorithm proceeds with the largest one.")
        if len(props) < 2:
            print("Only one region were found. Either, matrix is not separated from pixel and cannot be removed or no matrix was measured.")
        
        max_area_idx = np.argmax([prop.area for prop in props])
        self.sample_box = props[max_area_idx].bbox
        #matrix_boxes = []
        #for idx, prop in enumerate(props):
        #    if idx == max_area_idx:
        #        self.sample_box = props[idx].bbox
        #    else:
        #        matrix_boxes.append(props[idx].bbox)
        sample_grid_x = self.dframe.index.get_level_values("grid_x").isin(range(self.sample_box[1], self.sample_box[3]))
        sample_grid_y = self.dframe.index.get_level_values("grid_y").isin(range(self.sample_box[0], self.sample_box[2]))
        #matrix_grid_x = np.array([self.dframe.index.get_level_values("grid_x").isin(range(box[1], box[3])) for box in matrix_boxes]).prod(axis=0).astype(bool)
        #matrix_grid_y = np.array([self.dframe.index.get_level_values("grid_y").isin(range(box[0], box[2])) for box in matrix_boxes]).prod(axis=0).astype(bool)
        #self.matrix_field = self.dframe[matrix_grid_x * matrix_grid_y]
        self.dframe = self.dframe[sample_grid_x * sample_grid_y]
        self.grid_x = self.grid_x[sample_grid_x * sample_grid_y].astype(int)
        self.grid_y = self.grid_y[sample_grid_x * sample_grid_y].astype(int)
        #self.height = self.grid_y.max() + 1
        #self.width = self.grid_x.max() + 1

        #self.dframe.rename(lambda n: n-min(self.grid_x), level="grid_x")
        #self.dframe.rename(lambda n: n-min(self.grid_y), level="grid_y")


    def _create_empty_img(self, rgba):
        if rgba:
            return np.zeros((self.height + 1, self.width + 1, 4))
        else:
            return np.zeros((self.height + 1, self.width + 1))



    # DataFrame adjustemnt funcitons to be on par with the new images
    def write_dframe(self, savepath):
        print(os.path.basename(savepath).split(".")[0])
        self.dframe.to_hdf(savepath, key=os.path.basename(savepath).split(".")[0], complib="blosc", complevel=9)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--readpath", type=str, required=True, nargs='+', help="Path to h5 files.")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=False, help="Path to save output.")
    parser.add_argument("--remove_matrix", required=False, action='store_true', help="Removes matrix fields. Works only if they are spatially separated from the main sample.")
    parser.add_argument("--clip", required=False, action='store_true', help="Adjusts pixel positions in the h5 file to remove as many zero areas as possible, i.e. offsets in all directions will be removed.")
    parser.add_argument("--write_mz", required=False, action='store_true', help="Save pngs of all m/z values within the h5 file.")
    parser.add_argument("--write_hdf", required=False, action='store_true', help="Save the processed h5 file, i.e. save the changes due to --remove_matrix and --clip.")
    args=parser.parse_args()
    
    readpath = args.readpath
    savepath = args.savepath

    if not (args.remove_matrix or args.clip or args.write_mz or args.write_hdf):
        raise ValueError("Flag at least one operation to proceed.")

    h5_files, fnames, paths = read_h5_files(args.readpath)

    def set_savepath(path, idx, paths=paths):
        if path:
            savepath = path
        else:
            savepath = paths[idx]
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        return savepath

    for idx, h5_file in enumerate(h5_files):
        writer = MsiImageWriter(h5_file, set_savepath(savepath, idx))
        
        if args.remove_matrix:
            writer.matrix_remover()
            print("ATTENTION: If there are spatially separated sample areas, they will be removed along with the matrix fields! Only the largest connected measurement area will remain!")
        
        if args.clip:
            writer.image_pruner()
            if not args.write_hdf:
                print()
                print("ATTENTION: --clip is selected but without --write-hdf clipping will not be saved in a file!")
                print()
        
        if args.write_mz:
            writer.write_msi_imgs(fnames[idx])
            
        if args.write_hdf:
            if args.clip:
                dframe_savepath = os.path.join(set_savepath(savepath, idx), "{0}_pruned{1}".format(fnames[idx], ".h5"))
            else:
                raise ValueError("Use --write_hdf only in combination with --clip.")
            writer.write_dframe(dframe_savepath)

import numpy as np
import pandas as pd
from sys import argv
from os.path import join, dirname
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from msi_dimension_reducer import UMAP, LSA
from msi_utils import read_h5_files, str2bool
from msi_image_writer import MsiImageWriter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to _processed_simplified.h5 file.")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
    parser.add_argument("--subtract_artifacts", type=bool, required=False, help="Boolean whether the artifacts mean spectrum should be subtracted.", default=False)
    parser.add_argument("--delete_matrix", type=bool, required=False, help="Boolean whether the matrix pixels should be deleted.", default=False)
    parser.add_argument("--delete_artifacts", type=bool, required=False, help="Boolean whether the artifact pixels should be deleted.", default=False)
    parser.add_argument("--quality_control", type=bool, required=False, help="Saves some plot for quality control purposes.", default=True)
  
    args = parser.parse_args()
    
    filepath = args.filepath
    savepath = args.savepath
    artifacts = args.subtract_artifacts
    delete_matrix = args.delete_matrix
    delete_artifacts = args.delete_artifacts

    h5_files, fnames = read_h5_files(filepath)

    quality_control = args.quality_control
    
    for idx, h5_file in enumerate(h5_files):                            
        matrixframe = pd.read_hdf(join(dirname(filepath), fnames[idx] + "_matrix" + ".h5"))
        artifactsframe = pd.read_hdf(join(dirname(filepath), fnames[idx] + "_artifacts" + ".h5"))
        dataframe = h5_file
        
        if len(matrixframe) == 0:
            raise ValueError("Matrix data is empty!")
            
        dataframe[:] = dataframe.values - np.array(matrixframe.mean(axis=0))

        if artifacts:
            if len(artifactsframe.values) == 0:
                print("Artifacts data is empty. The parameter will be ignored!")
            else:
                print(artifactsframe)
                dataframe[:] = dataframe.values - np.array(artifactsframe.mean(axis=0))

        dataframe[dataframe<0] = 0

        if delete_matrix:
            dataframe = dataframe.drop(matrixframe.index)

        if delete_artifacts:
            dataframe = dataframe.drop(artifactsframe.index)


        if quality_control:
            subpath = join(savepath, fnames[idx])
            if not os.path.isdir(subpath):
                os.makedirs(subpath)

            winsorize = 5
            normalize = True
            mean_spec = dataframe.mean(axis=0)
            if winsorize > 0:
                winsorize_limit = sorted(mean_spec)[-winsorize]
                mean_spec[mean_spec > winsorize_limit] = winsorize_limit
            if normalize:
                mi = mean_spec.min()
                ma = mean_spec.max()
                mean_spec = (mean_spec - mi) / (ma - mi)

            plt.figure(figsize=(16,9))
            plt.title("No Matrix Spectrum " + fnames[idx] + " (winsorized with %i and normalized in [0,1])"%(winsorize))
            plt.plot(dataframe.columns, mean_spec)
            plt.savefig(join(subpath, "No Matrix Spectrum " + fnames[idx] + ".png"), bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(16,9))
            plt.title("Only Matrix Spectrum " + fnames[idx])
            plt.plot(matrixframe.columns, matrixframe.mean(axis=0))
            plt.savefig(join(subpath, "only_matrix_spectrum_" + fnames[idx] + ".png"), bbox_inches='tight')
            plt.close()
            if artifacts:
                plt.figure(figsize=(16,9))
                plt.title("Only Artifacts Spectrum " + fnames[idx])
                plt.plot(artifactsframe.columns, artifactsframe.mean(axis=0))
                plt.savefig(join(subpath, "only_artifacts_spectrum_" + fnames[idx] + ".png"), bbox_inches='tight')
                plt.close()

            ImgWriter = MsiImageWriter(h5_file, None)
            img = ImgWriter._create_empty_img(False)
            sgx = dataframe.index.get_level_values("grid_x").astype(int)
            sgy = dataframe.index.get_level_values("grid_y").astype(int)
            mgx = matrixframe.index.get_level_values("grid_x").astype(int)
            mgy = matrixframe.index.get_level_values("grid_y").astype(int)
            agx = artifactsframe.index.get_level_values("grid_x").astype(int)
            agy = artifactsframe.index.get_level_values("grid_y").astype(int)
            img[(sgy, sgx)] = 1
            img[(mgy, mgx)] = 2
            img[(agy, agx)] = 3
            plt.title("Matrix, Sample, Artifact Regions of  + fnames[idx]")
            plt.imsave(join(subpath, "matrix_sample_artifact_regions_of_" + fnames[idx] + ".png"), img) 
            #ImgWriterMatrix = MsiImageWriter(matrixframe, join(subpath, "matrix_imgs"))
            #ImgWriterMatrix.write_msi_imgs()
            plt.close()

        
        dirpath = os.path.dirname(filepath)
        dataframe.to_hdf(join(dirpath, fnames[idx] + "_matrixremoved.h5"), key=fnames[idx]+"_matrixremoved", complib="blosc", complevel=9)    
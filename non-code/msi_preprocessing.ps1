#!/bin/bash
#Don't add any spaces around the '='. Only change parameters with a comment '#' behind 
#This script was not tested and is just a simple modification of msi_preprocessing.sh for Windows-Users, use at own risk

$datafolder=/vol/ims/Wuellems/Manuel/imzML/BI_180718_Glio_5/ #Folder the data (.imzML and .ibd) is located
$nameHDF5=BI_180718_Glio_5 #Name of pyBASIS HDF5 Format (no .h5 needed)
$multidata=false #true if multiple data sets should be processed, otherwise false
echo "Starting workflow_pybasis.py..."
$tmp="$nameHDF5.h5"
python /homes/mfeser/gf-projekt/pyBASIS_workflow/pyBASIS/workflow_pybasis.py -d $datafolder -n $tmp -m $multidata

echo "Finished workflow_pybasis.py.\nStarting fast_convert_hdf.py..."
#output data will be saved at the location of the .imzML and .ibd
#console output is saved at the location of the .imzML and .ibd
$massRangeStart=0 #changable
$massRangeEnd=100 #changable
python /homes/mfeser/gf-projekt/ims-sideprojects/processing/fast_convert_hdf.py -f $datafolder$nameHDF5"_processed_pybasis.h5" -s $datafolder --mass_range_start $massRangeStart --mass_range_end $massRangeEnd > $datafolder"fast_convert__hdf_statistics.txt"

echo "Finished fast_convert_hdf.py.\nStarting matrix_preprocessing.py..."
python /homes/mfeser/gf-projekt/ims-sideprojects/processing/matrix_preprocessing.py -f $datafolder$nameHDF5"_processed_simplified.h5" -s $datafolder

echo "Finished matrix_preprocessing.py.\nStarting interactive_matrix_detection.py..."
python /homes/mfeser/gf-projekt/ims-sideprojects/processing/interactive_matrix_detection.py -f $datafolder$nameHDF5"_processed_simplified.h5" -e $datafolder$nameHDF5"_processed_simplified_embedding.npy" -s $datafolder

echo "Finished interactive_matrix_detection.py.\nStarting matrix_postprocessing.py..."
$subtractArtifacts=true #true if artifacts should be subracted, false otherwise
$deleteMatrix=false #true if matrix should be deleted, false otherwise
$deleteArtifacts=false #true if matrix should be deleted, false otherwise
python /homes/mfeser/gf-projekt/ims-sideprojects/processing/matrix_postprocessing.py -f $datafolder$nameHDF5"_processed_simplified.h5" -s $datafolder --subtract_artifacts $subtractArtifacts --delete_matrix $deleteMatrix --delete_artifacts $deleteArtifacts

echo "Finished matrix_postprocessing.py.\nStarting workflow_peakpicking..."
$winsorize=5 #number of highest peaks
$unionAllData=false #true to pick on union of all data sets, false otherwise
$transformation="log" #transformation function (log, sqrt, false)
$interactive=true #interactive tool for threshold adjustment
$rangeDeisotoping=0 #range for deisotoping, if 0 then off
$thresholdPicking=0.5 #threshold for picking (value between 0 and 1)
python /homes/mfeser/gf-projekt/ims-sideprojects/processing/workflow_peakpicking.py -f $datafolder -s $datafolder -w $winsorize -u $unionAllData -t $transformation -i $interactive -d $rangeDeisotoping -p $tresholdPicking
echo "Finished workflow_peakpicking.\n\n Finished Workflow."

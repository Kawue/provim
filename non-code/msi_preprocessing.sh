#!/bin/bash
#Don't add any spaces around the '='. Only change parameters with a comment '#' behind 
installationspath=/homes/mfeser/gf-projekt

datafolder=/vol/ims/Wuellems/Manuel/imzML/BI_180718_Glio_5/ #Folder the data (.imzML and .ibd) is located
dataname=BI_180718_Glio_5 #Name of pyBASIS HDF5 Format (no .h5 needed)
multidata=True #true if multiple data sets should be processed, otherwise false
echo "Starting workflow_pybasis.py..."
python $installationspath"/"pyBASIS_workflow/pyBASIS/workflow_pybasis.py -d $datafolder -n $dataname".h5" -m $multidata

echo "Finished workflow_pybasis.py.\nStarting fast_convert_hdf.py..."
#output data will be saved at the location of the .imzML and .ibd
#console output is saved at the location of the .imzML and .ibd
massRangeStart=48 #changable
massRangeEnd=101 #changable
python $installationspath"/"ims-sideprojects/processing/fast_convert_hdf.py -f $datafolder$dataname"_processed_pybasis.h5" -s $datafolder --mass_range_start $massRangeStart --mass_range_end $massRangeEnd > $datafolder"fast_convert_hdf_statistics.txt"

echo "Finished fast_convert_hdf.py.\nStarting matrix_preprocessing.py..."
workdir=$dataname"_converted/"
dataproc=$dataname"_processed_simplified"
python $installationspath"/"ims-sideprojects/processing/matrix_preprocessing.py -f $datafolder$workdir$dataproc".h5" -s $datafolder$workdir

echo "Finished matrix_preprocessing.py.\nStarting interactive_matrix_detection.py..."
python $installationspath"/"ims-sideprojects/processing/interactive_matrix_detection.py -f $datafolder$workdir$dataproc".h5" -e $datafolder$workdir$dataproc"_embedding.npy" -s $datafolder$workdir

echo "Finished interactive_matrix_detection.py.\nStarting matrix_postprocessing.py..."
subtractArtifacts=true #true if artifacts should be subracted, false otherwise
deleteMatrix=false #true if matrix should be deleted, false otherwise
deleteArtifacts=false #true if matrix should be deleted, false otherwise
python $installationspath"/"ims-sideprojects/processing/matrix_postprocessing.py -f $datafolder$workdir$dataproc"-h5" -s $datafolder$workdir --subtract_artifacts $subtractArtifacts --delete_matrix $deleteMatrix --delete_artifacts $deleteArtifacts

echo "Finished matrix_postprocessing.py.\nStarting workflow_peakpicking..."
winsorize=5 #number of highest peaks
unionAllData=false #true to pick on union of all data sets, false otherwise
transformation="log" #transformation function (log, sqrt, false)
interactive=true #interactive tool for threshold adjustment
rangeDeisotoping=0 #range for deisotoping, if 0 then off
thresholdPicking=0.5 #threshold for picking (value between 0 and 1)
data=$datafolder$nameHDF5"_converted/"$dataname"_matrixremoved.h5"
python $installationspath"/"ims-sideprojects/processing/workflow_peakpicking.py -f $datafolder$workdir$dataproc"_matrixremoved.h5" -s $datafolder -w $winsorize -u $unionAllData -t $transformation -i $interactive -d $rangeDeisotoping -p $tresholdPicking
echo "Finished workflow_peakpicking.\n\n Finished Workflow."

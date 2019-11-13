# -*- coding: utf-8 -*-
"""

***********************************************
Preprocessing workflow object management module
***********************************************

The modules includes class for creation and management of pre-processing 
workflow objects to ensure that all passed user parameters are sound.
 
"""


import os
from basis.utils.cmdline import Option, Value, AllValues;

ImportSet_options=[Option('-t,--filetype', help = 'Input File Format.', values=[\
                        Value('HDI', help = 'HDI File Format', parameters=[\
                            Option('--fileext', help = 'Set input file extension.', values=['.txt', None], type=str, 
                                   targets=['/filereadinfo']),\
                            Option('--mzline', help = 'Set input file mz-line value.', values=[3, None], type=int, 
                                   conditions=[('mzline>=0', lambda x: x.mzline>=0)], targets=['/filereadinfo']),\
                            Option('--delimiter', help = 'Set input file delimiter.', values=['\t', None], type=str, 
                                   targets=['/filereadinfo']),\
                            ]),\
                        Value('imzML', help = 'imzML File Format', parameters=[\
                            Option('--fileext', help = 'Set input file extension.', values=['.imzML', None], type=str, 
                                   targets=['/filereadinfo']),\
                            Option('--mzunits', help = 'M/Z units', values=[\
                                Value('Da', help = 'Dalton units', parameters=[
                                    Option('--mzmaxshift', help = 'Maximum allowed m/z peak drift within sample.', values=[0.01, None], 
                                           type=float, conditions=[('mzmaxshift>=0.0', lambda x: x.mzmaxshift>=0.0)], 
                                           targets=['/filereadinfo']),\
                                    Option('--cmzbinsize', help = 'Histogram bin size for reference m/z feature vector calculations.', 
                                           values=[0.001, None], 
                                           type=float, conditions=[('cmzbinsize>=0.0', lambda x: x.cmzbinsize>=0.0)], 
                                           targets=['/filereadinfo']),\
                                    ]),\
                                Value('ppm', help = 'ppm units', parameters=[
                                    Option('--mzmaxshift', help = 'Maximum allowed m/z peak drift within sample.', values=[50.0, None],
                                           type=float, conditions=[('mzmaxshift>=0.0', lambda x: x.mzmaxshift>=0.0)], 
                                           targets=['/filereadinfo']),\
                                    Option('--cmzbinsize', help = 'Histogram bin size for reference m/z feature vector calculations.', 
                                           values=[5, None], 
                                           type=float, conditions=[('cmzbinsize>=0.0', lambda x: x.cmzbinsize>=0.0)], 
                                           targets=['/filereadinfo']),\
                                    ]),\
                                ], type=str, targets=['/filereadinfo']),\
                            ]),\
                        ], type=str),\
                   Option('datapath', help = 'Sets data directory path.', values=[os.getcwd(), None], type=str, 
                                conditions=[('Path must exist!', lambda x: os.path.isdir(x.datapath))]),\
                   Option('h5rawdbname', help = 'Output HDF5 file name.', values=['', None], type=str)\
                   ];


ExportSet_options=[Option('-t,--filetype', help = 'Export File Format.', values=[\
                        Value('HDI', help = 'HDI File Format', parameters=[\
                            Option('--fileext', help = 'Set export file extension.', 
                                   values=['.txt', None], type=str, targets=['/params']),\
                            Option('--exportpath', help = 'Set the export path (if not specified, exports data files into' +\
                                                                                '``datapath/basis_hdi_timestamp folder``).', 
                                   values=['', None], type=str,targets=['/params']),\
                            ]),\
                        Value('BASIS_matlab', help = 'BASIS (matlab) h5-based format', parameters=[\
                            Option('--israw', help = 'Select data type for export (0: raw data, 1: processed data)', 
                                   values=[0,1], type=str,targets=['/params'])\
                            ])
                        ], type=str),\
                   
                   Option('h5dbname', help = 'Input HDF5 file name(s).', is_list=False, 
                          values=['', None], type=str, optional=False)
                   ];


PeakAlign_options=[\

                   Option('--method', help = 'The type of a peak alignment method.', values=[\
                        Value('NN', help = 'Nearest Neighbour', parameters=[\
                            Option('--mzunits', help = 'M/Z units', values=[\
                                Value('Da', help = 'Dalton units', parameters=[
                                    Option('--mzmaxshift', help = 'Maximum allowed m/z peak drift.', values=[0.1, None], 
                                           type=float, conditions=[('mzmaxshift>=0.0', lambda x: x.mzmaxshift>=0.0)], 
                                           targets=['/params']),\
                                    Option('--cmzbinsize', help = 'Histogram bin size for reference m/z feature vector calculations.', 
                                           values=[0.01, None], 
                                           type=float, conditions=[('cmzbinsize>=0.0', lambda x: x.cmzbinsize>=0.0)], 
                                           targets=['/params']),\
                                    ]),
                                Value('ppm', help = 'ppm units', parameters=[
                                    Option('--mzmaxshift', help = 'Maximum allowed m/z peak drift.', values=[100.0, None],
                                           type=float, conditions=[('mzmaxshift>=0.0', lambda x: x.mzmaxshift>=0.0)], 
                                           targets=['/params']),\
                                    Option('--cmzbinsize', help = 'Histogram bin size for reference m/z feature vector calculations.', 
                                           values=[10, None], 
                                           type=float, conditions=[('cmzbinsize>=0.0', lambda x: x.cmzbinsize>=0.0)], 
                                           targets=['/params']),\
                                    ]),
                                ], type=str, targets=['/params']),\
                            ])
                        ], type=str),\
                                    
                 Option('--lockmz', help = 'a vector of external lock masses', values = ['', None], type = str,
                        targets=['/params']),         

                 Option('h5rawdbname', help =  'HDF5-based database file with multiple deposited msi datasets '+\
                                            'for peak alignment. Each MSI dataset assumes to contain peak picked '+\
                                            'MSI data with their spatial coordinates and mz (and optionally drift time) '+\
                                            'feature vectors.', \
                                            values=['', None], type=str, optional=False),
  
                 Option('h5dbname', help = 'Path to a HDF5-based msi database for storage and organization of pre-processed data. '+\
                                               'if this db file exists, all pre-processing parameters will be extracted from it, '+\
                                               'to make sure that the newly imported data are compatible with the ones stored in '+\
                                               'the processed db. The pre-processing workflow can be customized for newly '+\
                                               'created db instance.',\
                                               values=['', None], type=str)
                 
                   ] #end of peak alignment module
                        
IntraNorm_options=[\
                   Option('--method', help = 'Intranormalization method.', values=[\
                        Value('mfc', help = 'Median fold change', parameters=[\
                            Option('--reference', help = 'Refence dataset with respect to which the fold intensity changes of other datasets are calculated.', \
                                values=['mean'], type=str, targets=['/params']),\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['yes', 'no'], type=str, targets=['/params']),\
                            ]),\

                        Value('mean', help = 'Mean', parameters=[\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['yes', 'no'], type=str, targets=['/params']),\
                            ]),\
                        
                        Value('median', help = 'Median', parameters=[\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['yes', 'no'], type=str, targets=['/params']),\
                            ])\
                        ], type=str),\
                   Option('--min_mz', help = 'Lower limit for M/Z values.', values=[0.0, None], type=float, conditions=[('min_mz>=0.0', lambda x: x.min_mz>=0.0)]),\
                   Option('--max_mz', help = 'Upper limit for M/Z values.', values=[50000.0, None],  type=float, conditions=[('max_mz>=0.0', lambda x: x.max_mz>=0.0)]),\
                   
                   Option('h5dbname', help = 'Input HDF5 file name(s).', is_list=False, values=['', None], type=str, optional=False)                   
                   ]; #end of value IntraNorm


InterNorm_options=[\
                   Option('--method', help = 'Inter-normalization method.', values=[\
                        Value('mfc', help = 'Median fold change', parameters=[\
                            Option('--reference', help = 'Refence dataset with respect to which the fold intensity changes of other datasets are calculated.', \
                                values=['mean'], type=str, targets=['/params']),\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0.0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['no', 'yes'], type=str, targets=['/params']),\
                            ]),\

                        Value('mean', help = 'Mean', parameters=[\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0.0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['no', 'yes'], type=str, targets=['/params']),\
                            ]),\
                        
                        Value('median', help = 'Median', parameters=[\
                            Option('--offset', help = 'Disregard peak intensity smaller that this value.', \
                                values=[0.0, None], type=float, conditions=[('offset>=0.0', lambda x: x.offset>=0.0)], targets=['/params']),\
                            Option('--outliers', help = 'Outliers.', \
                                values=['no', 'yes'], type=str, targets=['/params']),\
                            ])\
                        ], type=str),\
                   Option('--min_mz', help = 'Lower limit for M/Z values.', values=[0.0, None], type=float, conditions=[('min_mz>=0.0', lambda x: x.min_mz>=0.0)]),\
                   Option('--max_mz', help = 'Upper limit for M/Z values.', values=[50000.0, None],  type=float, conditions=[('max_mz>=0.0', lambda x: x.max_mz>=0.0)]),\
                   
                   Option('h5dbname', help = 'Input HDF5 file name(s).', is_list=False, values=['', None], type=str, optional=False)
                   ]; #end of value InterNorm


VST_options=[\
                   Option('-m,--method', help = 'Variance stabilizing transformation.', values=[\
                        Value('started-log', help = 'StartedLog', parameters=[\
                            Option('--offset', help = 'Offset.', values=[50.0, None], type=float, targets=['/params']),\
                            ])\
                        ], type=str),\
                   Option('h5dbname', help = 'Input HDF5 file name(s).', is_list=False, values=['', None], type=str, optional=False)
                   ]; 
                   #end of value VST


PeakFilter_options=[\
                   Option('-m,--method', help = 'Solvent/Matrix Peak Removal.', values=[\
                        Value('kmeans', help = 'k-means cluster-driven solvent filter', parameters=[\
                            Option('--nclusters', help = 'Number of clusters.', \
                                values=[2, None], type=int, targets=['/params']),\
                           # Option('--metric', help = 'Metric for clustering.', \
                           #     values=['correlation'], type=str, targets=['/params']),\
                            ])\
                        ], type=str),\
                   Option('h5dbname', help = 'Input HDF5 file name(s).', is_list=False, values=['', None], type=str, optional=False)
                   ]; 
                   #end of peak filtering options
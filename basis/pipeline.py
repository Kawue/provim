from basis.io import importmsi, exportmsi
from basis.preproc import palign, intranorm, vst, pfilter

datapath = '/Users/Dieter/Desktop/Data'
h5rawdbname = '/Users/Dieter/Desktop/Data/rawdata.h5'
h5dbname = '/Users/Dieter/Desktop/Data/procdata.h5'

importmsi.do_import(datapath, h5dbname, filetype='imzML',
                    params={'fileext': '.txt','cmzbinsize': 5, 'mzmaxshift':50, 'mzunits': 'ppm', 'mzline': 3, 'delimiter':'\t'})

palign.do_alignment(h5rawdbname, h5dbname=h5dbname, method='NN',
                    params={'mzmaxshift': 100, 'cmzbinsize': 10, 'mzunits':'ppm', 'lockmz': ''})

intranorm.do_normalize(h5dbname=h5dbname, method='mfc',
                       params={'reference':'median'}, mzrange=[0, 2000],  istrain=1)


vst.do_vst(h5dbname=h5dbname, method='started-log', params={'offset': 10})


pfilter.do_filtering(h5dbname=h5dbname, method='kmeans', params={'nclusters': 2})

exportmsi.do_export(h5dbname, 'HDI', params={'fileext': '.txt'})

exportmsi.do_export(h5dbname, 'BASIS_matlab', params={})


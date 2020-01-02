import numpy as np
import pandas as pd
import argparse
import statistics
from os.path import join, basename

class StatisticalEvaluation:
    def __init__(self, dframe, method, name, savepath):
        self.dframe = dframe
        self.name = name
        self.savepath = savepath
        self.method = method
        self.result = self.create_dframe()
        self.datasets = self.getDatasets()
        self.datas = self.extractData()
        self.evaluate()
        self.result.to_hdf(join(self.savepath, self.name + "_" + self.method + "_evaluated" + ".h5"), key=self.name, complib="blosc", complevel=9)

    def create_dframe(self):
        h5groups = self.dframe
        mzs = h5groups.columns
        return pd.DataFrame(columns=mzs, index=pd.MultiIndex(levels=[[]],
                             codes=[[]],
                             names=['dataset']))
        
    def getDatasets(self):
        datasets = []
        for x in self.dframe.index:
            d = list(x)[2]
            if d not in datasets:
                datasets.append(d)
        return datasets

    def extractData(self):
        datas = {}
        for d in self.datasets:
            datas.update({d : self.dframe.xs(d, level=2)})
        return datas
        
    def evaluate(self):
        for d in self.datasets:
            dct = {}
            for c in self.dframe.columns:
                if self.method=="mean":
                    dct.update({c:statistics.mean(self.datas[d][c].tolist())})
                elif self.method=="median":    
                    dct.update({c:statistics.median(self.datas[d][c].tolist())})
                else:
                    raise ValueError("No valid method. Use mean or median")
            row = pd.Series(dct, name=d)
            self.result = self.result.append(row)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--h5file", type=str, required=True ,help="Path to _processed_simplified.h5 File")
    parser.add_argument("-s", "--savepath", type=str, required=False, default=".", help="Path to save Output")
    parser.add_argument("-m", "--method", type=str, required=False, default="mean", help="Method for statistical evaluation")
    args = parser.parse_args()
    
    h5_file = pd.read_hdf(args.h5file)
    method = args.method
    name = basename(args.h5file).split(".")[0]
    savepath = args.savepath
    stats = StatisticalEvaluation(h5_file, method, name, savepath)
    

import numpy as np
import pandas as pd
import argparse

class Reader:
    def __init__(self, dframe):
        self.dframe = dframe
        print(self.dframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--h5file", type=str, required=True ,help="Path to _processed_simplified.h5 File")
    args = parser.parse_args()
    
    h5_file = pd.read_hdf(args.h5file)
    reader = Reader(h5_file)
    

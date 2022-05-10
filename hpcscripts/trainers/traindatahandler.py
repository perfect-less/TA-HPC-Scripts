"""This Module Compile Training Data in to separate csv for training
and test"""

import shutil
from math import ceil, floor

from os import listdir
from os.path import isfile, join

import pandas as pd 
import numpy as np

from hpcscripts.sharedutils.fileprocessing import GetFilesName
from hpcscripts.option import pathhandler as ph

def copy_datafile(read_dir, to_dir, flight_file):
    """Copy flight_file from read_dir to to_dir"""
    origin = join(read_dir, flight_file)
    target = join(to_dir,   flight_file)

    shutil.copyfile(origin, target)

def run ():
    print ("------Start Creating Data Set------")

    read_dir       = ph.GetProcessedPath("Selected")
    write_dir      = ph.GetProcessedPath("Combined")
    train_file_dir = ph.GetProcessedPath("Train")
    test_file_dir  = ph.GetProcessedPath("Test")
    eval_file_dir  = ph.GetProcessedPath("Eval")

    flight_files = GetFilesName(read_dir, False)

    ind = np.arange(len(flight_files))
    np.random.shuffle(ind)

    print ("There are {} available files".format (len(flight_files)))

    train_count = ceil (0.7 * len(flight_files))
    test_count  = floor((len(flight_files) - train_count)/2) 
    eval_count = len(flight_files) - train_count - test_count

    print ("..{} of it will be in training set".format (train_count))
    print ("..{} of it will be in test set".format (test_count))
    print ("..and {} of it will be in eval set".format (eval_count))

    flightDFs = list()
    for i in range(len(flight_files)):
        file = pd.read_csv( join(read_dir, flight_files[ind[i]]) )

        hd = False
        mo = 'a'
        if i == 0 or i == train_count or i == train_count+test_count:
            hd = True
            mo = 'w'
            
        # Write Into One CSV. Also copy to each folder
        if i < train_count:
            file.to_csv(join(write_dir, "Train_set.csv"), mode=mo, header=hd, index=False)
            copy_datafile(
                            read_dir, train_file_dir,
                            flight_files[ind[i]]
                        )

        elif i < (train_count + test_count):
            file.to_csv(join(write_dir, "Test_set.csv"), mode=mo, header=hd, index=False)
            copy_datafile(
                            read_dir, test_file_dir,
                            flight_files[ind[i]]
                        )

        else:
            file.to_csv(join(write_dir, "Eval_set.csv"), mode=mo, header=hd, index=False)
            copy_datafile(
                            read_dir, eval_file_dir,
                            flight_files[ind[i]]
                        )

        
    print ("Train, test, and eval set created")
    print ("---------------------------------------------")


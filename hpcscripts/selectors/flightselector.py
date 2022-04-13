"""This module responsible for selecting usable data from bundles of 
Approach data in 'Approach' folders. All usable data will then copied
to 'Selected' folder."""

import os
import shutil
import random

from typing import List
from math import pi

import pandas as pd
import numpy as np

from hpcscripts.sharedutils.fileprocessing import GetFilesName
from hpcscripts.option import pathhandler as ph


def run ():

    read_dir  = ph.GetProcessedPath("Approach")
    write_dir = ph.GetProcessedPath("Selected")
    flight_files = GetFilesName(read_dir)

    unusable_indexes = []

    # Read Files to List of DataFrames
    flight_DFs = ReadFiles(flight_files, read_dir)

    # Gathering indexes of files with empty data
    unusable_indexes = GatherEmptyElevatorIndexes(flight_DFs, unusable_indexes)

    # Gathering indexes of files with out of range data
    unusable_indexes = GatherUnusableBasedOnGSAndLocalizer(flight_DFs, unusable_indexes)

    # Copy usable file left
    CopyUsableFlightsData(read_dir, write_dir, unusable_indexes, flight_files)
    


def ReadFiles(flight_files: list, read_dir: str):
    flight_DFs = list()
    for i in range(len(flight_files)):
        flight_DFs.append( pd.read_csv(os.path.join(read_dir, flight_files[i])) )
    return flight_DFs

def CopyUsableFlightsData(read_dir: str, write_dir: str, unsuable_indexes: list, flight_files: list):
    print ("..begin copying")
    for i in range(len(flight_files)):

        if i in unsuable_indexes:
            continue

        origin = os.path.join(read_dir,  flight_files[i])
        target = os.path.join(write_dir, flight_files[i])

        shutil.copyfile(origin, target)
    print ("..finished copying usable files")

def GatherEmptyElevatorIndexes(flight_DFs: List[pd.DataFrame], unusable_indexes: List[int]):
    empty_count = 0

    for i, flight_DF in enumerate(flight_DFs):
        flight_DF['elv_l_rad'].replace('', np.nan, inplace=True)

        if (abs(flight_DF["elv_l_rad"].diff().std(axis=0)) <= 0.0008):
            unusable_indexes.append(i)
            empty_count += 1

        if flight_DF.shape[0] == 0:
            unusable_indexes.append(i)
            empty_count += 1
        
        if flight_DF.shape[0] < 200: # temp
            unusable_indexes.append(i)
            empty_count += 1
            
        if flight_DF.shape[0] > 0 and np.isnan(flight_DF.loc[random.randint(0, flight_DF.shape[0]-1) , "elv_l_rad"]):
            unusable_indexes.append(i)
            empty_count += 1
    
    print ("empty count: {}".format(empty_count))
    return unusable_indexes

def GatherUnusableBasedOnGSAndLocalizer(flight_DFs: List[pd.DataFrame], unusable_indexes: List[int]):
    deg2rad = pi / 180
    unusable_count = 0

    for i, flight_DF in enumerate(flight_DFs):
        if i in unusable_indexes:
            continue

        flight_DF['loc_dev_ddm'].replace('', np.nan, inplace=True)
        flight_DF['lat_rad'].replace('', np.nan, inplace=True)

        if (flight_DF.lat_rad.max() > 35.047973 * deg2rad):
            unusable_indexes.append(i)
            unusable_count += 1

        if (flight_DF.loc[flight_DF.shape[0]-1, "lat_rad"] < 35.012 * deg2rad):
            unusable_indexes.append(i)
            unusable_count += 1

    print ("unusable based on glideslope and localizer: {}".format(unusable_count))
    return unusable_indexes

        

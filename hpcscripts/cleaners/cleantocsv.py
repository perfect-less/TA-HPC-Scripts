import pandas as pd 
import numpy as np
import time
import math

from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
from math import sin, cos, asin, sqrt, pi, ceil,floor

from hpcscripts.sharedutils.fileprocessing import GetFilesName
from hpcscripts.cleaners.utils import datacleaning as dc
from hpcscripts.option import pathhandler


def log_result(retval):
    # results.append(retval)
    # if len(results) % max(data_count//50, 1) == 0 and data_count > 0:
    #     print("\r", "...{:.0%} done".format(len(results)/data_count), end= "")

    # print ("..callback called")
    pass


def CleaningWork(func, p_num, filepath, lname = "Sampling"):

    print("Starting {} Process".format(lname))

    start_time = time.time()

    flight_files = GetFilesName(filepath)

    pool = Pool(p_num)
    results = []
    for item in flight_files:
        pool.apply_async(func, args=[item], callback=log_result)
        
    print ("..start making pool")
    pool.close()
    pool.join()

    end_time = time.time()
    work_time = end_time - start_time

    print ("[V]{} process took ".format(lname) + str(work_time) + " seconds to complete")
    print ("--------------------")

    return []



def run(pool_size = 16):

    results = []

    raw_path = pathhandler.GetRawDataPath()
    sam_path = join(pathhandler.GetProcessedDataPath(), "Sampled")
    cln_path = join(pathhandler.GetProcessedDataPath(), "Cleaned")
    app_path = join(pathhandler.GetProcessedDataPath(), "Approach")

    pathhandler.ClearProcessedDir("Sampled")
    pathhandler.ClearProcessedDir("Cleaned")
    pathhandler.ClearProcessedDir("Approach")

    flight_files = GetFilesName(raw_path, False)
    data_count = len(flight_files)

    # Mark start time
    startTime = time.time()
    print ("Begin data cleaning process")
    print ("start at {}".format(startTime))


    # Start with reworking the sampling
    results = CleaningWork(dc.ReSampling, pool_size, raw_path, "ReSampling")

    # And then Processing our data
    results = CleaningWork(dc.CleanAndCompleteData, pool_size, sam_path, "CleanAndCut")

    #mark the end time
    endTime = time.time()

    #calculate the total time it took to complete the work
    workTime =  endTime - startTime

    #print results
    print ("[***]Data cleaning took {} Minutes and {:.0f} Seconds to complete".format(floor(workTime/60), workTime % 60))
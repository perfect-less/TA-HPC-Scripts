import pandas as pd 
import numpy as np
import time
import math
import DataProcessCollection as dpc

from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
from math import sin, cos, asin, sqrt, pi, ceil,floor

def log_result(retval):
	results.append(retval)
	if len(results) % max(data_count//50, 1) == 0 and data_count > 0:
		print("\r", "...{:.0%} done".format(len(results)/data_count), end= "") # end= '\r'


def GetFilesName(mypath, printIt = True):
	#mypath = "Flights_35/"

	flight_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	av_flights_num = len(flight_files)

	if printIt == True:
    	print(".Number of Available Flights: {}".format(av_flights_num))

	return flight_files

def DoTheWork(func, p_num, filepath, lname = "Sampling"):

	print("[*]============Starting {} Process".format(lname))

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
	print ("---------------------------------------------")

	return []



if __name__ == '__main__':

	results = []

	D_Id = "_cut_UC"

	raw_path = "Flights/"
	sam_path = "Processed Data{}/Sampled/".format(D_Id)
	cln_path = "Processed Data{}/Cleaned/".format(D_Id)
	app_path = "Processed Data{}/Approach/".format(D_Id)

	flight_files = GetFilesName(raw_path, False)
	data_count = len(flight_files)


	# Mark start time
	startTime = time.time()
	print ("=============================================")
	print ("Start at {}".format(startTime))


	# Map Our Function to Pool Processes

	#with Pool(8) as p:
	#    p.apply_async(dpc.ReSampling, args= flight_files, callback=log_result)

	# Start with reworking the sampling
	results = DoTheWork(dpc.ReSampling, 4, raw_path, "ReSampling")

	# And then Processing our data
	results = DoTheWork(dpc.CleanAndCompleteData, 4, sam_path, "CleanAndCut")

	#mark the end time
	endTime = time.time()

	#calculate the total time it took to complete the work
	workTime =  endTime - startTime

	#print results
	#print ("[***]The job took " + str(workTime) + " seconds to complete")
	print ("[***]The job took {} Minutes and {:.0f} Seconds to complete".format(floor(workTime/60), workTime % 60))
import pandas as pd
import numpy as np
import math

from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS

from hpcscripts.cleaners.utils.resamplefunctions import *
from hpcscripts.cleaners.utils.cleaningfunctions import *


def ReSampling(fname):
    print("..start sampling {}".format(fname))

    # Resampling Parameters
    selected_columns = G_PARAMS.SELECTED_COLUMNS
    target_frequency = G_PARAMS.TARGET_FREQUENCY
    cutOut_distance  = G_PARAMS.CUTOUT_DISTANCE
    
    read_dir  = ph.GetRawDataPath()
    write_dir = ph.GetProcessedPath("Sampled")
    
    flight_DF = pd.read_csv("{readdir}/{fname}".format(readdir= read_dir, fname= fname))

    # Creating new hdot
    CalculateSmoothVerticalSpeed(flight_DF)
    
    # Determining CutOff point
    # Walk back and cut when the distance to runway is above 10 Nautical Mile
    touchdown_index, trace_index = TraceCutOffPoint(flight_DF, cutOut_distance)

    fparam_np = np.empty([flight_DF.shape[0],1])
    fparam_np[0,0] = flight_DF.loc[trace_index, "time_s"]
    fparam_np[1,0] = flight_DF.loc[touchdown_index+1, "time_s"] - 50 #15 # minus 15 seconds

    # ReIndexing the data
    flight_DF, touchdown_index = ReIndexFlightData(flight_DF, target_frequency, touchdown_index, selected_columns)

    # Adding 'Distance to landing' column
    flight_DF = CalculateAdditionalColumn(flight_DF, touchdown_index, fparam_np)

    # Save data
    flight_DF.to_csv("{writedir}/{fname}.csv".format(writedir= write_dir, fname= fname.removesuffix(".zip")), index=False)
    print("..done sampling {}".format(fname))



def CleanAndCompleteData(fname):        
    print("..start Cleaning {}".format(fname))
    
    # Cleaning Params
    read_dir = ph.GetProcessedPath("Sampled") 
    write_dir = ph.GetProcessedPath("Cleaned") 
    approach_dir = ph.GetProcessedPath("Approach") 

    target_gamma = G_PARAMS.TARGET_GAMMA
            
    flight_DF = pd.read_csv("{readdir}/{fname}".format(readdir= read_dir, fname= fname))
    

    # Process each column that needed to be processed
    flight_DF = CleanColumns(flight_DF)
    

    # Calculate glide slope and glide slope error
    hdotd_np = flight_DF["hdot_2_mps"].to_numpy()
    gs_np = flight_DF["gs_mps"].to_numpy()

    gamma_np = np.arctan2(hdotd_np, gs_np)

    flight_DF = pd.concat( [flight_DF, pd.DataFrame(gamma_np, columns=["gamma_rad"])], axis=1)
    flight_DF["gamma_error_rad"] = target_gamma - flight_DF["gamma_rad"]

    # Calculate delta between theta and theta_trim
    flight_DF["theta_del_rad"] = flight_DF["theta_rad"] - flight_DF["theta_trim_rad"]

    # Calculate average N1 and flap position    
    flight_DF["N1s_rpm"] = (flight_DF["n11_rpm"] + flight_DF["n12_rpm"] + flight_DF["n13_rpm"] + flight_DF["n14_rpm"]) / 4
    flight_DF["flap_te_pos"] = flight_DF["flap_te_pos"] / 100
    

    # Save to  CSV for documentation
    flight_DF.to_csv("{writedir}/{fname}".format(writedir= write_dir, fname= fname), index=False)
    

    # Cut Into Only Approach Phase    
    minT = flight_DF.loc[0, "fParam"]
    maxT = flight_DF.loc[1, "fParam"]

    ind = flight_DF.loc[(flight_DF["time_s"] >= minT) & (flight_DF["time_s"] <= maxT)].index
    flight_DF = flight_DF.loc[ind,:].copy()
    
    # Remove fParam
    flight_DF.drop("fParam", axis = 1, inplace= True)
    

    # Save another new CSV
    flight_DF.to_csv("{writedir}/{fname}".format(writedir= approach_dir, fname= fname), index=False)
    
    print("..done Cleaning {}".format(fname))
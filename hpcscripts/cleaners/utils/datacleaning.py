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

    sampling_freq = 1/flight_DF.loc[1, "time_s"]
    divider = int (sampling_freq / target_frequency)
    new_length = int (flight_DF.shape[0]/divider)

    fparam_np = np.empty([new_length,1])
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
    

    # Process (deleting outliars and filtering) each column that needed to be processed
    flight_DF = CleanColumns(flight_DF)
    

    # Calculate 'glide' slope and 'glide slope error'
    hdotd_np = flight_DF["hdot_2_mps"].to_numpy()
    gs_np = flight_DF["gs_mps"].to_numpy()

    gamma_np = np.arctan2(hdotd_np, gs_np)

    flight_DF = pd.concat( [flight_DF, pd.DataFrame(gamma_np, columns=["gamma_rad"])], axis=1)
    flight_DF["gamma_error_rad"] = target_gamma - flight_DF["gamma_rad"]

    # Calculate delta between 'theta' and 'theta_trim'
    flight_DF["theta_del_rad"] = flight_DF["theta_rad"] - flight_DF["theta_trim_rad"]

    # Calculate average 'N1', total fqty, and 'flap position'    fqty_1_kg fqty_2_kg fqty_3_kg fqty_4_kg
    flight_DF["N1s_rpm"] = (flight_DF["n11_rpm"] + flight_DF["n12_rpm"] + flight_DF["n13_rpm"] + flight_DF["n14_rpm"]) / 4
    flight_DF["fqty_kg"] = flight_DF["fqty_1_kg"] + flight_DF["fqty_2_kg"] + flight_DF["fqty_3_kg"] + flight_DF["fqty_4_kg"]
    flight_DF["flap_te_pos"] = flight_DF["flap_te_pos"] / 100

    # Calculate pitch-rate and alpha-rate
    flight_DF["theta_rate_rps"] = flight_DF["theta_rad"].diff().fillna(0, inplace=False)
    flight_DF["aoac_rate_rps"] = flight_DF["aoac_rad"].diff().fillna(0, inplace=False)

    # Calculate tail and crosswind
    flight_DF["tailwind_mps"] = flight_DF["ws_mps"] * np.cos(flight_DF["wdir_rad"] - flight_DF["psi_rad"])
    flight_DF["crosswind_mps"] = flight_DF["ws_mps"] * np.sin(flight_DF["wdir_rad"] - flight_DF["psi_rad"])

    # Flap Config, only 4: 25, 5: 30, and 6: 36 
    flight_DF["flap_4_bool"] = flight_DF["flap_te_pos"].apply(lambda x: 1 if x >= 20 and x < 27 else 0)
    flight_DF["flap_5_bool"] = flight_DF["flap_te_pos"].apply(lambda x: 1 if x >= 27 and x < 32 else 0)
    flight_DF["flap_6_bool"] = flight_DF["flap_te_pos"].apply(lambda x: 1 if x >= 32 else 0)


    # Calculate error derivative
    flight_DF["g_err_d_rad"] = flight_DF["gamma_error_rad"].diff().fillna(0, inplace=False)
    

    # Save to  CSV for documentation
    flight_DF.to_csv("{writedir}/{fname}".format(writedir= write_dir, fname= fname), index=False)
    

    # Cut Into Only Approach Phase    
    minT = flight_DF.loc[0, "fParam"]
    maxT = flight_DF.loc[1, "fParam"]

    ind = flight_DF.loc[(flight_DF["time_s"] >= minT) & (flight_DF["time_s"] <= maxT)].index
    flight_DF = flight_DF.loc[ind,:].copy()

    # Calcualte error integral
    flight_DF["g_err_i_rad"]   = flight_DF["gamma_error_rad"].cumsum().interpolate()
    flight_DF["g_err_ii_rad"]  = flight_DF["g_err_i_rad"].cumsum().interpolate()
    flight_DF["g_err_iii_rad"] = flight_DF["g_err_ii_rad"].cumsum().interpolate()
    
    # Remove fParam
    flight_DF.drop("fParam", axis = 1, inplace= True)
    

    # Save another new CSV
    flight_DF.to_csv("{writedir}/{fname}".format(writedir= approach_dir, fname= fname), index=False)
    
    print("..done Cleaning {}".format(fname))
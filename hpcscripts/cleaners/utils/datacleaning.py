import pandas as pd
import numpy as np
import math

from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS

from hpcscripts.cleaners.utils.resamplefunctions import *
from hpcscripts.cleaners.utils.filters import *
D_Id = ""

def ReSampling(fname):

    print("..start Sample {}".format(fname))

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
    print("..done Sample {}".format(fname))

    
def CleanAndCompleteData(fname):
    #column_list = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
    #            "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
    #            "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "gs_dev_ddm", "loc_dev_ddm", "gamma_acc_mps2", "lon_rad", "lat_rad", 
    #            "ws_mps", "wshear_warn", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2",
    #            "very_clean_hdot_2_mps", "dist_to_land"]
    
    #column_list = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
    #            "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
    #            "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "gs_dev_ddm", "loc_dev_ddm", "lon_rad", "lat_rad", "phi_rad", 
    #            "ws_mps", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2", "chi_rad", "psi_rad", "psi_selected_rad",
    #            "very_clean_hdot_2_mps", "dist_to_land"]
    
    column_list = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
                "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
                "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "pla_1_rad", "pla_2_rad", "pla_3_rad", "pla_4_rad", "ctrlcolumn_pos_capt", "ctrlcolumn_pos_fo", "ctrlcolumn_pos_fo", "ctrlwheel_pos_fo",
                "gs_dev_ddm", "loc_dev_ddm", "lon_rad", "lat_rad", "phi_rad", "gmt_hr", "gmt_min", "gmt_s", "drift_rad", "temp_static_deg", "temp_total_degC",
                "ws_mps", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2", "chi_rad", "psi_rad", "psi_selected_rad",
                "very_clean_hdot_2_mps", "delta_very_clean_hdot_2_mps", "dist_to_land"]
    
    filter_list = ["hbaro_m", "hralt_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "elv_l_rad", "elv_r_rad", "flap_te_pos", "hdot_2_mps", "gs_dev_ddm", "dist_to_land"]
    
    filter_list = [] # Not doing low pass filter

    #column_list = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
    #            "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
    #            "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "gs_dev_ddm", "gamma_acc_mps2", "very_clean_hdot_2_mps",                         "dist_to_land"]
    
    rdir = "Processed Data{}/Sampled".format(D_Id)
    wdir = "Processed Data{}/Cleaned".format(D_Id)
    fdir = "Processed Data{}/Approach".format(D_Id)
    edir = "Processed Data{}/Early".format(D_Id)
    
    print("..start Cleaning {}".format(fname))
    
    new_DF = pd.read_csv("{readdir}/{fname}".format(readdir= rdir, fname= fname))
    
    for column_name in column_list:

        #delete_outliars = 1
        #filtering_steps = 0

        movingAverageWin = 0
        #movingAverageWin = 50

        if True:

            if column_name == "time_s" or column_name == "flap_te_pos" or column_name == "cas_mps" or column_name == "theta_rad" or column_name == "aoac_rad":
                movingAverageWin = 0

            # Deleting Outliars
                
            if column_name == "elv_l_rad":
                # Just Delete all the out of range elevator param.
                # new_DF.loc[ (new_DF[column_name]).abs() > 30 * math.pi / 180 , column_name ] = None
                new_DF.loc[ (new_DF[column_name]).abs() > 15 * math.pi / 180 , column_name ] = None
                
            if column_name == "elv_r_rad":
                # Just Delete all the out of range elevator param.
                new_DF.loc[ (new_DF[column_name]) < 50 * math.pi / 180 , column_name ] = None
                
            new_DF[column_name] = new_DF[column_name].interpolate()

            # Butter Filtering
            if column_name in filter_list:
                new_DF[column_name] = pd.Series( butter_lowpass_filter(new_DF[column_name].to_numpy(), cutoff, fs, order) )                    

            if movingAverageWin > 0:
                new_DF[column_name] = new_DF[column_name].rolling(window=movingAverageWin).mean()
    
        #new_DF[column_name] = new_DF[column_name].interpolate()

    ## Calculate the Rest of variables
    target_gamma = -3 * math.pi / 180
    
    hdotd_np = new_DF["hdot_2_mps"].to_numpy()
    gs_np = new_DF["gs_mps"].to_numpy()

    gamma_np = np.arctan2(hdotd_np, gs_np)

    new_DF = pd.concat( [new_DF, pd.DataFrame(gamma_np, columns=["gamma_rad"])], axis=1)

    new_DF["gamma_error_rad"] = target_gamma - new_DF["gamma_rad"]
    new_DF["N1s_rpm"] = (new_DF["n11_rpm"] + new_DF["n12_rpm"] + new_DF["n13_rpm"] + new_DF["n14_rpm"]) / 4

    new_DF["flap_te_pos"] = new_DF["flap_te_pos"] / 100
    
    # Save new CSV
    new_DF.to_csv("{writedir}/{fname}".format(writedir= wdir, fname= fname), index=False)
    
    
    
    ## Cut Into Only Approach Phase
    
    minT = new_DF.loc[0, "fParam"]
    maxT = new_DF.loc[1, "fParam"]

    ind = new_DF.loc[(new_DF["time_s"] >= minT) & (new_DF["time_s"] <= maxT)].index
    e_ind = new_DF.loc[new_DF["time_s"] <= 300].index

    ap_DF = new_DF.loc[ind,:].copy()
    e_DF = new_DF.loc[e_ind,:]
    
    
    # Remove fParam
    ap_DF.drop("fParam", axis = 1, inplace= True)
    e_DF.drop("fParam", axis = 1, inplace= True)
    
    # Save another new CSV
    ap_DF.to_csv("{writedir}/{fname}".format(writedir= fdir, fname= fname), index=False)
    e_DF.to_csv("{writedir}/{fname}".format(writedir= edir, fname= fname), index=False)
    
    print("..done Sample {}".format(fname))
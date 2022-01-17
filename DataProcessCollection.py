import pandas as pd
import numpy as np
import math

from math import sin, cos, asin, sqrt, pi, ceil, floor

D_Id = "_cut_UC"

## Butter low pass filter
from scipy.signal import butter,filtfilt# Filter requirements.
fs = 1.0       # sample rate, Hz
cutoff = 0.06 #0.055 #0.07     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 2 Hz
order = 8       
def butter_lowpass_filter(data, cutoff, fs, order):
    
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y

def haversine_dist(lat1, lat2, lon1, lon2):
    r = 6371 # km
    
    h = sin( (lat2 - lat1)/2 )**2 + cos(lat1)*cos(lat2)*( sin( (lon2 - lon1)/2 )**2 )
    # print (h)
    return 2*r*math.atan2( math.sqrt( h ), math.sqrt( 1-h ) )

def ReSampling(fname):
    selected_columns = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
                "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
                "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "pla_1_rad", "pla_2_rad", "pla_3_rad", "pla_4_rad", "ctrlcolumn_pos_capt", "ctrlcolumn_pos_fo", "ctrlcolumn_pos_fo", "ctrlwheel_pos_fo",
                "gs_dev_ddm", "loc_dev_ddm", "lon_rad", "lat_rad", "phi_rad", "gmt_hr", "gmt_min", "gmt_s", "drift_rad", "temp_static_deg", "temp_total_degC",
                "ws_mps", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2", "chi_rad", "psi_rad", "psi_selected_rad"]
    
    target_frequency = 1 # Hz    
    cutOut_dist = 15.39  #10.5 #18.52 # Km, ~ 10 Nm
    
    rdir = "Flights"
    wdir = "Processed Data{}/Sampled".format(D_Id)
    
    print("..start Sample {}".format(fname))
    
    f_dat = pd.read_csv("{readdir}/{fname}".format(readdir= rdir, fname= fname))

    sampling_freq = 1/f_dat["time_s"][1]
    divider = int (sampling_freq / target_frequency)

    # ReIndexing the data
    i_list = [a*divider for a in range(int(f_dat.shape[0]/divider))]

    n_data = pd.DataFrame( f_dat.loc[i_list, selected_columns] )
    n_data = n_data.reset_index(drop=True)

    # Creating new hdot
    n_data["very_clean_hdot_2_mps"] = n_data["hdot_2_mps"] * 1
    n_data["very_clean_hdot_2_mps"] = n_data["very_clean_hdot_2_mps"].interpolate()
    n_data["very_clean_hdot_2_mps"] = n_data["very_clean_hdot_2_mps"].rolling(window=60).mean()

    n_data["delta_very_clean_hdot_2_mps"] = n_data["very_clean_hdot_2_mps"].shift(periods= -5) - n_data["hdot_2_mps"]
    
    # Determining CutOff point
    Tmax = f_dat.loc[ f_dat.shape[0] - 1, "time_s" ]
    ind = f_dat.loc[ (f_dat["time_s"] > Tmax/2) & (f_dat["wow"] < 0.5) ].index 

    # Walk back and cut when the distance to runway is above 10 Nautical Mile

    dist = 0
    ii = ind[0]

    f_dat["lat_rad"] = f_dat["lat_rad"].interpolate()
    f_dat["lon_rad"] = f_dat["lon_rad"].interpolate()
    f_dat["loc_dev_ddm"] = f_dat["loc_dev_ddm"].interpolate()
    f_dat["hralt_m"] = f_dat["hralt_m"].interpolate()

    # and abs(f_dat.loc[ii, "loc_dev_ddm"]) <= 1.5 * math.pi / 180 
    while (dist < cutOut_dist and ii > 0 and abs(f_dat.loc[ii, "loc_dev_ddm"]) <= 1.5 * math.pi / 180  and not (n_data.loc[floor(ii/divider), "hralt_m"] > 400 and n_data.loc[floor(ii/divider), "very_clean_hdot_2_mps"] > -0.4)):
        ii -= 1
        dist = haversine_dist(f_dat.loc[ii, "lat_rad"], f_dat.loc[ind[0], "lat_rad"], f_dat.loc[ii, "lon_rad"], f_dat.loc[ind[0], "lon_rad"])

    d2l_np = f_dat["wow"].to_numpy()
    for ik in range(n_data.shape[0]):
        d2l_np[ik] = haversine_dist(n_data.loc[ik, "lat_rad"], f_dat.loc[ind[0], "lat_rad"], n_data.loc[ik, "lon_rad"], f_dat.loc[ind[0], "lon_rad"])

    fparam_np = np.empty([n_data.shape[0],1])
    fparam_np[0,0] = f_dat.loc[ii, "time_s"]
    fparam_np[1,0] = f_dat.loc[ind[0]+1, "time_s"] - 50 #15 # minus 15 seconds
    
    d2l_np = d2l_np.reshape((d2l_np.shape[0], 1))
    n_data = pd.concat( [n_data, pd.DataFrame(d2l_np, columns=["dist_to_land"]), pd.DataFrame(fparam_np, columns=["fParam"])], axis=1 )
    n_data.to_csv("{writedir}/{fname}.csv".format(writedir= wdir, fname= fname.removesuffix(".zip")), index=False)
    
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
                new_DF.loc[ (new_DF[column_name]).abs() > 30 * math.pi / 180 , column_name ] = None
                new_DF.loc[ (new_DF[column_name]).abs() <-15 * math.pi / 180 , column_name ] = None
                
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
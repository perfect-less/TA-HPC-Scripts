import pandas as pd
import numpy as np
import math
from math import sin, cos, asin, sqrt, pi, ceil, floor

def HaversineDistace(lat1, lat2, lon1, lon2):
    """Calculate Distance between two point on earth"""
    r = 6371 # km

    h = sin( (lat2 - lat1)/2 )**2 + cos(lat1)*cos(lat2)*( sin( (lon2 - lon1)/2 )**2 )
    # print (h)
    return 2*r*math.atan2( math.sqrt( h ), math.sqrt( 1-h ) )


def CalculateSmoothVerticalSpeed(flight_DF: pd.DataFrame):
    """Add smooth_hdot_mps and delta_smooth_hdot_mps to flight DF"""

    flight_DF["smooth_hdot_mps"] = flight_DF["hdot_2_mps"].interpolate()
    flight_DF["smooth_hdot_mps"] = flight_DF["smooth_hdot_mps"].rolling(window=60).mean()

    flight_DF["delta_smooth_hdot_mps"] = flight_DF["smooth_hdot_mps"].shift(periods= -5) - flight_DF["hdot_2_mps"]


def TraceCutOffPoint(flight_DF: pd.DataFrame, cutOut_distance):
    """Calculate the index of the point in which cutOut_distance away
    from the landing spot (where 'wow'/weight on wheels became one)"""

    Tmax = flight_DF.loc[ flight_DF.shape[0] - 1, "time_s" ]
    touchdown_index = flight_DF.loc[ (flight_DF["time_s"] > Tmax/2) & (flight_DF["wow"] < 0.5) ].index[0]

    flight_DF["lat_rad"] = flight_DF["lat_rad"].interpolate()
    flight_DF["lon_rad"] = flight_DF["lon_rad"].interpolate()
    flight_DF["hralt_m"] = flight_DF["hralt_m"].interpolate()
    flight_DF["loc_dev_ddm"] = flight_DF["loc_dev_ddm"].interpolate()

    trace_dist  = 0
    trace_index = touchdown_index

    # Trace back to cut off distance, from landing point
    while (
        trace_dist < cutOut_distance and trace_index > 0 and
        abs(flight_DF.loc[trace_index, "loc_dev_ddm"]) <= 1.5 * math.pi / 180 and
        not (flight_DF.loc[trace_index, "hralt_m"] > 400 and
        flight_DF.loc[trace_index, "smooth_hdot_mps"] > -0.4)
    ):

        trace_index -= 1
        trace_dist = HaversineDistace(flight_DF.loc[trace_index, "lat_rad"], flight_DF.loc[touchdown_index, "lat_rad"], flight_DF.loc[trace_index, "lon_rad"], flight_DF.loc[touchdown_index, "lon_rad"])

    return touchdown_index, trace_index


def ReIndexFlightData(flight_DF, target_frequency: int, touchdown_index, selected_columns: list):
    """Return new flight_DF and touchdown_index with new index based on
    target_frequency[Hz]"""
    sampling_freq = 1/flight_DF["time_s"][1]
    divider = int (sampling_freq / target_frequency)

    i_list = [a*divider for a in range(int(flight_DF.shape[0]/divider))]
    selected_columns.append("smooth_hdot_mps")
    selected_columns.append("delta_smooth_hdot_mps")

    # remove duplicates
    selected_columns = list (set(selected_columns))

    flight_DF = pd.DataFrame( flight_DF.loc[i_list, selected_columns] )
    flight_DF.reset_index(drop=True, inplace=True)

    touchdown_index = floor(touchdown_index/divider)

    return flight_DF, touchdown_index


def CalculateAdditionalColumn(flight_DF, touchdown_index, fparam_np):
    """Add distance to land columnt and fparam(flight param) which
    contains time range of approach phase."""
    distance_to_landing_arr = np.zeros((flight_DF.shape[0],))
    for ik in range(flight_DF.shape[0]):
        distance_to_landing_arr[ik] = HaversineDistace(flight_DF.loc[ik, "lat_rad"], flight_DF.loc[touchdown_index, "lat_rad"], flight_DF.loc[ik, "lon_rad"], flight_DF.loc[touchdown_index, "lon_rad"])
    distance_to_landing_arr = distance_to_landing_arr.reshape((distance_to_landing_arr.shape[0], 1))

    return pd.concat( [flight_DF, pd.DataFrame(distance_to_landing_arr, columns=["dist_to_land"]), pd.DataFrame(fparam_np, columns=["fParam"])], axis=1 )

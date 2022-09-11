"""This module is where data cleaning related function were stored"""

import pandas as pd
import numpy as np
import math

from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners.utils.filters import *

def CleanColumns(flight_DF):
    column_list = G_PARAMS.COLUMN_LIST
    filter_list = G_PARAMS.FILTER_LIST
    left_alone_list = G_PARAMS.LEFT_ALONE_LIST

    for column_name in column_list:
        movingAverageWin = 0 # 50

        if not G_PARAMS.SKIP_FILTER_PROCESS:

            # Deleting Outliars                
            if column_name == "elv_l_rad":
                # Just Delete all the out of range elevator param.
                flight_DF.loc[ (flight_DF[column_name]).abs() > 15 * math.pi / 180 , column_name ] = None
                
            if column_name == "elv_r_rad":
                # Just Delete all the out of range elevator param.
                flight_DF.loc[ (flight_DF[column_name]) < 50 * math.pi / 180 , column_name ] = None

            if column_name == "ail_l_rad":
                # Just Delete all the out of range aileron param.
                flight_DF.loc[ (flight_DF[column_name]) < 60 * math.pi / 180 , column_name ] = None
                
            if column_name == "ail_r_rad":
                # Just Delete all the out of range aileron param.
                flight_DF.loc[ (flight_DF[column_name]) < 60 * math.pi / 180 , column_name ] = None
                

            flight_DF[column_name] = flight_DF[column_name].interpolate()


            # Butter Filtering
            if column_name in filter_list:
                flight_DF[column_name] = pd.Series( butter_lowpass_filter(flight_DF[column_name].to_numpy(), cutoff, fs, order) )                    


            # Moving average filtering
            if column_name in left_alone_list:
                movingAverageWin = 0

            if movingAverageWin > 0:
                flight_DF[column_name] = flight_DF[column_name].rolling(window=movingAverageWin).mean()
    
    return flight_DF



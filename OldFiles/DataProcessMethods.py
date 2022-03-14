import pandas as pd 
import numpy as np

from os import listdir
from os.path import isfile, join

import math
import shutil 
import random


## Get Files Names in a Folder
def GetFilesName(mypath, printIt = True):
    "Get all Files's Name inside a Folder"
    #mypath = "Flights_35/"

    flight_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    av_flights_num = len(flight_files)
    
    if printIt == True:
        print(".Number of Available Flights: {}".format(av_flights_num))
    
    return flight_files


## Normalization
def normalized(data, z, s, k = 1, minmax= False):
    return _norm(data, z, s, k, minmax), z, s
    #return data, z, s

def _norm(data, z, s, k = 1, minmax= False):
    if minmax == True:
        return (data - z) / (s - z) # z: min; s: max
    else:
        return (data - z) / (k * s) # z:mean; s: std
    #return data

def denorm(data, z, s, k = 1, minmax= False):
    if minmax == True:
        return data * (s - z) + z
    else:
        return data * k * s + z
    #return data

def DF_Nomalize(our_DF, ColParam = None):
    
    columns = our_DF.columns
    
    if ColParam == None:
        ColParam = {}
    
        for column in columns:
            min_max = False
            
            if column == "flap_te_pos":
                min_max = True
                #our_DF[column], z, s = normalized(our_DF[column], our_DF.min()[column], our_DF.max()[column], minmax=min_max)
            else:
                our_DF[column], z, s = normalized(our_DF[column], our_DF.mean()[column], our_DF.std()[column], minmax=min_max)


            ColParam[column] = (z, s)
            
        return our_DF, ColParam
    
    else:
        for column in columns:
            min_max = False
            if column == "flap_te_pos":
                min_max = True
            else:
                our_DF[column] = _norm(our_DF[column], ColParam[column][0], ColParam[column][1], minmax= min_max)

        
        return our_DF
    













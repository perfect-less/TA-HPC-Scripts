import pandas as pd 
import numpy as np

from os import listdir
from os.path import isfile, join
import DataProcessMethods as dpm

import math
import shutil 
import random


# Importing all the datas

D_Id = "_cut_UC"
file_path =  "Processed Data{}/Approach/".format(D_Id)
write_path = "Processed Data{}/Selected/".format(D_Id)
flight_files = dpm.GetFilesName(file_path)

flightDFs = list()
for i in range(len(flight_files)):
    flightDFs.append( pd.read_csv("{path}{fname}".format(path= file_path, fname= flight_files[i])) )
    
len(flightDFs)

if True:
    col_name = "theta_rad"

    for i in range(len(flight_files)):

        flightDFs[i]["theta_del_rad"] = flightDFs[i][col_name] - flightDFs[i]["theta_trim_rad"]

        #lightDFs[i] = flightDFs[i].drop(columns= ["Unnamed: 0"])

        flightDFs[i].to_csv("{path}{fname}".format(path= file_path, fname= flight_files[i]), index = False)

# Calculating Max C
max_c = 0
rad2deg = 180 / math.pi

# Set Max length for x axis
for i in range(len(flightDFs)):
    if flightDFs[i].shape[0] > max_c:
        max_c = flightDFs[i].shape[0]

print ("max_C :", max_c)


## First Pass

i_zeros = list()

min_elv = 10000
max_elv =-10000
zeroest = 10000

i_zeros = list()
empty_n = 0

for i in range(len(flightDFs)):
    #print( ".{}; {}".format(i, flightDFs[i].elv_l_rad.diff().mean()) )
    flightDFs[i]['elv_l_rad'].replace('', np.nan, inplace=True)
    
    std = flightDFs[i].elv_l_rad.diff().std()
    mean = flightDFs[i].elv_l_rad.diff().mean()
    
    if (min_elv > std):
        min_elv = std
        
    if (max_elv < std):
        max_elv = std
        
    if (abs(zeroest) > abs(std)):
        zeroest = std
        
    ## Selecting
        
    if (abs(std) <= 0.0008): #10c: 0.0005  #uc: 0.0015 #best uc old = 0.001   0.0028
        i_zeros.append(i)
        #pass
        
    #if (mean > 0.01 / rad2deg):
    #    i_zeros.append(i)
    
    if (flightDFs[i].elv_l_rad.mean() > 0.01 / rad2deg):
        #i_zeros.append(i)
        pass
        
    if (flightDFs[i].elv_l_rad.max() > 1 / rad2deg):
        i_zeros.append(i)
        #pass
        
    if flightDFs[i].shape[0] == 0:
        i_zeros.append(i)
        empty_n += 1
        
    if flightDFs[i].shape[0] > 0 and np.isnan(flightDFs[i].loc[random.randint(0, flightDFs[i].shape[0]-1) , "elv_l_rad"]):
        i_zeros.append(i)
        empty_n += 1

    if (flightDFs[i].theta_trim_rad.min() < -6 / rad2deg): # -10
        i_zeros.append(i)
        empty_n += 1
        
print("min_elv = {}".format(min_elv * rad2deg))
print("max_elv = {}".format(max_elv * rad2deg))
print("zeroest = {}".format(zeroest * rad2deg))

print("numbers of i_zeros: {}".format ( len(i_zeros) ))
print("numbers of empty n: {}".format ( empty_n ))
print("total usable files: {}".format ( len(flight_files) - len(i_zeros) ))

old_i_zeros = i_zeros.copy()



## Second Pass

min_loc = 10000
max_loc =-10000
zeroest = 10000
empty_n = 0
deg2rad = math.pi / 180

new_i_zeros = list()

for i in range(len(flightDFs)):
    #print( ".{}; {}".format(i, flightDFs[i].elv_l_rad.diff().mean()) )
    if (i not in old_i_zeros):
        flightDFs[i]['loc_dev_ddm'].replace('', np.nan, inplace=True)

        std = flightDFs[i].loc_dev_ddm.diff().std()
        mean = flightDFs[i].loc_dev_ddm.diff().mean()

        if (min_loc > std):
            min_loc = std

        if (max_loc < std):
            max_loc = std

        if (abs(zeroest) > abs(std)):
            zeroest = std

        ## Selecting
        
        if flightDFs[i].loc_dev_ddm.diff().std() < 0.0007:
            #new_i_zeros.append(i)
            pass
            
        if flightDFs[i].gs_dev_ddm.diff().mean() < -0.0005:
            #new_i_zeros.append(i)
            pass
        
        #if (abs(std) <= 2.8): #10c: 0.0005  #uc: 0.0015
        #    new_i_zeros.append(i)

        if (max(abs(flightDFs[i].loc_dev_ddm.min()), abs(flightDFs[i].loc_dev_ddm.max())) > 1.5 / rad2deg):
            #new_i_zeros.append(i)
            pass
            
        if (flightDFs[i].lat_rad.max() > 35.047973 * deg2rad):
            new_i_zeros.append(i)

        if (flightDFs[i].loc[flightDFs[i]["lat_rad"].shape[0]-1, "lat_rad"] < 35.012 * deg2rad):
            new_i_zeros.append(i)
        #if (mean > -1.1):
        #    new_i_zeros.append(i)
        
        #if flightDFs[i].shape[0] > 1300:
        #    new_i_zeros.append(i)
        
        #if True and np.isnan(flightDFs[i].loc[random.randint(0, flightDFs[i].shape[0]-1) , "cas_mps"]):
        #    new_i_zeros.append(i)
        #    empty_n += 1
        

print("numbers of ni_zeros: {}".format ( len(new_i_zeros) ))
print("numbers of empty n : {}".format ( empty_n ))
print("total usable files : {}".format ( len(flight_files) - len(old_i_zeros) - len (new_i_zeros)))

i_zeros = old_i_zeros.copy()
i_zeros = old_i_zeros + new_i_zeros



## Copying all usable files in to one folder

if True:
    D_Id = "_cut_UC"
    file_path =  "Processed Data{}/Approach/".format(D_Id)
    write_path = "Processed Data{}/Selected/".format(D_Id)

    print ("START COPYING")
    for i in range(len(flight_files)):
        if i in i_zeros:
            continue

        origin = "{o_path}{fname}".format(o_path= file_path, fname = flight_files[i])
        target = "{w_path}{fname}".format(w_path= write_path, fname = flight_files[i])

        shutil.copyfile(origin, target)

    print ("DONE COPYING")

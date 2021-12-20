import pandas as pd 
import numpy as np
from math import ceil

import shutil
from os import listdir
from os.path import isfile, join

def GetFilesName(mypath, printIt = True):
    #mypath = "Flights_35/"

    flight_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    av_flights_num = len(flight_files)
    
    if printIt == True:
        print(".Number of Available Flights: {}".format(av_flights_num))
    
    return flight_files





# Importing all the datas

F_Id = "uc"
D_Id = "_cut_UC"

file_path =  "Processed Data{}/Selected/".format (D_Id)
write_path = "Processed Data{}/Ready/".format (D_Id)
copy_path = "Processed Data{}/Test/".format (D_Id)
flight_files = GetFilesName(file_path)

ind = np.arange(len(flight_files))
np.random.shuffle(ind)

print ("There are {} available files".format (len(flight_files)))

train_num = ceil (0.7 * len(flight_files))
test_num  = len(flight_files) - train_num

print ("..{} of it will be in training set".format (train_num))
print ("..and {} of it will be in test set".format (test_num))

print ("------Start Creating Data Set------")
flightDFs = list()
for i in range(len(flight_files)):
    file = pd.read_csv("{path}{fname}".format(path= file_path, fname= flight_files[ind[i]]))
    
    hd = False
    mo = 'a'
    if i == 0 or i == train_num:
        hd = True
        mo = 'w'
        
    # Write Into One CSV.
    if i < train_num:
        file.to_csv("{}Train_set_{}.csv".format (write_path, F_Id), mode=mo, header=hd, index=False)
    else:
        file.to_csv("{}Test_set_{}.csv".format (write_path, F_Id), mode=mo, header=hd, index=False)
        
        origin = "{o_path}{fname}".format(o_path= file_path, fname = flight_files[ind[i]])
        target = "{w_path}{fname}".format(w_path= copy_path, fname = flight_files[ind[i]])
    
        shutil.copyfile(origin, target)
    
print ("------All Done------")
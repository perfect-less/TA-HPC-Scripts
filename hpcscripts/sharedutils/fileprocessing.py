from os import listdir
from os.path import isfile, join
from typing import List



def GetFilesName(file_path, printIt = True):
    """List files inside file_path"""

    flight_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    av_flights_num = len(flight_files)

    if printIt == True:
        print(".Number of Available Flights: {}".format(av_flights_num))

    return flight_files
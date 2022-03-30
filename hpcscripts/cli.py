import sys

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.cleaners.utils import datacleaning
from hpcscripts.selectors import flightselector

def main ():
    print ("CLI main")
    
    # Initialize Program
    pathhandler.InitDataDirectories()

    # Clean Raw Data to CSV
    cleantocsv.run (G_PARAMS.DATAPROCESSING_POOL)
    
    # Select usable data
    flightselector.run()

    # Create training csv and train data

    # Post process and evaluate model

    print ("done")



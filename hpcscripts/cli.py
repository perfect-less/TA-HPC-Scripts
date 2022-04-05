import sys

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.selectors import flightselector
from hpcscripts.trainers import anntrainer, traindatahandler

def main ():
    print ("Begin")
    
    # Initialize Program
    pathhandler.InitDataDirectories()

    # Clean Raw Data to CSV
    cleantocsv.run (G_PARAMS.DATAPROCESSING_POOL)
    
    # Select usable data
    flightselector.run()

    # Create training csv and train data
    traindatahandler.run()

    # Post process and evaluate model
    # anntrainer.run()

    print ("Done")



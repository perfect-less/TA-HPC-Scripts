import sys
import time
import datetime

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.selectors import flightselector
from hpcscripts.trainers import anntrainer, traindatahandler
from hpcscripts.postprocesses import rsquared
from hpcscripts.sharedutils.trainingutils import SetLowTFVerbose

def main ():
    SetLowTFVerbose()

    print ("HPCSCRIPTS called at {}".format(datetime.datetime.now()))
    start_time = datetime.datetime.now()
    
    # Initialize Program
    pathhandler.InitDataDirectories()


    # # Clean Raw Data to CSV
    # cleantocsv.run (G_PARAMS.DATAPROCESSING_POOL)
    
    # Select usable data
    # flightselector.run()

    # Create training csv and train data
    # traindatahandler.run()

    # Post process and evaluate model
    anntrainer.run()
    rsquared.run()

    # print ( "G_PARAMS, PARAMS: {}, {}, {}".format (
    #                     G_PARAMS.INPUT_WINDOW_WIDTH,
    #                     G_PARAMS.LABEL_WINDOW_WIDTH,
    #                     G_PARAMS.LABEL_SHIFT
    #                 ))
    
    # print ( "G_PARAMS, MODEL: \n{}".format(G_PARAMS.MODEL))
    # print ( "G_PARAMS, SEQUENTIAL_HIDDENLAYERS: {}".format(G_PARAMS.SEQUENTIAL_HIDDENLAYERS))

    # Runtime related calculation
    exit_time = datetime.datetime.now()
    run_time = exit_time - start_time

    print ("Exiting HPCSCRIPTS")
    print ("exit time {}".format(datetime.datetime.now()))
    print ("total runtime: {}".format( run_time ))



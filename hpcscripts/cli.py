import sys

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.cleaners.utils import datacleaning

def main ():
    print ("CLI main")
    
    pathhandler.InitDataDirectories()

    cleantocsv.run (G_PARAMS.DATAPROCESSING_POOL)
    

    print ("done")



import sys

from hpcscripts.option import pathhandler
from hpcscripts.cleaners import cleantocsv
from hpcscripts.cleaners.utils import datacleaning

def main ():
    print ("CLI main")
    
    pathhandler.InitDataDirectories()

    # cleantocsv.run (4)
    datacleaning.ReSampling("flight_10134.zip")
    

    print ("done")



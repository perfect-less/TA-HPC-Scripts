import sys 
import os
import glob

import hpcscripts.option.globalparams as G_PARAMS


def GetDataPath():
    return os.path.abspath(
        os.path.join(GetThisDir(), os.pardir, os.pardir, "Data")
    ) 

def GetRawDataPath():
    return os.path.join(GetDataPath(), "Raw")

def GetProcessedDataPath():
    return os.path.join(GetDataPath(), "Processed")

def GetModelsPath():
    return os.path.join(GetDataPath(), "Models")

def GetResultsPath():
    return os.path.join(GetDataPath(), "Results")

def GetThisDir():
    return os.path.dirname( os.path.abspath(__file__) )

def GetProcessedPath(dir="Sampled"):
    return os.path.join(GetProcessedDataPath(), dir)


def ClearProcessedDir(dir):
    files = glob.glob(os.path.join(GetProcessedPath(dir), '*'))

    for file in files:
        os.remove(file)


def InitDataDirectories():
    important_dir = G_PARAMS.DATA_DIRECORIES

    for dir in important_dir:
        full_path = os.path.join (GetThisDir(), os.pardir, os.pardir, dir)
        full_path = os.path.abspath (full_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)
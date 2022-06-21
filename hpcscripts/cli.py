import sys
import time
import datetime
import argparse

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.selectors import flightselector
from hpcscripts.trainers import anntrainer, traindatahandler
from hpcscripts.trainers import modeldefinitions as mdef
from hpcscripts.postprocesses import rsquared
from hpcscripts.sharedutils.trainingutils import SetLowTFVerbose


COMMAND_FLAG = {
    'clean': cleantocsv.run,
    'select': flightselector.run,
    'trainready': traindatahandler.run,
    'train': anntrainer.run,
    'post': rsquared.run
}

command_control = {
    'start': 'clean',
    'finish': 'post',
    'last': 'no'
}



def invalid_input(arg: str):
    print ("Invalid Argument -> {}", arg)
    exit()

def RunProcess(process_name: str):

    if process_name == 'clean':
        COMMAND_FLAG[process_name](G_PARAMS.DATAPROCESSING_POOL)
        return
    
    if process_name == 'post' and command_control['last'] == 'yes':
        COMMAND_FLAG[process_name](9999)
        return

    COMMAND_FLAG[process_name]()


def main ():
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser = create_parser(parser)

    args = parser.parse_args()
    process_args(args)
    
    # Begin Process
    print ("HPCSCRIPTS called at {}".format(datetime.datetime.now()))
    start_time = datetime.datetime.now()

    # Initialize Program
    pathhandler.InitDataDirectories()

    has_started = False
    for cmd_key in COMMAND_FLAG.keys():
        if not has_started and cmd_key == command_control['start']:
            has_started = True

        if has_started:
            RunProcess(
                    cmd_key
                )

        if cmd_key == command_control['finish']:
            break


    # Runtime related calculation
    exit_time = datetime.datetime.now()
    run_time = exit_time - start_time

    print ("Exiting HPCSCRIPTS")
    print ("exit time {}".format(datetime.datetime.now()))
    print ("total runtime: {}".format( run_time ))




# ==============PARSER==============

def create_parser(parser: argparse.ArgumentParser):

    prs = parser

    # Command flag
    prs.add_argument('-p', '--post', action="store_true",
					help = "Run only post-process command")

    prs.add_argument('-l', '--last', action="store_true",
					help = "Automatically select last model on post process")

    # Set Specific command
    prs.add_argument('--since', '--from', action="store",
					help = """Run from specified command,
					example: `hpc_scripts --since train` to run from train command onwards.""")

    prs.add_argument('--until', action="store",
					help = """Run up to specified command,
					example: `hpc_scripts --until train` to run up to train command.
                    Command begins on --since, if --since never specified, command run from beginning.""")

    prs.add_argument('--only', action="store",
					help = """Run only this particular command,
					example: `hpc_scripts --only clean` to only clean the datasets.""")

    return prs

def process_args(prs: argparse.ArgumentParser):
    if prs.last:        
        command_control['last'] = 'yes'

    if prs.post:        
        command_control['start'] = 'post'
        command_control['finish']= 'post'
        return

    if prs.only:
        if not str(prs.only) in COMMAND_FLAG.keys():
            invalid_input(str(prs.only))
        
        command_control['start'] = str(prs.only)
        command_control['finish']= str(prs.only)
        return
    
    if prs.since:
        if not str(prs.since) in COMMAND_FLAG.keys():
            invalid_input(str(prs.since))
        
        command_control['start'] = str(prs.since)

    if prs.until:
        if not str(prs.until) in COMMAND_FLAG.keys():
            invalid_input(str(prs.until))
        
        command_control['finish'] = str(prs.until)
import os
import datetime
import argparse

# TENSORFLOW LOGS:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from hpcscripts.option import pathhandler
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.cleaners import cleantocsv
from hpcscripts.selectors import flightselector
from hpcscripts.trainers import anntrainer, traindatahandler
from hpcscripts.postprocesses import rsquared
from hpcscripts.sharedutils.trainingutils import SetLowTFVerbose

#
# IMPORTANT DATA
#

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
    'last': 'no',
    'model_id': None
}

# 
# FUNCTION DEFINITION
#

def invalid_input(arg: str):
    print ("Invalid Argument -> {}", arg)
    exit()

def RunProcess(process_name: str):

        if process_name == 'clean':
            COMMAND_FLAG[process_name](G_PARAMS.DATAPROCESSING_POOL)
            return
        
        if process_name == 'train':
            COMMAND_FLAG[process_name](command_control['model_id'])
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
    print ()
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

#
# ==============PARSER==============
#
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
    
    # Model id
    prs.add_argument('--mid', nargs="*", type=str,
                    help = "Valid model id")

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

    if prs.mid and len(prs.mid) > 0:
        command_control['model_id'] = str(prs.mid[0])



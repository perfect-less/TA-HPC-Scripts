"""Calculating rsquared for Models in Data direcory"""
import os
from typing import List

import numpy as np
import pandas as pd
from tensorflow.python.keras.api import keras

from hpcscripts.sharedutils.trainingutils import LoadModel, SetLowTFVerbose, MakeSinglePrediction
from hpcscripts.sharedutils.nomalization import DF_Nomalize
from hpcscripts.sharedutils.modelutils import SelectModelPrompt
from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS


def CalculateRSquared(test_files: List[str], model: keras.Model, norm_param):
    
    labels = G_PARAMS.SEQUENTIAL_LABELS
    r2_list = []
    columns = ["filename"] + labels

    print ("calculating r2")

    for i, test_file in enumerate(test_files):
        test_df, predictions = MakeSinglePrediction(
                test_file, 
                model, 
                labels,
                norm_param
            )
        
        row_list = [os.path.basename(test_file)]
        for j, label in enumerate(labels):
            real_np = test_df[label].to_numpy()
            pred_np = predictions[j]

            r2 = 1 - np.sum( np.square(real_np-pred_np) )/np.sum( np.square(real_np-np.mean(real_np)) )
            row_list.append(r2)
        r2_list.append(row_list)

    r2_df = pd.DataFrame(r2_list, columns=columns)
    return r2_df

def run():

    # Make TensorFlow little bit quiter
    SetLowTFVerbose()

    # Get Training Data to calculate norm param
    train_file = os.path.join(ph.GetProcessedPath("Ready"), "Train_set.csv")
    train_data, norm_param = DF_Nomalize(pd.read_csv(train_file))
    
    model_path = SelectModelPrompt(ph.GetModelsPath())
    model, _ = LoadModel(model_path)

    # List test files
    test_dir = ph.GetProcessedPath("Test")
    test_files = os.listdir(test_dir)
    test_files = [os.path.join(test_dir, file_path) for file_path in test_files]

    # Calculate R squared
    r2_df = CalculateRSquared(test_files, model, norm_param)
    r2_df.to_csv(os.path.join(model_path, "r2.csv"), index=False)




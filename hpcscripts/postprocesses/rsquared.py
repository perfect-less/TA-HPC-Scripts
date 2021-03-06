"""Calculating rsquared for Models in Data direcory"""

import os
from math import pi
from typing import List

import numpy as np
import pandas as pd
from tensorflow import keras

from hpcscripts.sharedutils.trainingutils import LoadModel, SetLowTFVerbose, MakeSinglePrediction, CreateWindowGenerator
from hpcscripts.sharedutils.nomalization import DF_Nomalize, denorm
from hpcscripts.sharedutils.modelutils import SelectModelPrompt
from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS


def CalculateRSquared(test_files: List[str], model: keras.Model, norm_param):
    
    # Reconfiguring G_PARAMS
    # G_PARAMS.ApplyModelDefinition()

    labels = G_PARAMS.SEQUENTIAL_LABELS
    r2_list = []
    columns = ["filename"]
    for label in labels:
        for metric in ["r2", "mae", "mse"]:
            columns.append("{}_{}".format(metric, label))
            if label.endswith("rad") and metric == "mae":
                columns.append("{}_{}".format(metric, label.removesuffix("rad") + "deg"))

    windowG = CreateWindowGenerator(
                    train_list=test_files,
                    test_list=None, eval_list=None,
                    norm_param=norm_param
    )

    print ("calculating r2")

    for i, test_file in enumerate(test_files):
        test_df, predictions = MakeSinglePrediction(
                csvfile_path=test_file, 
                model=model, 
                window=windowG
            )
        
        row_list = [os.path.basename(test_file)]
        for j, label in enumerate(labels):
            real_np = test_df[label].to_numpy()
            pred_np = predictions[:, j]

            # Calculate R^2
            r2 = 1 - np.sum( np.square(real_np-pred_np) )/np.sum( np.square(real_np-np.mean(real_np)) )

            # Calculate MAE
            z, s = norm_param[label]
            real_np = denorm(real_np, z, s)
            pred_np = denorm(pred_np, z, s)
            mae = keras.metrics.mean_absolute_error(
                    real_np,
                    pred_np
                )
            mae = float(mae)

            # Calculate MSE
            mse = keras.metrics.mean_squared_error(
                    real_np,
                    pred_np
                )
            mse = float(mse)

            # Convert to degrees
            if label.endswith("rad"):
                mae_deg = mae * 180/pi

            # Apending files r2, MAE, and MSE per label
            if label.endswith("rad"):
                row_list.append(r2)
                row_list.append(mae)
                row_list.append(mae_deg)
                row_list.append(mse)
            else:
                row_list.append(r2)
                row_list.append(mae)
                row_list.append(mse)
        r2_list.append(row_list)

    r2_df = pd.DataFrame(r2_list, columns=columns)
    return r2_df

def run(selected_index: int=None, optional_path: str=None):

    # Make TensorFlow little bit quiter
    SetLowTFVerbose()

    # Get Training Data to calculate norm param
    train_file = os.path.join(ph.GetProcessedPath("Combined"), "Train_set.csv")
    train_data, norm_param = DF_Nomalize(pd.read_csv(train_file))
    
    models_path = ph.GetModelsPath() if optional_path==None else optional_path

    model_path = SelectModelPrompt(models_path, selected_index)
    model, _, modelsmeta = LoadModel(model_path)

    G_PARAMS.SetParams(modelsmeta['param'])

    # List test files
    test_dir = ph.GetProcessedPath("Test")
    test_files = os.listdir(test_dir)
    test_files = [os.path.join(test_dir, file_path) for file_path in test_files]

    # Calculate R squared
    r2_df = CalculateRSquared(test_files, model, norm_param)
    r2_df.to_csv(os.path.join(model_path, "r2.csv"), index=False)




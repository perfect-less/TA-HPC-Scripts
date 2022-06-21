"""This module is responsible for creating ANN model and train it based on configuration"""

import os
from typing import List

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from hpcscripts.sharedutils.nomalization import *
from hpcscripts.sharedutils.trainingutils import *
from hpcscripts.trainers.windowmanager import WindowGenerator
from hpcscripts.trainers.modeldefinitions import ResidualWrapper
from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS


def run(model_id: str=None):
    SetLowTFVerbose()
    print ("--------------------")
    print ("----Begin Training Process")

    # Set Global parameters
    # if not model_id == None:
    #     if not model_id in mdef.MODEL_DEFINITIONS.keys():
    #         print ("ERR: Invaild model id -> {}".format(model_id))
    #         exit()

    #     _param, _model = mdef.MODEL_DEFINITIONS[model_id]()

    #     G_PARAMS.MODEL = _model
    #     G_PARAMS.SetParams(_param)
    #     G_PARAMS.PARAMS = _param
    # else:
    #     model_id = ''

    
    # Import Data
    train_comb= ImportCombinedTrainingData()

    # Pre-process Data
    train_comb, norm_param = DF_Nomalize(train_comb)
    train_list, test_list, eval_list = GetFileList()

    # Create ANN Model
    model = CreateANNModel()

    # Create WindowGenerator
    windowG = CreateWindowGenerator(train_list, 
                    test_list, eval_list, norm_param)

    print ("model ready..")
    print ("begin training..\n")

    # Train Model
    history = TrainModel(
        model, 
        windowG.train, 
        windowG.val,
        G_PARAMS.CALLBACKS,
        epochs=G_PARAMS.TRAIN_EPOCHS
    )

    # Save Model and Training history
    SaveModel(model, model_id, history)
    print ("---------------------------------------------")




def ImportCombinedTrainingData():
    train_file = os.path.join(ph.GetProcessedPath("Combined"), "Train_set.csv")
    train_data = pd.read_csv(train_file)

    return train_data

def GetFileList():
    train_dir = ph.GetProcessedPath("Train")
    test_dir  = ph.GetProcessedPath("Test")
    eval_dir  = ph.GetProcessedPath("Eval")

    train_list = os.listdir(train_dir)
    test_list  = os.listdir(test_dir)
    eval_list  = os.listdir(eval_dir)

    for i, train_file in enumerate (train_list):
        train_list[i] = os.path.join(train_dir, train_file)

    for i, test_file in enumerate (test_list):
        test_list[i] = os.path.join(test_dir, test_file)

    for i, eval_file in enumerate (eval_list):
        eval_list[i] = os.path.join(eval_dir, eval_file)

    return train_list, test_list, eval_list


def CreateANNModel():
    """Create and Compile Model"""
    # Get Model Definition from Global Param
    model = G_PARAMS.MODEL
    
    # Add output layer
    model.add(keras.layers.Dense(
                    units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                    kernel_initializer=tf.initializers.zeros()
            ))
    model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

    # Apply Residual Wrapper
    if G_PARAMS.USE_RESIDUAL_WRAPPER:
        model = ResidualWrapper(model)

    # Compile our model
    model.compile(optimizer=G_PARAMS.OPTIMIZER,
                loss=G_PARAMS.LOSS,
                metrics=G_PARAMS.METRICS
                )

    return model





## ======================================================================
## DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED
## ======================================================================
def CreateFeatureLayer():
    feature_columns = []

    # Represent altitude (hralt) as a floating-point value.
    hralt_m = tf.feature_column.numeric_column("hralt_m")
    feature_columns.append(hralt_m)

    # Represent theta_rad as a floating-point value.
    theta_rad = tf.feature_column.numeric_column("theta_rad")
    feature_columns.append(theta_rad)

    theta_del_rad = tf.feature_column.numeric_column("theta_del_rad")
    feature_columns.append(theta_del_rad)

    # Represent aoac_rad as a floating-point value.
    aoac_rad = tf.feature_column.numeric_column("aoac_rad")
    feature_columns.append(aoac_rad)

    # Represent cas_mps as a floating-point value.
    cas_mps = tf.feature_column.numeric_column("cas_mps")
    feature_columns.append(cas_mps)

    # Represent flap_te_pos as a floating-point value.
    hdot_2_mps = tf.feature_column.numeric_column("hdot_2_mps")
    feature_columns.append(hdot_2_mps)

    # Represent flap_te_pos as a bucketized floating-point value.
    flap_te_pos_num = tf.feature_column.numeric_column("flap_te_pos")
    flap_te_pos = tf.feature_column.bucketized_column(flap_te_pos_num, [5, 23, 27.5, 34])
    feature_columns.append(flap_te_pos)

    feature_layer = keras.layers.DenseFeatures(feature_columns)
    return feature_layer

def ToTFReadyFeatureAndLabel(train_data, test_data, eval_data):
    training_labels = G_PARAMS.SEQUENTIAL_LABELS

    train_features, train_label = SplitDataFrame(train_data, training_labels)
    test_features, test_label = SplitDataFrame(test_data, training_labels)
    eval_features, eval_label = SplitDataFrame(eval_data, training_labels)

    return (
        (train_features, train_label), 
        (test_features, test_label),
        (eval_features, eval_label)
    )

def PreProcessData(train_data: pd.DataFrame, test_data: pd.DataFrame, eval_data: pd.DataFrame):
    train_data, norm_param = DF_Nomalize(train_data)
    test_data = DF_Nomalize(test_data, norm_param)
    eval_data = DF_Nomalize(eval_data, norm_param)

    return train_data, test_data, eval_data, norm_param
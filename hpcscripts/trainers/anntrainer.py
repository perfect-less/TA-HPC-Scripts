"""This module is responsible for creating ANN model and train it based on configuration"""

import os
from typing import List

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.api import keras

from hpcscripts.sharedutils.nomalization import *
from hpcscripts.sharedutils.trainingutils import *
from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS


def run():
    SetLowTFVerbose()
    print ("--------------------")
    print ("----Begin Training Process")

    # Import Data
    train_data, test_data, eval_data = ImportTrainingData()

    # Pre-process Data
    train_data, test_data, eval_data, norm_param = PreProcessData(train_data, test_data, eval_data)
    ready_data = ToTFReadyFeatureAndLabel(train_data, test_data, eval_data)
    train_features, train_labels = ready_data[0]
    test_features, test_labels =   ready_data[1]
    eval_features, eval_labels =   ready_data[2]

    # Create Feature Column
    feature_layer = CreateFeatureLayer()

    print ("features ready..")

    # Create ANN Model
    model = CreateANNModel(feature_layer)

    print ("model ready..")
    print ("begin training..\n")

    # Train Model
    history = TrainModel(
        model, 
        (train_features, train_labels), 
        (eval_features, eval_labels),
        G_PARAMS.CALLBACKS,
        epochs=G_PARAMS.TRAIN_EPOCHS
    )

    # Save Model and Training history
    SaveModel(model, history)
    print ("---------------------------------------------")




def ImportTrainingData():
    train_file = os.path.join(ph.GetProcessedPath("Ready"), "Train_set.csv")
    test_file  = os.path.join(ph.GetProcessedPath("Ready"), "Test_set.csv")
    eval_file  = os.path.join(ph.GetProcessedPath("Ready"), "Eval_set.csv")

    train_data = pd.read_csv(train_file)
    test_data  = pd.read_csv(test_file)
    eval_data  = pd.read_csv(eval_file)

    return train_data, test_data, eval_data

def PreProcessData(train_data: pd.DataFrame, test_data: pd.DataFrame, eval_data: pd.DataFrame):
    train_data, norm_param = DF_Nomalize(train_data)
    test_data = DF_Nomalize(test_data, norm_param)
    eval_data = DF_Nomalize(eval_data, norm_param)

    return train_data, test_data, eval_data, norm_param

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

def CreateANNModel(feature_layer):
    """Create and Compile Model"""
    # Create Model
    model = CreateSequentialModel(
        feature_layer,
        G_PARAMS.SEQUENTIAL_HIDDENLAYERS,
        G_PARAMS.ACTIVATION
    )

    # Compile our model
    model.compile(optimizer=keras.optimizers.Adam(name='Adam'),
                loss=G_PARAMS.LOSS,
                metrics=G_PARAMS.METRICS
                )

    return model

def CreateSequentialModel(feature_layer, hidden_layer_conf: List[int], activation):

    # Create a sequential model
    model = keras.models.Sequential()

    # Create input layer
    model.add(feature_layer)

    for nodes_count in hidden_layer_conf:
        model.add(keras.layers.Dense(
                            units=nodes_count,
                            activation=activation
                        ))

    # Create output layer
    model.add(keras.layers.Dense(units=len(G_PARAMS.SEQUENTIAL_LABELS)))

    return model


import os
import pickle
import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from hpcscripts.option import pathhandler as ph
from hpcscripts.option import globalparams as G_PARAMS
from hpcscripts.sharedutils.nomalization import DF_Nomalize
from hpcscripts.trainers.windowmanager import WindowGenerator


def SetLowTFVerbose():
    tf.get_logger().setLevel('ERROR')

def TrainModel(model, training_data, eval_data, callbacks, epochs=10):
    
    history = model.fit(
        training_data,
        epochs=epochs,
        validation_data=eval_data,
        callbacks=callbacks
    )

    return history

def CreateWindowGenerator(train_list, test_list, eval_list, norm_param:dict):
    windowG = WindowGenerator(

            input_width=G_PARAMS.INPUT_WINDOW_WIDTH,
            label_width=G_PARAMS.LABEL_WINDOW_WIDTH,
            shift=G_PARAMS.LABEL_SHIFT,
            
            train_list=train_list,
            test_list=test_list,
            val_list=eval_list,
            
            norm_param=norm_param,
            label_columns=G_PARAMS.SEQUENTIAL_LABELS,
            
            shuffle_train=True,
            print_check=False

                )
    return windowG

def MakeSinglePrediction(csvfile_path: str, 
                        model: keras.Model, 
                        window: WindowGenerator=None):
    
    ds = window.make_dataset([csvfile_path], batch_size=None)
    features, labels = next(iter(ds))
    
    predictions = model.predict(features)
    
    _rows = predictions.shape[0]
    predictions = predictions.reshape(_rows, len(G_PARAMS.SEQUENTIAL_LABELS))
    
    test_df = DF_Nomalize (pd.read_csv(csvfile_path), window.norm_param)
    test_df = test_df.iloc[(window.total_window_size-1):, :]
    # ^ cut test_df so it have the same amount of rows with predictions
    assert _rows == test_df.shape[0]

    return test_df, predictions

def SaveModel(model: keras.Model, history):
    """Save both model and history"""
    folder_name = "ANN " + str (datetime.datetime.now())[:-7]
    model_directory = os.path.join (ph.GetModelsPath(), folder_name)
    history_file = os.path.join(model_directory, 'history.pkl')

    model.save(model_directory)
    print ("Model saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))

def LoadModel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("model loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history





## ======================================================================
## DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED, DEPRECATED
## ======================================================================
def SplitDataFrame(df: pd.DataFrame, labels):
    """Split DataFrames into feature dictionary and labels array.
    We do this because TensorFlow accept this formating."""

    feature_dict = {name:np.array(value) for name, value in df.items()}

    for i, label_name in enumerate(labels):
        if i == 0:
            label_array = feature_dict.pop(label_name)
            continue

        label_array = np.stack(
            (
                label_array, 
                np.array(feature_dict.pop(label_name))
            )
        )

    label_array = np.transpose(label_array)
    return feature_dict, label_array
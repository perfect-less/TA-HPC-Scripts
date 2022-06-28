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
from hpcscripts.trainers import modeldefinitions as mdef


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

def SaveModel(model: keras.Model, history, model_id: str = None, optional_path: str=None):
    """Save both model and history"""
    folder_name = "ANN " + str (datetime.datetime.now())[:-7]
    if optional_path != None:
        model_directory = os.path.join (optional_path, folder_name)
    else:
        model_directory = os.path.join (ph.GetModelsPath(), folder_name)

    history_file = os.path.join(model_directory, 'history.pkl')
    modelsmeta_file = os.path.join(model_directory, 'modelsmeta.pkl')

    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    param = {
        'model_id': G_PARAMS.MODEL_ID if model_id == None else model_id,
        'param' : G_PARAMS.PARAMS
    }

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))
    with open(modelsmeta_file, 'wb') as mf:
        pickle.dump(param, mf)
    print ("Model metadata saved to {}".format(modelsmeta_file))

def LoadModel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    modelsmeta_file = os.path.join(path_to_model, 'modelsmeta.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")
    with open(modelsmeta_file, 'rb') as nf:
        modelsmeta = pickle.load(nf)
    print ("model metadata loaded")

    return model, history, modelsmeta





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
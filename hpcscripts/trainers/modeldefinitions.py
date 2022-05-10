import tensorflow as tf
from tensorflow.python.keras.api import keras

from hpcscripts.option import globalparams as G_PARAMS

def AddDenseHiddenLayer(model):
    hidden_layer_conf = G_PARAMS.SEQUENTIAL_HIDDENLAYERS
    activation = G_PARAMS.ACTIVATION

    for nodes_count in hidden_layer_conf:
        model.add(keras.layers.Dense(
                            units=nodes_count,
                            activation=activation
                        ))
    
    return model


# V V -  -  -  -  -  -  -  -  -  -  V V
# V V  MODEL DEFINITION BEGIN HERE  V V
# V V -  -  -  -  -  -  -  -  -  -  V V

## Multi-step Linear Model with 1 Dense
def Linear():
    linear = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])
    return linear

## Conv. Dense with HiddenLayer
def Conv():
    conv = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(G_PARAMS.INPUT_WINDOW_WIDTH),
                            activation='relu')
    ])
    conv = AddDenseHiddenLayer(conv)
    return conv





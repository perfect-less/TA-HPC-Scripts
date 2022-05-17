import tensorflow as tf
from tensorflow import keras

from hpcscripts.option import globalparams as G_PARAMS

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    # 
    # PLEASE NOTED THAT FOR THIS TO WORK, LABELS MUST BE PUT ON THE LAST COLUMNS
    # eg.
    # for data with columns: {"alpha", "gamma", "beta", "theta", "lambda"}
    # if we want "beta" and "gamma" as our label -> ["beta", "gamma"]
    # then our feature must be ["alpha", "theta", "lambda", "beta", "gamma"]
    return inputs[:, -(G_PARAMS.LABEL_WINDOW_WIDTH+1):-1, -len(G_PARAMS.SEQUENTIAL_LABELS):] + delta

def AddDenseHiddenLayer(model, sequential_hiddenlayers=None):
    
    if sequential_hiddenlayers == None:
        hidden_layer_conf = G_PARAMS.SEQUENTIAL_HIDDENLAYERS
    else:
        hidden_layer_conf = sequential_hiddenlayers
    
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

# Model Definition didn't have to specify 
# the last layer (output layer) because 
# the trainers module should have taken 
# care of that.

# Each model definition function returns
# _param and model objects. Set _param as
# None if you want to use default parameters
# on globalparams.py

## Multi-step Linear Model with 1 Dense
def Linear():
    _param = None
    _input_window_width = 1
    _label_window_width = 1
    _label_shift        = 0 

    linear = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    _param = (_input_window_width, _label_window_width, _label_shift, None)
    return _param, linear

## Conv. Dense with Default HiddenLayer
def Conv():
    _param = None

    conv = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(G_PARAMS.INPUT_WINDOW_WIDTH),
                            activation='relu')
    ])
    conv = AddDenseHiddenLayer(conv)
    return _param, conv

## Conv. Dense with Custom HiddenLayer
def Conv_CustomHiddenLayer():
    _param = None
    _input_window_width  = 5
    _label_window_width  = 1
    _label_shift         = 0
    _sequential_hidden_l = [30, 30]

    conv = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(_input_window_width),
                            activation='relu')
    ])
    conv = AddDenseHiddenLayer(conv, _sequential_hidden_l)

    _param = (_input_window_width, _label_window_width, _label_shift, _sequential_hidden_l)
    return _param, conv


# DefaultModelDefinition
DefaultModelDefinition = Conv_CustomHiddenLayer
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
    return inputs[:, -(G_PARAMS.LABEL_WINDOW_WIDTH):, -len(G_PARAMS.SEQUENTIAL_LABELS):] + delta

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

def ModelDefBuilder(
    input_window_width  = 1,
    label_window_width  = 1,
    label_shift         = 1,
    sequential_hidden_l = None,
    feature_columns    = None,
    seq_labels         = None,
    use_residual_wrap  = None,

    none_param = False,
    model = None
):
    param = None
    if none_param == False:
        param = (
            input_window_width, label_window_width, label_shift, 
            sequential_hidden_l,
            feature_columns, seq_labels,
            use_residual_wrap
        )
    return param, model


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

    linear = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    return ModelDefBuilder(
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        model=linear
    )

## Conv. Dense with Default HiddenLayer
def Conv():
    conv = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(G_PARAMS.INPUT_WINDOW_WIDTH),
                            activation='relu')
    ])
    conv = AddDenseHiddenLayer(conv)

    return ModelDefBuilder(
        none_param=True,

        model=conv
    )

## Conv. Dense with Custom HiddenLayer
def Conv_CustomHiddenLayer():

    input_window_width = 5
    sequential_hidden_l = [30, 30]

    conv = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(input_window_width),
                            activation='relu')
    ])
    conv = AddDenseHiddenLayer(conv, sequential_hidden_l)

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        sequential_hidden_l = sequential_hidden_l,
        feature_columns    = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps", 'elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = True,

        model=conv
    )

## Simple LSTM with Custom HiddenLayer
def LSTM_CustomHiddenLayer():
    
    sequential_hidden_l = [30, 30]
    
    lstm = tf.keras.Sequential([
        tf.keras.layers.LSTM(32)
    ])
    lstm = AddDenseHiddenLayer(lstm, sequential_hidden_l)

    return ModelDefBuilder(
        input_window_width  = 5,
        label_window_width  = 1,
        label_shift         = 1,
        sequential_hidden_l=sequential_hidden_l,
        seq_labels         = ['elv_l_rad'],

        model=lstm
    )

#
# MODEL_DEFINITIONS dictionary
MODEL_DEFINITIONS = {
    'default'       : Linear,
    'linear'        : Linear,
    'conv_simple'   : Conv,
    'conv_hidden'   : Conv_CustomHiddenLayer,
    'lstm_hidden'   : LSTM_CustomHiddenLayer
}
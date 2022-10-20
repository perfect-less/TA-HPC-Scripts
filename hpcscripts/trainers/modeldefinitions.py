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

class ForecastBaseline(tf.keras.Model):
  def __init__(self):
    super().__init__()

  def call(self, inputs, *args, **kwargs):
    
    # This is the baseline model for forcasting problem
    # this model basicaly takes the input and then output the same value
    # as predictions. Our forcasting model should improve upon this
    # 
    # PLEASE NOTED THIS WORKS THE SAME WAY WITH RESIDUAL WRAPPER
    return inputs[:, -(G_PARAMS.LABEL_WINDOW_WIDTH):, -len(G_PARAMS.SEQUENTIAL_LABELS):]

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

def Simple_Dense():

    sequential_hidden_l = [20]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=['hralt_m', 'theta_rad', 'cas_mps', 'gamma_error_rad', 'hdot_1_mps'],
        input_window_width=5,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Simple_Dense_Ail():

    sequential_hidden_l = [20]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=['hralt_m', 'phi_rad', 'loc_dev_ddm'],
        seq_labels=['ail_lr_rad'],
        input_window_width=10,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

## BESTs

def Elv_Simp_1():
    sequential_hidden_l = [20]# [4, 16]

    sdense = tf.keras.Sequential([
        # tf.keras.layers.GRU(10),
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            #'alpha', 'throttle', 'gs', 'gs_d', 'gs_i', 'gamma_err', 'hdot', 'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
            # 'ias_err', 'alpha', 'gs', 'gs_d', 'gs_i', 'gamma_err', 'hdot', 'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
            'alpha', 'gs', 'gs_d', 'gs_i', 'gamma_err', 'hdot', 'Q',
            'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
        ],
        seq_labels=['ctrl_col'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Elv_Simp_10():
    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'gs_dev_ddm', 'theta_rad'],
        seq_labels=['ctrlcolumn_pos_capt'],
        input_window_width=10,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Ail_Simp_1():
    sequential_hidden_l = [20]# [10] # [4, 16]

    sdense = tf.keras.Sequential([
        # tf.keras.layers.GRU(10),
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)
    #model = sdense

    return ModelDefBuilder(
        feature_columns=[
            #'loc', 'loc_i', 'loc_d', 'phi', 'ctrl_col', 'ctrl_rud', 'ias', 'hralt',
            'phi', 'phi_i', 'P', 'loc', 'loc_i', 'loc_d', 'psi'
            # 'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
            ],
        seq_labels=['ctrl_whl'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Thr_Simp_1():
    sequential_hidden_l = [20]# [4, 16]

    sdense = tf.keras.Sequential([
        # tf.keras.layers.GRU(5),
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'ias_err', 'gs', 'gs_i', 'gs_d', 'gamma_err',
            'flap_0_bool', 'flap_1_bool', 'flap_2_bool', 'flap_3_bool', 'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
            ],
        seq_labels=['throttle'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Sas_Simp_1():
    sequential_hidden_l = [4]# [4, 16]

    sdense = tf.keras.Sequential([
        # tf.keras.layers.GRU(20),
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'hralt'],
        seq_labels=['sas'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Flap_Simp_1():
    sequential_hidden_l = [4]# [4, 16]

    sdense = tf.keras.Sequential([
        # tf.keras.layers.GRU(20),
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'ias'],# 'ias'],
        seq_labels=['flap_rat'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )






def Best_Dense_Min_1():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            # 'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gamma_error_rad'],
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'tas_mps', 'gs_dev_ddm'],
        # seq_labels=['elv_l_rad', 'N1s_rpm'],
        seq_labels=['ctrlcolumn_pos_capt', 'pla_mean_rad'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Best_Dense_Min_10():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            # 'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gamma_error_rad'],
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'tas_mps', 'gs_dev_ddm'],
        # seq_labels=['elv_l_rad', 'N1s_rpm'],
        seq_labels=['ctrlcolumn_pos_capt', 'pla_mean_rad'],
        input_window_width=10,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Best_Dense_Min_20():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gamma_error_rad'],
        seq_labels=['elv_l_rad', 'N1s_rpm'],
        input_window_width=20,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )


def Best_Dense_Comp_1():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gs_dev_ddm',
            'tailwind_mps'], # 'g_err_d_rad'
        # seq_labels=['elv_l_rad', 'N1s_rpm'],
        seq_labels=['ctrlcolumn_pos_capt', 'pla_mean_rad'],
        input_window_width=1,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Best_Dense_Comp_10():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gs_dev_ddm',
            'tailwind_mps'], # 'g_err_d_rad'
        # seq_labels=['elv_l_rad', 'N1s_rpm'],
        seq_labels=['ctrlcolumn_pos_capt', 'pla_mean_rad'],
        input_window_width=10,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Best_Dense_Comp_20():

    sequential_hidden_l = [4, 16]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 'gamma_error_rad',
            'tailwind_mps', 'g_err_d_rad'],
        seq_labels=['elv_l_rad', 'N1s_rpm'],
        input_window_width=20,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )


def Best_Ail_1():
    sequential_hidden_l = [2, 30, 26]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)
    model.add(tf.keras.layers.Dropout(0.5))

    return ModelDefBuilder(
        feature_columns=[
            'phi_rad'],
        seq_labels=['ail_lr_rad'],
        input_window_width=20,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

def Best_Ail_10():
    sequential_hidden_l = [12, 30]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dropout(0.3)
    ])

    model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        feature_columns=[
            'phi_rad'],
        seq_labels=['ail_lr_rad'],
        input_window_width=20,
        label_window_width=1,
        label_shift=0,

        sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )



def MinimalBatchNorm_Dense():

    sequential_hidden_l = [22, 2]

    sdense = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(22, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='relu')
    ])

    # model = AddDenseHiddenLayer(sdense, sequential_hidden_l)

    return ModelDefBuilder(
        input_window_width=10,
        label_window_width=1,
        label_shift=0,

        #sequential_hidden_l=sequential_hidden_l,

        model=sdense
    )

## Time-Series Baseline
def ForecastBase():

    base = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])

    return ModelDefBuilder(
        input_window_width=1,
        label_window_width=1,
        label_shift=1,
        feature_columns    = ['elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = False,

        model=base
    )

## Simple Custom Dense HiddenLayer
def Simple_CustomHiddenLayer():
    
    sequential_hidden_l = [30, 30]
    
    dense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])
    dense = AddDenseHiddenLayer(dense, sequential_hidden_l)

    return ModelDefBuilder(
        input_window_width  = 1,
        label_window_width  = 1,
        label_shift         = 5,
        # feature_columns    = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps", 'elv_l_rad', 'N1s_rpm'],
        feature_columns    = ['elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = False,

        model=dense
    )

## Wrapped Custom Dense HiddenLayer
def Wrap_CustomHiddenLayer():
    
    sequential_hidden_l = [30, 30]
    
    dense = tf.keras.Sequential([
        tf.keras.layers.Flatten()
    ])
    dense = AddDenseHiddenLayer(dense, sequential_hidden_l)

    return ModelDefBuilder(
        input_window_width  = 5,
        label_window_width  = 1,
        label_shift         = 1,
        # feature_columns    = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps", 'elv_l_rad', 'N1s_rpm'],
        feature_columns    = ['elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = True,

        model=dense
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

    tf.keras.backend.clear_session()
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
        feature_columns    = ['hralt_m', 'theta_rad', 'cas_mps', 'gs_dev_ddm', 'hdot_1_mps', 'elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = False,

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
        feature_columns    = ['hralt_m', 'theta_rad', 'cas_mps', 'gs_dev_ddm', 'hdot_1_mps', 'elv_l_rad', 'N1s_rpm'],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = True,

        model=lstm
    )

#
# MODEL_DEFINITIONS dictionary
MODEL_DEFINITIONS = {
    'default'       : Linear,
    'linear'        : Linear,
    'simp_dense'    : Simple_Dense,
    'simp_dense_ail': Simple_Dense_Ail,
    'baseline'      : ForecastBase,
    'conv_simple'   : Conv,
    'conv_hidden'   : Conv_CustomHiddenLayer,
    'lstm_hidden'   : LSTM_CustomHiddenLayer,
    'simp_hidden'   : Simple_CustomHiddenLayer,
    'wrap_hidden'   : Wrap_CustomHiddenLayer,
    'minim_batchnorm': MinimalBatchNorm_Dense,

    'best_e_min_1'  : Best_Dense_Min_1,
    'best_e_min_10' : Best_Dense_Min_10,
    'best_e_min_20' : Best_Dense_Min_20,

    'best_e_comp_1' : Best_Dense_Comp_1,
    'best_e_comp_10': Best_Dense_Comp_10,
    'best_e_comp_20': Best_Dense_Comp_20,

    'e_simp_1'      : Elv_Simp_1,
    'e_simp_10'     : Elv_Simp_10,
    'a_simp_1'      : Ail_Simp_1,
    't_simp_1'      : Thr_Simp_1,

    's_simp_1'      : Sas_Simp_1,
    'f_simp_1'      : Flap_Simp_1,

    'best_a_1'      : Best_Ail_1,
    'best_a_10'     : Best_Ail_10,

}
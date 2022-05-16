from math import pi
from tensorflow import keras

from hpcscripts.trainers.modeldefinitions import Conv_CustomHiddenLayer, Linear, Conv
import hpcscripts.trainers.modeldefinitions as mod_def

def ApplyModelDefinition(model_definition):
    """Set model inside model_definition as current model and
    apply its parameter to current settings."""
    global MODEL, INPUT_WINDOW_WIDTH, LABEL_WINDOW_WIDTH, LABEL_SHIFT, SEQUENTIAL_HIDDENLAYERS

    _param, MODEL = model_definition()
    _input_window_width, _label_window_width, _label_shift, _seq_hl = _param

    if not _param == None:
        INPUT_WINDOW_WIDTH = _input_window_width
        LABEL_WINDOW_WIDTH = _label_window_width
        LABEL_SHIFT = _label_shift

        if not _seq_hl == None:
            SEQUENTIAL_HIDDENLAYERS = _seq_hl





# V V -  -  -  -  -  -  -  -  -  -  V V
# V V  GLOBAL PARAMETERS SETTINGS   V V
# V V -  -  -  -  -  -  -  -  -  -  V V

# DEFINITION
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# NEURAL NETWORK
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
METRICS = [
            keras.metrics.MeanSquaredError(), 
            keras.metrics.MeanAbsoluteError()
        ]
FEATURE_COLUMNS = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps"]
SEQUENTIAL_LABELS = ['elv_l_rad']
CALLBACKS = [early_stop]
TRAIN_EPOCHS = 20

# PROCESSING 
DATAPROCESSING_POOL = 4

# WINDOW GENERATOR, DEFAULT (CAN BE OVERWRITTEN BY MODEL DEFINITION)
INPUT_WINDOW_WIDTH = 1
LABEL_WINDOW_WIDTH = 1
LABEL_SHIFT = 0

SEQUENTIAL_HIDDENLAYERS = [50, 30]
USE_RESIDUAL_WRAPPER = True

# MODEL DEFINITION
# See modeldefinitions.py in trainers folder
_, MODEL = mod_def.DefaultModelDefinition()
ApplyModelDefinition(mod_def.DefaultModelDefinition)

# CLEANING AND RESAMPLING
TARGET_GAMMA = -3 * pi / 180
TARGET_FREQUENCY = 1 # Hz    
CUTOUT_DISTANCE  = 15.39  #10.5 #18.52 # Km, ~ 10 Nm

SAVEUNCUTCLEANED = False
SKIP_FILTER_PROCESS = False

SELECTED_COLUMNS = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
                "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
                "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "pla_1_rad", "pla_2_rad", "pla_3_rad", "pla_4_rad", "ctrlcolumn_pos_capt", "ctrlcolumn_pos_fo", "ctrlcolumn_pos_fo", "ctrlwheel_pos_fo",
                "gs_dev_ddm", "loc_dev_ddm", "lon_rad", "lat_rad", "phi_rad", "gmt_hr", "gmt_min", "gmt_s", "drift_rad", "temp_static_deg", "temp_total_degC",
                "ws_mps", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2", "chi_rad", "psi_rad", "psi_selected_rad"]

COLUMN_LIST = ["time_s", "hbaro_m", "hralt_m", "hselected_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "tas_mps",
                "gs_mps", "airspd_selected_mps", "elv_l_rad", "elv_r_rad", "hdot_1_mps", "hdot_2_mps", "hdot_selected_mps", "flap_te_pos", 
                "n11_rpm", "n12_rpm", "n13_rpm", "n14_rpm", "n1_cmd_rpm", "pla_1_rad", "pla_2_rad", "pla_3_rad", "pla_4_rad", "ctrlcolumn_pos_capt", "ctrlcolumn_pos_fo", "ctrlcolumn_pos_fo", "ctrlwheel_pos_fo",
                "gs_dev_ddm", "loc_dev_ddm", "lon_rad", "lat_rad", "phi_rad", "gmt_hr", "gmt_min", "gmt_s", "drift_rad", "temp_static_deg", "temp_total_degC",
                "ws_mps", "wdir_rad", "rud_rad", "rud_pedal_pos", "ail_l_rad", "ail_r_rad", "ax_mps2", "ay_mps2", "az_mps2", "chi_rad", "psi_rad", "psi_selected_rad"]#,
                #"smooth_hdot_mps", "delta_smooth_hdot_mps", "dist_to_land"]

FILTER_LIST = ["hbaro_m", "hralt_m", "aoac_rad", "theta_rad", "theta_trim_rad", "cas_mps", "elv_l_rad", "elv_r_rad", "flap_te_pos", "hdot_2_mps", "gs_dev_ddm", "dist_to_land"]
FILTER_LIST = [] 

LEFT_ALONE_LIST = ["time_s", "flap_te_pos", "cas_mps", "theta_rad", "aoac_rad"]


# OPERATIONAL
DATA_DIRECORIES = [
        "Data",
        "Data/Models",
        "Data/Raw",
        "Data/Results",
        "Data/Processed",
        "Data/Processed/Sampled",
        "Data/Processed/Approach",
        "Data/Processed/Cleaned",
        "Data/Processed/Selected",
        "Data/Processed/Ready",
        "Data/Processed/Combined",
        "Data/Processed/Train",
        "Data/Processed/Test",
        "Data/Processed/Eval"
    ]

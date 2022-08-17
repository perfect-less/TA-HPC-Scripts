
import tensorflow as tf
import tensorflow.keras as keras

from hpcscripts.trainers.modeldefinitions import ModelDefBuilder, ResidualWrapper
from hpcscripts.option import globalparams as G_PARAMS

FEATURES_COLUMNS = [
            'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 
            'gamma_error_rad', 'tailwind_mps', 'g_err_d_rad',
            'flap_4_bool', 'flap_5_bool', 'flap_6_bool',
        ]
LABELS = ['elv_l_rad', 'N1s_rpm']
INPUT_WINDOW_WIDTH = 10


def minimal_tuner():
    
    input_window_width = INPUT_WINDOW_WIDTH


    def model_builder(hp):
        model = keras.Sequential()
        model.add(tf.keras.layers.Flatten())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=2, max_value=40, step=2),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.8, step=0.1))
                )


        # Add output layer
        model.add(keras.layers.Dense(
                        units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                        kernel_initializer=tf.initializers.zeros()
                ))
        model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

        # Compile our model
        model.compile(
            optimizer=G_PARAMS.OPTIMIZER,
            loss=G_PARAMS.LOSS,
            metrics=G_PARAMS.METRICS
        )        
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = FEATURES_COLUMNS,
        seq_labels         = LABELS,
        use_residual_wrap  = False,

        model=model_builder
    )


def lstm_tuner():
    
    input_window_width = INPUT_WINDOW_WIDTH

    def model_builder(hp):
        model = keras.Sequential()

        lstm_layers = hp.Int("lstm_layers", 1, 2)
        is_bidirectional = hp.Boolean("is_bidirectional")

        if lstm_layers > 0:
            for i in range (lstm_layers):
                ret_seq = not (i == (lstm_layers - 1))
                if is_bidirectional:
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                hp.Int(f"lstm_units_{i}", min_value=2, max_value=32, step=2),
                                return_sequences=ret_seq
                            )
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.LSTM(
                            hp.Int(f"lstm_units_{i}", min_value=2, max_value=32, step=2),
                            return_sequences=ret_seq
                        )
                    )
        else:
            model.add(tf.keras.layers.Flatten())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 0, 2)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=2, max_value=40, step=2),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.8, step=0.1))
                )

        # Add output layer
        model.add(keras.layers.Dense(
                        units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                        kernel_initializer=tf.initializers.zeros()
                ))
        model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

        # # Apply Residual Wrapper
        # if G_PARAMS.USE_RESIDUAL_WRAPPER:
        #     model = ResidualWrapper(model)

        # Compile our model
        model.compile(
            optimizer=G_PARAMS.OPTIMIZER,
            loss=G_PARAMS.LOSS,
            metrics=G_PARAMS.METRICS
        )
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = FEATURES_COLUMNS,
        seq_labels         = LABELS,
        use_residual_wrap  = False,

        model=model_builder
    )


def gru_tuner():
    
    input_window_width = INPUT_WINDOW_WIDTH

    def model_builder(hp):
        model = keras.Sequential()

        lstm_layers = hp.Int("gru_layers", 1, 2)
        is_bidirectional = hp.Boolean("is_bidirectional")

        if lstm_layers > 0:
            for i in range (lstm_layers):
                ret_seq = not (i == (lstm_layers - 1))
                if is_bidirectional:
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(
                                hp.Int(f"gru_units_{i}", min_value=2, max_value=32, step=2),
                                return_sequences=ret_seq
                            )
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.GRU(
                            hp.Int(f"gru_units_{i}", min_value=2, max_value=32, step=2),
                            return_sequences=ret_seq
                        )
                    )
        else:
            model.add(tf.keras.layers.Flatten())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 0, 2)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=2, max_value=40, step=2),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.8, step=0.1))
                )

        # Add output layer
        model.add(keras.layers.Dense(
                        units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                        kernel_initializer=tf.initializers.zeros()
                ))
        model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

        # # Apply Residual Wrapper
        # if G_PARAMS.USE_RESIDUAL_WRAPPER:
        #     model = ResidualWrapper(model)

        # Compile our model
        model.compile(
            optimizer=G_PARAMS.OPTIMIZER,
            loss=G_PARAMS.LOSS,
            metrics=G_PARAMS.METRICS
        )
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = FEATURES_COLUMNS,
        seq_labels         = LABELS,
        use_residual_wrap  = False,

        model=model_builder
    )


def conv_tuner():
    
    input_window_width = INPUT_WINDOW_WIDTH

    def model_builder(hp):
        model = keras.Sequential()

        conv_layers = hp.Int("conv_layers", 1, 2)
        padd = 'valid'

        if conv_layers > 0:
            for i in range (conv_layers):
                ret_seq = not (i == (conv_layers - 1))
                if ret_seq:
                    padd = 'same'

                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(f"conv_filters_{i}", min_value=2, max_value=32, step=2),
                        kernel_size=hp.Int(f"conv_kernel_{i}", min_value=3, max_value=input_window_width, step=1),
                        activation='relu',
                        padding=padd  
                    )
                )
                if not ret_seq:
                    model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.Flatten())
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 0, 2)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=2, max_value=40, step=2),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.8, step=0.1))
                )

        # Add output layer
        model.add(keras.layers.Dense(
                        units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                        kernel_initializer=tf.initializers.zeros()
                ))
        model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

        # Compile our model
        model.compile(
            optimizer=G_PARAMS.OPTIMIZER,
            loss=G_PARAMS.LOSS,
            metrics=G_PARAMS.METRICS
        )        
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = FEATURES_COLUMNS,
        seq_labels         = LABELS,
        use_residual_wrap  = False,

        model=model_builder
    )


def mixed_tuner():
    
    input_window_width = INPUT_WINDOW_WIDTH

    def model_builder(hp):
        model = keras.Sequential()

        early_layers = hp.Int(f"early_layers", 1, 2)
        padd = 'valid'

        if early_layers > 0:
            for i in range (early_layers):
                kind = hp.Choice(f"early_layer_{i}", ["lstm", "conv1d"])
                ret_seq = not (i == (early_layers - 1))
                if ret_seq:
                    padd = 'same'

                if kind == 'conv1d':
                    model.add(
                        tf.keras.layers.Conv1D(
                            hp.Int(f"conv_filters_{i}", min_value=2, max_value=64, step=2),
                            kernel_size=hp.Int(f"conv_kernel_{i}", min_value=3, max_value=input_window_width, step=1),
                            activation='relu',
                            padding=padd             
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                hp.Int(f"lstm_units_{i}", min_value=2, max_value=64, step=2),
                                return_sequences=ret_seq
                            )
                        )
                    )

                if not ret_seq and kind == 'conv1d':
                    model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.Flatten())
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 0, 2)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=2, max_value=100, step=2),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(
                    keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.8, step=0.1))
                )

        # Add output layer
        model.add(keras.layers.Dense(
                        units=len(G_PARAMS.SEQUENTIAL_LABELS)*G_PARAMS.LABEL_WINDOW_WIDTH,
                        kernel_initializer=tf.initializers.zeros()
                ))
        model.add(keras.layers.Reshape([G_PARAMS.LABEL_WINDOW_WIDTH, len (G_PARAMS.SEQUENTIAL_LABELS)]))

        # Compile our model
        model.compile(
            optimizer=G_PARAMS.OPTIMIZER,
            loss=G_PARAMS.LOSS,
            metrics=G_PARAMS.METRICS
        )        
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = FEATURES_COLUMNS,
        seq_labels         = LABELS,
        use_residual_wrap  = False,

        model=model_builder
    )


TUNER_DEFINITIONS = {
    'default' : minimal_tuner,
    'minimal' : minimal_tuner,
    'gru'     : gru_tuner,
    'lstm'    : lstm_tuner,
    'conv'    : conv_tuner,
    'mixed'   : mixed_tuner,
}

import tensorflow as tf
import tensorflow.keras as keras

from hpcscripts.trainers.modeldefinitions import ModelDefBuilder, ResidualWrapper
from hpcscripts.option import globalparams as G_PARAMS

def default_tuner():
    
    input_window_width = 10

    def model_builder(hp):
        model = keras.Sequential()

        lstm_layers = hp.Int("lstm_layers", 0, 2)
        is_bidirectional = hp.Boolean("is_bidirectional")

        if lstm_layers > 0:
            for i in range (lstm_layers):
                ret_seq = not (i == (lstm_layers - 1))
                if is_bidirectional:
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                hp.Int(f"lstm_units_{i}", min_value=16, max_value=128, step=16),
                                return_sequences=ret_seq
                            )
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.LSTM(
                            hp.Int(f"lstm_units_{i}", min_value=16, max_value=128, step=16),
                            return_sequences=ret_seq
                        )
                    )
        else:
            model.add(tf.keras.layers.Flatten())
        
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 2)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=20, max_value=20, step=100),
                    activation= 'relu' # hp.Choice("activation", ["relu", "tanh"]),
                )
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
        # if hp.Boolean("dropout"):
        #     model.add(keras.layers.Dropout(rate=0.25))

        # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        # model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        #     loss="categorical_crossentropy",
        #     metrics=["accuracy"],
        # )
        return model

    return ModelDefBuilder(
        input_window_width  = input_window_width,
        label_window_width  = 1,
        label_shift         = 1,
        feature_columns    = [
                                'hralt_m', 'hdot_1_mps', 'theta_rad', 'cas_mps', 
                                'gamma_error_rad', 'tailwind_mps', # 'crosswind_mps'
                                'flap_4_bool', 'flap_5_bool', 'flap_6_bool'
                            ],
        seq_labels         = ['elv_l_rad', 'N1s_rpm'],
        use_residual_wrap  = False,

        model=model_builder
    )



TUNER_DEFINITIONS = {
    'default' : default_tuner,
}
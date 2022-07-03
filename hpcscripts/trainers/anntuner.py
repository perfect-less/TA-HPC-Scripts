
import keras_tuner as kt
import tensorflow.keras as keras

from hpcscripts.sharedutils.trainingutils import *
from hpcscripts.trainers.tunerdefinitions import TUNER_DEFINITIONS
from hpcscripts.option.pathhandler import InitTuningSubdir
from hpcscripts.postprocesses import rsquared
from hpcscripts import tuning as TUNING_CONF

def run(tuner_id: str = None):

    # Apply tunner definition
    tf.keras.backend.clear_session()
    if tuner_id != None:
        if not tuner_id in TUNER_DEFINITIONS:
            print ("Err: invalid tuner id -> {}".format(tuner_id))
            exit (1)

        G_PARAMS.ApplyModelDefinition(
            TUNER_DEFINITIONS[tuner_id]
        )
    else:
        G_PARAMS.ApplyModelDefinition(
            TUNER_DEFINITIONS['default']
        ) 

    # Initialize tuning folder
    tuning_dir = InitTuningSubdir()

    # Import Data
    train_comb= ImportCombinedTrainingData()

    # Pre-process Data
    train_comb, norm_param = DF_Nomalize(train_comb)
    train_list, test_list, eval_list = GetFileList()

    # Create WindowGenerator
    windowG = CreateWindowGenerator(train_list, 
                    test_list, eval_list, norm_param)

    # Data is ready, time to create our oracle
    _, model_builder = TUNER_DEFINITIONS[tuner_id]()

    print ("Data is ready, initializing tuner")

    # Iniializing tuner
    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective="val_mean_absolute_error",
        max_trials=TUNING_CONF.MAX_TRIALS,
        executions_per_trial=2,
        overwrite=True,
        directory="_tuning",
        project_name="ann_tuning",
    )

    # Search for optimal model
    tuner.search(
        windowG.train, 
        epochs=G_PARAMS.TRAIN_EPOCHS,
        validation_data=windowG.val,
        callbacks=G_PARAMS.CALLBACKS
    )

    # Processing searched models
    best_hps_list = tuner.get_best_hyperparameters(num_trials=TUNING_CONF.PICK_MODEL_COUNTS)

    for i in range( min(TUNING_CONF.PICK_MODEL_COUNTS, len(best_hps_list)) ):
        tf.keras.backend.clear_session()
        train_data = windowG.train
        val_data = windowG.val

        model = tuner.hypermodel.build(best_hps_list[i])
        history = TrainModel(
            model=model,
            training_data=train_data,
            eval_data=val_data,
            epochs=G_PARAMS.TRAIN_EPOCHS,
            callbacks=G_PARAMS.CALLBACKS
        )

        val_acc_per_epoch = history.history['val_mean_absolute_error']
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
        print('Best epoch for i->{} : {}'.format(i, best_epoch))
        print('Retrain to best epoch')

        # Retrain the model
        hypermodel = tuner.hypermodel.build(best_hps_list[i])
        history = TrainModel(
            model=hypermodel,
            training_data=train_data,
            eval_data=val_data,
            epochs=best_epoch,
            callbacks=G_PARAMS.CALLBACKS
        )

        # Save Model
        SaveModel(
            hypermodel, history, 
            'tune_{}_{}'.format(tuner_id, i),
            optional_path=tuning_dir
        )

        # Run post process on model
        rsquared.run(99999, tuning_dir)
        

    # Done
    print ('\nTuning done, saved to -> {}'.format(tuning_dir))








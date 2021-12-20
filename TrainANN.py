## Importing important lib
import tensorflow as tf
from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras import datasets, layers, models, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping

import math
import numpy as np
import pandas as pd
import DataProcessMethods as dpm


## Save Model Location
model_dir = 'SavedModels/ann_2021_1220_2030'

## Importing the Datas (train and test datas)

F_Id = "uc"
D_Id = "_cut_UC"
file_path = "Processed Data{}/Ready/".format(D_Id)

tr_data_name = "{}Train_set_{}.csv".format(file_path, F_Id)
te_data_name = "{}Test_set_{}.csv".format(file_path, F_Id)
#ev_data_name = "Processed Data/eval_data_uc_0.csv"

train_data = pd.read_csv(tr_data_name) # _c50
test_data = pd.read_csv(te_data_name)
#eval_data = pd.read_csv(ev_data_name)

train_data, norm_param = dpm.DF_Nomalize(train_data)

test_data = dpm.DF_Nomalize(test_data, norm_param)
#eval_data = DF_Nomalize(eval_data, norm_param)

# Defining our feature columns
# In this case our feature are theta, altitude, and speed
# or as theta_rad, hbaro_m, cas_mps
feature_columns = []

# Represent altitude (hralt) as a floating-point value.
hralt_m = tf.feature_column.numeric_column("hralt_m")
feature_columns.append(hralt_m)

# Represent theta_rad as a floating-point value.
theta_rad = tf.feature_column.numeric_column("theta_rad")
feature_columns.append(theta_rad)

theta_del_rad = tf.feature_column.numeric_column("theta_del_rad")
feature_columns.append(theta_del_rad)

# Represent aoac_rad as a floating-point value.
aoac_rad = tf.feature_column.numeric_column("aoac_rad")
feature_columns.append(aoac_rad)

# Represent cas_mps as a floating-point value.
cas_mps = tf.feature_column.numeric_column("cas_mps")
feature_columns.append(cas_mps)

# Represent flap_te_pos as a floating-point value.
hdot_2_mps = tf.feature_column.numeric_column("hdot_2_mps")
feature_columns.append(hdot_2_mps)

# Represent gs_dev_ddm as a floating-point value.
# gamma_error_rad = tf.feature_column.numeric_column("gamma_error_rad") #gs_dev_ddm gamma_error_rad
# feature_columns.append(gamma_error_rad)

# Represent flap_te_pos as a bucketized floating-point value.
flap_te_pos_num = tf.feature_column.numeric_column("flap_te_pos")
flap_te_pos = tf.feature_column.bucketized_column(flap_te_pos_num, [5, 23, 27.5, 34])
feature_columns.append(flap_te_pos)

feature_layer = keras.layers.DenseFeatures(feature_columns)


## Creating Model

# Reset the model
model = None

# Create a sequential model
model = keras.models.Sequential()

# Create input layer
model.add(feature_layer)

# Create the first hidden layer
model.add(keras.layers.Dense(units=50, activation='relu')) # 30

# Create the second hidden layer
model.add(keras.layers.Dense(units=30, activation='relu'))

# Create the third hidden layer
#model.add(tf.keras.layers.Dense(units=30, activation='relu'))

# Create the third hidden layer
#model.add(tf.keras.layers.Dense(units=350, activation='relu'))

# Create output layer
model.add(keras.layers.Dense(units=1))

# Compile our model
model.compile(optimizer=keras.optimizers.Adam(name='Adam'),
              loss="mean_squared_error",
              metrics=[keras.metrics.MeanSquaredError()]
             )


# Eval Features and Labels
test_features = {name:np.array(value) for name, value in test_data.items()}
#test_labels = np.array(test_features.pop('N1s_rpm'))
test_labels = np.array(test_features.pop('elv_l_rad'))
#test_labels = np.stack((test_labels, np.array(test_features.pop('N1s_rpm'))))

#test_labels = np.stack((test_labels, np.array(test_features.pop('N1s_rpm')), np.array(test_features.pop('flap_te_pos'))))
test_labels = np.transpose(test_labels)


## Train our Model

train_epochs = 50
label_name = ['elv_l_rad'] #['elv_l_rad', 'N1s_rpm'] #['N1s_rpm']
#label_name = ['elv_l_rad', 'N1s_rpm' , 'flap_te_pos']

features = {name:np.array(value) for name, value in train_data.items()}
#label = np.array(features.pop('N1s_rpm'))
label = np.array(features.pop('elv_l_rad'))
#label = np.stack((label, np.array(features.pop('N1s_rpm'))))

#label = np.stack((label, np.array(features.pop('N1s_rpm')), np.array(features.pop('flap_te_pos'))))
label = np.transpose(label)

early_stop = EarlyStopping(monitor='val_loss',patience=3)

# Training
history = model.fit(features, label, epochs=train_epochs, validation_data=(test_features,test_labels),callbacks=[early_stop])

epochs = history.epoch
hist = pd.DataFrame(history.history)
acc = hist["mean_squared_error"]

hist_list.append(history)
plot_the_curve(epochs, acc)



## Evaluating model agains test data

test_loss, test_acc = model.evaluate(test_features,  test_labels)

print('\nTest squared error:', test_acc)




# Save The Entire Model

model.save(model_dir)

print ("Model Saved")


# Definition for WindowGenerator are written based on tensorflow official
# tutorial on:
#        https://www.tensorflow.org/tutorials/structured_data/time_series
# which licensed under Apache License, Version 2.0



# Copyright 2019 The TensorFlow Authors

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

from hpcscripts.sharedutils.nomalization import DF_Nomalize, denorm
from hpcscripts.option import globalparams as G_PARAMS

class WindowGenerator():

    USABLE_COLUMNS  = set (G_PARAMS.FEATURE_COLUMNS + G_PARAMS.SEQUENTIAL_LABELS)
    FEATURE_COLUMNS = G_PARAMS.FEATURE_COLUMNS

    def __init__(self, input_width:int, label_width:int=1, shift:int=1,
                train_list=None, test_list=None, val_list=None,
                label_columns=None,
                norm_param:dict=None,
                shuffle_train:bool=True,
                print_check:bool=True):
        # Store list of the data.
        self.train_list = train_list
        self.test_list = test_list
        self.val_list = val_list

        # Set params
        self.shuffle_train = shuffle_train
        self.norm_param = norm_param

        # Work out the label column indices.
        self.input_columns = self.FEATURE_COLUMNS
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = self.__get_column_indices()

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Check if there's missing parameters
        if print_check:
            self.check_params()

    def __get_column_indices(self):
        # Get 1 Sample DataFrame and use it to determine column indices
        df = pd.read_csv(self.train_list[0])
        return {name: i for i, name in enumerate(df.columns) if name in self.USABLE_COLUMNS}

    def check_params(self):
        print ("train_list \t {}".format("Found" if self.train_list != None else "Missing"))
        print ("test_list \t {}".format("Found" if self.test_list != None else "Missing"))
        print ("val_list \t {}".format("Found" if self.val_list != None else "Missing"))
        print ("label_columns \t {}".format("Found" if self.label_columns != None else "Missing"))
        print ("norm_param \t {}".format("Found" if self.norm_param != None else "Missing"))

        print ("""\nMissing norm_columns will cause data not to be normalized when creating datasets.""")


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.input_columns is not None:
            inputs = tf.stack(
                            [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                            axis=-1
                    )
        if self.label_columns is not None:
            labels = tf.stack(
                            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                            axis=-1
                    )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, file_list: List[str], do_shuffle:bool = False, batch_size:int=32):
        np_data = []
        for file in file_list:
            try:
                data = pd.read_csv(file)
                #
                # PREPROCESS HERE, 
                if not self.norm_param == None:
                    data = DF_Nomalize(data, self.norm_param)
                # NORMALIZATION, ETC.
            except:
                raise Exception("Can't process {}".format(file))

            ds = timeseries_dataset_from_array(
                        data=data.to_numpy(),
                        targets=None,
                        sequence_length=self.total_window_size,
                        sequence_stride=1,
                        batch_size=99999
                    )
            
            for elem in ds.take(1):
                np_data.append(
                        elem.numpy()
                    )
        
        concated_np = np.concatenate(np_data)
        buffer_size = concated_np.shape[0]

        if do_shuffle:
            ds = tf.data.Dataset.from_tensor_slices(concated_np).shuffle(buffer_size)
        else:
            ds = tf.data.Dataset.from_tensor_slices(concated_np)

        _batch_size = buffer_size
        if not batch_size == None:
            _batch_size = batch_size
            
        ds = ds.batch(_batch_size, drop_remainder=True)
        ds = ds.map(self.split_window)

        return ds

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    @property
    def train(self):
        return self.make_dataset(self.train_list, self.shuffle_train)

    @property
    def val(self):
        return self.make_dataset(self.val_list)

    @property
    def test(self):
        return self.make_dataset(self.test_list)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
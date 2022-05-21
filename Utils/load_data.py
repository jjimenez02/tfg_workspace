import os
import pandas as pd
import numpy as np
from utils.data_extraction import Data
from scipy.io import arff
from sktime.datatypes._panel._convert import (from_pd_wide_to_nested,
                                              from_nested_to_multi_index)
from sktime.datasets import load_from_tsfile_to_dataframe
from itertools import zip_longest


def import_heartbeat_dataset(heartbeat_path: str):
    train = None
    test = None

    # Dimensions 6, 7, 8, & 9 not included (not in Dataset)
    dimensions = list(range(1, 5+1))
    dimensions.extend(list(range(10, 60+1)))

    for i in dimensions:
        actual_dim_train_path = heartbeat_path + \
            '/HeartbeatDimension' + str(i) + '_TRAIN.arff'
        actual_dim_test_path = heartbeat_path + \
            '/HeartbeatDimension' + str(i) + '_TEST.arff'

        actual_dim_X_train, actual_dim_y_train = __extract_heartbeat_dimension(
            actual_dim_train_path, str(i))
        actual_dim_X_test, actual_dim_y_test = __extract_heartbeat_dimension(
            actual_dim_test_path, str(i))

        if train is None or test is None:
            train = actual_dim_X_train
            y_train = actual_dim_y_train.apply(lambda x: x.decode())

            test = actual_dim_X_test
            y_test = actual_dim_y_test.apply(lambda x: x.decode())
            continue

        train = pd.merge(train, actual_dim_X_train,
                         left_on=['id', 'TimeStamp'],
                         right_on=['id', 'TimeStamp'])
        test = pd.merge(test, actual_dim_X_test,
                        left_on=['id', 'TimeStamp'],
                        right_on=['id', 'TimeStamp'])

    y_train = __from_nested_to_long_class(train, y_train)
    y_test = __from_nested_to_long_class(test, y_test)

    train.insert(train.shape[1], 'class', y_train)
    test.insert(test.shape[1], 'class', y_test)

    return Data(train), Data(test)


def import_epilepsy_dataset(epilepsy_path: str):
    train_df, y_train = load_from_tsfile_to_dataframe(
        epilepsy_path + "/Epilepsy_TRAIN.ts")
    test_df, y_test = load_from_tsfile_to_dataframe(
        epilepsy_path + "/Epilepsy_TEST.ts")

    train_df = from_nested_to_multi_index(train_df).reset_index(level=[0, 1])
    test_df = from_nested_to_multi_index(test_df).reset_index(level=[0, 1])

    train_df = train_df.rename(
        {'instance': 'id', 'timepoints': 'TimeStamp'}, axis=1)
    test_df = test_df.rename(
        {'instance': 'id', 'timepoints': 'TimeStamp'}, axis=1)

    y_train = __from_nested_to_long_class(train_df, y_train)
    y_test = __from_nested_to_long_class(test_df, y_test)

    train_df.insert(train_df.shape[1], "class", y_train)
    test_df.insert(test_df.shape[1], "class", y_test)

    return Data(train_df), Data(test_df)


def import_seguimiento_ocular_dataset(
        data_path,
        folders_id,
        file_prefix='data_',
        file_ext='.tsv'):
    '''
    It will extract data from `data_path`'s folders
    explicited in `folders_id`'s parameter.
    '''
    all_data = pd.DataFrame()

    for folder_id in folders_id:
        relative_path = data_path + '/' + str(folder_id) + '/'
        n_files = len(os.listdir(relative_path))
        for i in range(1, n_files):
            file_data = __seguimiento_ocular_parse_file(
                relative_path, file_prefix, folder_id, i, file_ext)
            all_data = all_data.append(file_data, ignore_index=True)

    return Data(all_data)


def __from_nested_to_long_class(X, y, id_col_name='id'):
    '''
    This method will extend all "y"'s vector values for each instance
    of "X"'s DataFrame.
    '''
    new_y = []

    for serie_id, serie_class in zip_longest(pd.unique(X[id_col_name]), y):
        new_y.extend([serie_class for _ in range(
            X[X[id_col_name] == serie_id].shape[0])])

    return np.asarray(new_y)


def __extract_heartbeat_dimension(path: str, dim_name: str):
    actual_dim = pd.DataFrame(arff.loadarff(path)[0])
    y = actual_dim.pop('target')
    actual_dim = from_nested_to_multi_index(from_pd_wide_to_nested(
        actual_dim)).reset_index(level=[0, 1])
    actual_dim = actual_dim.rename(
        {'instance': 'id',
         'timepoints': 'TimeStamp',
         0: 'signal_' + dim_name}, axis=1)

    return actual_dim, y


def __seguimiento_ocular_parse_file(
        relative_path,
        file_prefix,
        folder_id,
        file_id,
        file_ext):
    serie_file = relative_path + file_prefix + str(file_id) + file_ext
    serie_data = pd.read_csv(serie_file, delimiter='\t')

    # Hard-coded into Seguimiento Ocular's specific case
    with open(relative_path + 'class.txt') as class_file:
        class_data = class_file.readline()
    serie_class, serie_diagnosis, age = class_data.split('-')

    serie_data.insert(0, 'id', str(folder_id) + '-' + str(file_id))
    serie_data.insert(2, 'Age', age)
    serie_data.insert(serie_data.shape[1], 'class', serie_class)
    serie_data.insert(serie_data.shape[1], 'diagnosis', serie_diagnosis)

    return serie_data

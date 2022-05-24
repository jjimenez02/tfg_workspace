import math
import random
import numpy as np
import pandas as pd
import utils.constants as cs
from sklearn.base import clone
from joblib import Parallel, delayed
from utils import codifications
from collections import defaultdict
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, LSTM
from scipy.sparse.csr import csr_matrix


def compute_classification_reports_means(reports: list):
    final_report = defaultdict(lambda: defaultdict(lambda: ()))
    aux_report = defaultdict(lambda: defaultdict(lambda: []))
    final_report['accuracy'] = ()
    aux_report['accuracy'] = []

    # Accumulate all data into dict-lists
    for report in reports:
        for key in report.keys():
            if key == 'accuracy':
                aux_report['accuracy']\
                    .append(report['accuracy'])
            else:
                for subkey in report[key].keys():
                    aux_report[key][subkey]\
                        .append(report[key][subkey])

    # Mean & Std computation
    for key in aux_report.keys():
        if key == 'accuracy':
            final_report['accuracy'] =\
                (np.mean(aux_report['accuracy']),
                 np.std(aux_report['accuracy']))
        else:
            for subkey in aux_report[key].keys():
                final_report[key][subkey] =\
                    (np.mean(aux_report[key][subkey]),
                     np.std(aux_report[key][subkey]))
                final_report[key] = dict(final_report[key])

    return dict(final_report)


def apply_lstm_format(
        X: np.array,
        y: np.array,
        n_series: int,
        series_length: int,
        sequences_fragmenter: int,
        labels_encoder):
    X = X.astype('float32')
    X = X.reshape(
        n_series*sequences_fragmenter,
        int(series_length/sequences_fragmenter),
        X.shape[1]
    )
    y = labels_encoder.transform(y)
    if y.__class__ == csr_matrix:
        y = y.toarray()
    y = np.repeat(y, repeats=sequences_fragmenter, axis=0)

    return X, y


def lstm_build_model(units, learning_rate, n_classes):
    nn = Sequential()
    nn.add(LSTM(units=units))
    nn.add(Dense(n_classes, 'softmax'))
    lstm_compile_model(nn, learning_rate)
    return nn


def lstm_compile_model(nn, learning_rate):
    nn.compile(
        optimizer=RMSprop(learning_rate=learning_rate),
        loss='categorical_crossentropy'
    )


def train_test_split(
        df,
        relation_with_series=None,
        train_size=None,
        test_size=None,
        seed=None,
        standardize_columns=None,
        drop_columns=['class', 'id'],
        id_col_name='id',
        class_col_name='class',
        initialize_seed=True):
    identificators = list(pd.unique(df[id_col_name]))
    limit_index = __train_test_split_param_verification(
        identificators, train_size, test_size)

    if initialize_seed:
        random.seed(seed)
    train_identificators, test_identificators =\
        __get_train_test_identificators(identificators, limit_index)

    X_train, y_train = __fill_df_with_data(
        df,
        train_identificators,
        drop_columns,
        id_col_name,
        class_col_name
    )
    X_test, y_test = __fill_df_with_data(
        df,
        test_identificators,
        drop_columns,
        id_col_name,
        class_col_name
    )

    X_train, X_test = codifications.standardize_data(
        X_train, X_test, standardize_columns)

    return X_train, X_test, y_train, y_test


def windowed_train_test_split(
        windowed_series,
        relation_with_series,
        train_size=None,
        test_size=None,
        seed=None,
        standardize_columns=None,
        drop_columns=['id', 'class'],
        id_col_name='id',
        class_col_name='class',
        initialize_seed=True):
    identificators = list(relation_with_series.keys())
    limit_index = __train_test_split_param_verification(
        identificators, train_size, test_size)

    if initialize_seed:
        random.seed(seed)
    train_identificators, test_identificators =\
        __get_train_test_identificators(identificators, limit_index)

    X_train, y_train = __fill_windowed_df_with_data(
        windowed_series,
        relation_with_series,
        train_identificators,
        drop_columns,
        id_col_name,
        class_col_name
    )
    X_test, y_test = __fill_windowed_df_with_data(
        windowed_series,
        relation_with_series,
        test_identificators,
        drop_columns,
        id_col_name,
        class_col_name
    )

    X_train, X_test = codifications.standardize_data(
        X_train, X_test, standardize_columns)

    return X_train, X_test, y_train, y_test


def tfm_marta_train_test_split(
        windowed_series,
        relation_with_series,
        train_size=None,
        test_size=None,
        seed=None,
        standardize_columns=None,
        drop_columns=['id', 'class'],
        id_col_name='id',
        class_col_name='class',
        initialize_seed=True):
    __train_test_split_param_verification(
        train_size=train_size,
        test_size=test_size,
        compute_limit_index=False
    )
    identificators = list(relation_with_series.keys())
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = []
    y_test = []

    if initialize_seed:
        random.seed(seed)
    for identificator in identificators:
        windows_identificators = relation_with_series[identificator]
        limit_index = __get_train_test_split_limit_index(
            windows_identificators, train_size=train_size, test_size=test_size)

        train_identificators, test_identificators =\
            __get_train_test_identificators(
                windows_identificators, limit_index)

        new_X_train, new_y_train = __fill_tfm_marta_df_with_data(
            windowed_series,
            train_identificators,
            drop_columns,
            id_col_name,
            class_col_name
        )
        new_X_test, new_y_test = __fill_tfm_marta_df_with_data(
            windowed_series,
            test_identificators,
            drop_columns,
            id_col_name,
            class_col_name
        )

        X_train = X_train.append(new_X_train)
        X_test = X_test.append(new_X_test)
        y_train.extend(new_y_train)
        y_test.extend(new_y_test)

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)

    X_train, X_test = codifications.standardize_data(
        X_train, X_test, standardize_columns)

    return X_train, X_test, y_train, y_test


def train_validate(
        clf,
        df,
        relation_with_series=None,
        train_test_split_method=None,
        times_to_repeat=5,
        train_size=None,
        test_size=None,
        seed=None,
        standardize_columns=[],
        drop_columns=['id', 'class'],
        id_col_name='id',
        class_col_name='class',
        n_jobs=-2,
        custom_estimator=False,
        return_attrs=[]):
    random.seed(seed)
    return Parallel(n_jobs=n_jobs)(delayed(__parallel_train_validate)(
        clone_estimator(clf, custom_estimator),
        df,
        train_test_split_method,
        standardize_columns,
        drop_columns,
        train_size,
        test_size,
        id_col_name,
        class_col_name,
        relation_with_series,
        return_attrs
    ) for _ in range(0, times_to_repeat))


def windowed_cross_val(
        clf,
        windowed_series,
        relation_with_series,
        estimator_type,
        cv=3,
        seed=None,
        drop_columns=['id', 'class'],
        n_jobs=-2,
        id_col_name='id',
        lstm_dict={}):
    '''
    This method will execute a windowed cross validation over
    a windowed series and a classificator.

    You can determine wether the classificator is:
        - 'classic': SkLearn's classificators
        - 'LSTM': Keras' LSTM model
        - 'SMTS': SMTS' custom model

    If you select LSTM as the estimator you must give the following
    parameters via ``lstm_dict``:
        - series_length (int)
        - sequences_fragmenter (int)
        - fitted_labels_encoder (_BaseEncoder)
        - argmax_fun (fun) -> None for binary classification
        - n_classes (int)
        - epochs (int)
        - units (int)
        - learning_rate (float)
    '''
    partitions_ids = __get_cross_val_partition_series_ids(
        list(relation_with_series.keys()), cv, seed)
    windowed_partitions_ids = __build_windowed_partitions(
        partitions_ids, relation_with_series)

    return Parallel(n_jobs=n_jobs)(delayed(__parallel_windowed_cross_val)(
        clone_estimator(clf, estimator_type),
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict
    ) for windowed_partition in windowed_partitions_ids)


def clone_estimator(clf, estimator_type):
    if (estimator_type == cs.ESTIMATOR_SMTS):
        return clf.clone()
    elif (estimator_type == cs.ESTIMATOR_SKLEARN):
        return clone(clf)
    elif (estimator_type == cs.ESTIMATOR_LSTM):
        # This will be loaded later due to TensorFlow
        # errors with pickle and Keras' models
        return None
    else:
        raise Exception(cs.ERR_INVALID_ESTIMATOR_TYPE_MSSG)


def __get_train_test_identificators(
        identificators, limit_index, shuffle=True):
    if shuffle:
        random.shuffle(identificators)
    return identificators[:limit_index], identificators[limit_index:]


def __fill_df_with_data(
        df,
        identificators,
        drop_columns,
        id_col_name,
        class_col_name):
    X = pd.DataFrame()
    y = []

    for i in identificators:
        serie = df[df[id_col_name] == i]
        X = X.append(serie.drop(
            drop_columns, errors='ignore', axis=1))
        classes = list(serie[class_col_name])
        y.extend(classes)

    X.reset_index(inplace=True, drop=True)

    return X, y


def __fill_windowed_df_with_data(
        windowed_series,
        relation_with_series,
        data_identificators,
        drop_columns,
        id_col_name,
        class_col_name):
    X = pd.DataFrame()
    y = []

    for i in data_identificators:
        for window_identificator in relation_with_series[i]:
            windowed_serie =\
                windowed_series[windowed_series[id_col_name]
                                == window_identificator]
            X = X.append(windowed_serie.drop(
                drop_columns, errors='ignore', axis=1))
            y.extend(list(windowed_serie[class_col_name]))

    X.reset_index(inplace=True, drop=True)

    return X, y


def __fill_tfm_marta_df_with_data(
        windowed_series,
        data_identificators,
        drop_columns,
        id_col_name,
        class_col_name):
    X = windowed_series[windowed_series[id_col_name]
                        .isin(data_identificators)]
    y = list(X[class_col_name])
    X = X.drop(drop_columns, errors='ignore', axis=1)

    return X, y


def __train_test_split_param_verification(
        identificators=None,
        train_size=None,
        test_size=None,
        compute_limit_index=True):
    if (train_size is None) and (test_size is None):
        raise Exception('you must specify a train_size or a test_size')
    elif not (train_size is None) and not (test_size is None):
        raise Exception('you must specify only train_size or only test_size')

    if train_size is None:
        if test_size < 0 or test_size > 1:
            raise Exception(
                'you must specify a valid test_size between 0 and 1')

    elif test_size is None:
        if train_size < 0 or train_size > 1:
            raise Exception(
                'you must specify a valid train_size between 0 and 1')

    if compute_limit_index:
        return __get_train_test_split_limit_index(
            identificators,
            train_size=train_size,
            test_size=test_size)


def __parallel_train_validate(
        clf,
        df,
        train_test_split_method,
        standardize_columns,
        drop_columns,
        train_size=None,
        test_size=None,
        id_col_name='id',
        class_col_name='class',
        relation_with_series=None,
        return_attrs=[]):
    X_train, X_test, y_train, y_test = train_test_split_method(
        df,
        relation_with_series,
        train_size=train_size,
        test_size=test_size,
        standardize_columns=standardize_columns,
        drop_columns=drop_columns,
        id_col_name=id_col_name,
        class_col_name=class_col_name,
        initialize_seed=False
    )
    clf.fit(X_train, y_train)

    return_dict = {}
    return_dict['score'] = clf.score(X_test, y_test)
    for attr in return_attrs:
        return_dict[attr] = getattr(clf, attr)

    return return_dict


def __get_train_test_split_limit_index(
        identificators,
        train_size=None,
        test_size=None):
    if identificators is None:
        raise Exception(
            'you have to specify a list of identificators to split')
    return len(identificators) - math.floor(len(identificators)*test_size)\
        if train_size is None else math.floor(len(identificators)*train_size)


def __get_cross_val_partition_series_ids(series_ids, cv=3, seed=None):
    partitions = []
    random.seed(seed)
    random.shuffle(series_ids)
    proportion = int(len(series_ids)/cv)

    if cv <= 1:
        raise Exception(
            'You have to specify a number of folds greater than 1'
        )

    if proportion <= 0:
        raise Exception(
            'You have to specify a number of folds ' +
            'less than the number of samples')

    for i in range(0, cv):
        partition = {}

        if i == (cv-1):
            partition['test'] = series_ids[i*proportion:]
        else:
            partition['test'] = series_ids[i*proportion:((i+1)*proportion)]

        partition['train'] = series_ids.copy()
        partition['train'] =\
            [elem for elem in partition['train']
                if elem not in partition['test']]
        partitions.append(partition)

    return partitions


def __build_windowed_partitions(series_ids_partitions, relation_with_series):
    windowed_partitions_ids = []

    for partition in series_ids_partitions:
        windowed_partition = defaultdict(list)
        for train_id in partition['train']:
            windowed_partition['train'].extend(relation_with_series[train_id])
        for test_id in partition['test']:
            windowed_partition['test'].extend(relation_with_series[test_id])
        windowed_partitions_ids.append(windowed_partition)

    return windowed_partitions_ids


def __parallel_windowed_cross_val(
        clf,
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict):
    clf = __fit_estimator(
        clf,
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict
    )
    y_test, y_pred = __predict_estimator(
        clf,
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict
    )

    return classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)


def __fit_estimator(
        clf,
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict):
    X_train, y_train = __get_sample_and_class_by_series_ids(
        windowed_series,
        windowed_partition['train'],
        drop_columns=drop_columns
    )

    if (estimator_type == cs.ESTIMATOR_SKLEARN or
            estimator_type == cs.ESTIMATOR_SMTS):
        clf.fit(X_train, y_train)
    elif estimator_type == cs.ESTIMATOR_LSTM:
        # Collapse into one class per serie (auto. done in SMTS' train)
        y_train = X_train.assign(class_name=y_train).groupby(
            id_col_name).first()['class_name'].to_numpy()

        # LSTM's input shape
        X_train, y_train = apply_lstm_format(
            X_train.to_numpy(),
            y_train,
            len(pd.unique(X_train[id_col_name])),
            lstm_dict[cs.LSTM_SERIES_LENGTH],
            lstm_dict[cs.LSTM_SEQUENCES_FRAGMENTER],
            lstm_dict[cs.LSTM_FITTED_LABELS_ENCODER]
        )

        clf = lstm_build_model(
            lstm_dict[cs.LSTM_HYP_PARAM_UNITS],
            lstm_dict[cs.LSTM_HYP_PARAM_LEARNING_RATE],
            lstm_dict[cs.LSTM_N_CLASSES]
        )
        clf.fit(
            X_train,
            y_train,
            epochs=lstm_dict[cs.LSTM_HYP_PARAM_EPOCHS],
            class_weight=lstm_dict[cs.LSTM_CLASS_WEIGHTS],
            verbose=0
        )
    else:
        raise Exception(cs.ERR_INVALID_ESTIMATOR_TYPE_MSSG)

    return clf


def __predict_estimator(
        clf,
        windowed_series,
        windowed_partition,
        estimator_type,
        drop_columns,
        id_col_name,
        lstm_dict):
    X_test, y_test = __get_sample_and_class_by_series_ids(
        windowed_series,
        windowed_partition['test'],
        drop_columns=drop_columns
    )

    if (estimator_type == cs.ESTIMATOR_SMTS or
            estimator_type == cs.ESTIMATOR_LSTM):
        # Collapse into one class per serie
        y_test = X_test.assign(class_name=y_test).groupby(
            id_col_name).first()['class_name'].to_numpy()

        if estimator_type == cs.ESTIMATOR_LSTM:
            # LSTM's input shape
            X_test, y_test = apply_lstm_format(
                X_test.to_numpy(),
                y_test,
                len(pd.unique(X_test[id_col_name])),
                lstm_dict[cs.LSTM_SERIES_LENGTH],
                lstm_dict[cs.LSTM_SEQUENCES_FRAGMENTER],
                lstm_dict[cs.LSTM_FITTED_LABELS_ENCODER]
            )
    elif estimator_type != cs.ESTIMATOR_SKLEARN:
        raise Exception(cs.ERR_INVALID_ESTIMATOR_TYPE_MSSG)

    y_pred = clf.predict(X_test)

    if estimator_type == cs.ESTIMATOR_LSTM:
        # Argmax if its specified (binary vs multiclass classification)
        if lstm_dict[cs.LSTM_ARGMAX_FUNCTION] is not None:
            y_pred = lstm_dict[cs.LSTM_ARGMAX_FUNCTION](
                y_pred, lstm_dict[cs.LSTM_N_CLASSES])
        # Inverse transformation in order to get class' names in the
        # classification report.
        y_test = lstm_dict[cs.LSTM_FITTED_LABELS_ENCODER]\
            .inverse_transform(y_test)
        y_pred = lstm_dict[cs.LSTM_FITTED_LABELS_ENCODER]\
            .inverse_transform(y_pred)

    return y_test, y_pred


def __get_sample_and_class_by_series_ids(
        df,
        series_ids,
        drop_columns=[],
        id_col_name='id',
        class_col_name='class'):
    X = df.loc[df[id_col_name].isin(series_ids)].drop(
        drop_columns, errors='ignore', axis=1)
    y = df.loc[df[id_col_name].isin(series_ids)][class_col_name]\
        .to_numpy()
    return X, y

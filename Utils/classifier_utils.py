import math
import random
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed
from utils import codifications
from collections import defaultdict
from sklearn.metrics import classification_report


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
        cv=3,
        seed=None,
        drop_columns=['id', 'class'],
        n_jobs=-2,
        custom_estimator=False,
        id_col_name='id'):
    partitions_ids = __get_cross_val_partition_series_ids(
        list(relation_with_series.keys()), cv, seed)
    windowed_partitions_ids = __build_windowed_partitions(
        partitions_ids, relation_with_series)

    return Parallel(n_jobs=n_jobs)(delayed(__parallel_windowed_cross_val)(
        clone_estimator(clf, custom_estimator),
        windowed_series,
        windowed_partition,
        drop_columns,
        id_col_name
    ) for windowed_partition in windowed_partitions_ids)


def clone_estimator(clf, custom_estimator=False):
    if (custom_estimator):
        return clf.clone()
    else:
        return clone(clf)


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
        drop_columns,
        id_col_name='id'):
    __fit_stimator(clf, windowed_series, windowed_partition, drop_columns)
    X_test, y_test = __get_sample_and_class_by_series_ids(
        windowed_series, windowed_partition['test'],
        drop_columns=drop_columns)

    # FIXME: Specific to SMTS
    y_test = X_test.assign(class_name=y_test).groupby(
        id_col_name).first()['class_name'].to_numpy()

    y_pred = clf.predict(X_test)

    return classification_report(y_test, y_pred, output_dict=True)


def __fit_stimator(clf, windowed_series, windowed_partition, drop_columns):
    X_train, y_train = __get_sample_and_class_by_series_ids(
        windowed_series, windowed_partition['train'],
        drop_columns=drop_columns)
    clf.fit(X_train, y_train)


def __get_sample_and_class_by_series_ids(df, series_ids, drop_columns=[]):
    X = df.loc[df['id'].isin(series_ids)].drop(
        drop_columns, errors='ignore', axis=1)
    y = df.loc[df['id'].isin(series_ids)]['class'].to_numpy()
    return X, y

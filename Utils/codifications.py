import pandas as pd
from collections import defaultdict
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler


def standardize_data(
        X_train,
        X_test=None,
        headers=None):
    standardized_X_train = pd.DataFrame()
    standardized_X_test = pd.DataFrame() if X_test is not None else None
    scaler = StandardScaler()

    for col in X_train.columns:
        if col in headers:
            scaler.fit(X_train[col].to_numpy().reshape(-1, 1))
            standardized_X_train.insert(
                standardized_X_train.shape[1],
                col,
                scaler.transform(X_train[col].to_numpy().reshape(-1, 1))
            )
            if X_test is not None:
                standardized_X_test.insert(
                    standardized_X_test.shape[1],
                    col,
                    scaler.transform(X_test[col].to_numpy().reshape(-1, 1))
                )
        else:
            standardized_X_train.insert(
                standardized_X_train.shape[1],
                col,
                X_train[col]
            )
            if X_test is not None:
                standardized_X_test.insert(
                    standardized_X_test.shape[1],
                    col,
                    X_test[col]
                )
    if X_test is not None:
        return standardized_X_train, standardized_X_test
    else:
        return standardized_X_train


def mean_fn(
        df,
        headers,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    prefix = 'mean'
    columns = __setup_cols(
        prefix,
        df.columns,
        id_col_name,
        class_col_name,
        time_col_name
    )

    collapsed_data = []
    collapsed_data.append(df[id_col_name].iloc[0])
    for header in headers:
        collapsed_data.append(df[header].mean())
    collapsed_data.append(df[class_col_name].iloc[0])

    return pd.DataFrame([collapsed_data], columns=columns)


def std_fn(
        df,
        headers,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    prefix = 'std'
    columns = __setup_cols(
        prefix,
        df.columns,
        id_col_name,
        class_col_name,
        time_col_name
    )

    collapsed_data = []
    collapsed_data.append(df[id_col_name].iloc[0])
    for header in headers:
        collapsed_data.append(df[header].std())
    collapsed_data.append(df[class_col_name].iloc[0])
    return pd.DataFrame([collapsed_data], columns=columns)


def slope_fn(
        df,
        headers,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    prefix = 'slope'
    columns = __setup_cols(
        prefix,
        df.columns,
        id_col_name,
        class_col_name,
        time_col_name
    )
    time = df[time_col_name].to_numpy()

    collapsed_data = []
    collapsed_data.append(df[id_col_name].iloc[0])
    for header in headers:
        collapsed_data.append(linregress(time, df[header].to_numpy())[0])
    collapsed_data.append(df[class_col_name].iloc[0])
    return pd.DataFrame([collapsed_data], columns=columns)


def temporal_trend_fn(
        df,
        headers=None,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    '''
    It will get the differences between two neighboor points
    of the same feature.
    '''
    columns = __temporal_trend_setup_cols(
        df,
        id_col_name,
        class_col_name,
        time_col_name
    )
    data_with_features_diff = defaultdict(list)

    previous_sample = None
    for sample in df.iloc:
        if previous_sample is None:
            previous_sample = sample
            continue

        data_with_features_diff[columns[0]].append(sample[id_col_name])
        data_with_features_diff[columns[1]].append(sample[time_col_name])

        for i in range(2, len(columns)-2, 2):
            if columns[i] in headers:
                data_with_features_diff[columns[i]].append(
                    sample[columns[i]])
                data_with_features_diff[columns[i+1]].append(
                    sample[columns[i]] - previous_sample[columns[i]])
            else:
                data_with_features_diff[columns[i]].append(
                    sample[columns[i]])

        data_with_features_diff[columns[-1]].append(sample[class_col_name])
        previous_sample = sample

    return pd.DataFrame(data_with_features_diff)


def temporal_differences_fn(
        df,
        headers=None,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    '''
    It will get the numeric differences between all data's attributes

    [A1, B1
     A2, B2
     A3, B3
     ...] -> [A1-B1
              A2-B2
              A3-B3
              ...]
    '''
    columns = __temporal_differences_setup_cols(
        df,
        id_col_name,
        class_col_name,
        time_col_name
    )

    data_with_differences = {}

    for column in columns:
        attrs_to_sub = column.split('-')
        if len(attrs_to_sub) == 2:
            data_with_differences[column] = \
                (df[attrs_to_sub[0]]-df[attrs_to_sub[1]]).to_numpy()
        else:
            data_with_differences[column] = df[column].to_numpy()

    return pd.DataFrame(data_with_differences)


def __setup_cols(
        prefix,
        df_columns,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    columns = list(df_columns)
    columns.remove(id_col_name)
    columns.remove(class_col_name)
    columns.remove(time_col_name)

    final_columns = [col + prefix for col in columns]
    final_columns.insert(0, id_col_name)
    final_columns.append(class_col_name)
    return final_columns


def __temporal_trend_setup_cols(
        df,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    final_columns = []

    columns = list(df.columns)
    columns.remove(id_col_name)
    columns.remove(class_col_name)
    columns.remove(time_col_name)

    final_columns.append(time_col_name)
    for attr in columns:
        final_columns.append(attr)
        final_columns.append(attr + 'Diff')

    final_columns.insert(0, id_col_name)
    final_columns.append(class_col_name)

    return final_columns


def __temporal_differences_setup_cols(
        df,
        id_col_name='id',
        class_col_name='class',
        time_col_name='TimeStamp'):
    final_columns = []

    columns = list(df.columns)
    columns.remove(id_col_name)
    columns.remove(class_col_name)
    columns.remove(time_col_name)

    final_columns.append(time_col_name)
    for first_attr in columns:
        final_columns.append(first_attr)
        for second_attr in columns:
            attr_diff_col = first_attr + '-' + second_attr
            if (first_attr != second_attr) and \
                (attr_diff_col not in final_columns) and \
                    (second_attr + '-' + first_attr not in final_columns):
                final_columns.append(attr_diff_col)

    final_columns.insert(0, id_col_name)
    final_columns.append(class_col_name)

    return final_columns

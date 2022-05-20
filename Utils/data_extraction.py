import math
import pandas as pd
import numpy as np
import utils.classifier_utils as clfutils
import utils.codifications as codifications
from collections import defaultdict
from utils.plot_utils import plot_series_from_df_by_id


class Data():
    '''
    This class' target is to build a flat Pandas DataFrame
    originated from a pre-determined folder's jerarchy.

    All functions will have a `df`'s parameter, if it isn't specified
    then changes will be done with `derived_data`'s attr and saved into
    `derived_data`'s attr.
    '''

    @staticmethod
    def concat_data(
            data1: pd.DataFrame,
            data2: pd.DataFrame,
            id_col_name='id'):
        '''
        This static method will concat two "Data"'s types into a single one.

        If there are conflicts with the Primary Key (id) this method will
        concat the second "Data" to the first one replacing all ids by the
        subsequent to the last "Data"'s id (one by one).
        '''
        new_df = data1.copy(deep=True)
        data2_identificators = pd.unique(data2[id_col_name])

        for data2_identif in data2_identificators:
            data2_actual_serie = data2[data2[id_col_name] == data2_identif]

            if data2_identif in new_df[id_col_name].values:
                data2_actual_serie[id_col_name] =\
                    new_df.iloc[-1][id_col_name] + 1

            new_df = pd.concat([new_df, data2_actual_serie])

        return new_df

    def __init__(self, df: pd.DataFrame, windows_per_serie=None):
        self.original_data = df
        self.original_data_windows_per_serie = windows_per_serie
        self.reset_changes()

    def reset_changes(self):
        self.derived_data =\
            self.original_data.copy(deep=True)
        self.derived_data_windows_per_serie =\
            self.original_data_windows_per_serie

    def get_derived_data_windows_per_serie(self, default=None):
        windows_per_serie = self.derived_data_windows_per_serie\
            if default is None else default

        if windows_per_serie is None:
            raise Exception('you have to split \'derived_data\' into windows first\
                (see \'split_into_windows(...)\')')

        return windows_per_serie

    def get_derived_data_identifiers(self, id_col_name='id'):
        return pd.unique(self.derived_data[id_col_name])

    def get_derived_data_classes_count(
            self,
            id_col_name='id',
            class_col_name='class'):
        return self.derived_data.groupby(id_col_name).first()[
            class_col_name].value_counts()

    def get_shortest_serie(self, df=None, id_col_name='id'):
        data = self.derived_data if df is None else df
        identificators = pd.unique(data[id_col_name])

        shortest_serie = (-1, np.inf)
        for identif in identificators:
            actual_serie = data[data[id_col_name] == identif]
            if (shortest_serie[1] > actual_serie.shape[0]):
                shortest_serie = (identif, actual_serie.shape[0])

        if (shortest_serie[0] == -1):
            raise Exception("There are not valid series in the dataset")

        return data[data[id_col_name] == shortest_serie[0]]

    def get_largest_serie(self, df=None, id_col_name='id'):
        data = self.derived_data if df is None else df
        identificators = pd.unique(data[id_col_name])

        largest_serie = (-1, -np.inf)
        for identif in identificators:
            actual_serie = data[data[id_col_name] == identif]
            if (largest_serie[1] < actual_serie.shape[0]):
                largest_serie = (identif, actual_serie.shape[0])

        if (largest_serie[0] == -1):
            raise Exception("There are not valid series in the dataset")

        return data[data[id_col_name] == largest_serie[0]]

    def get_derived_data_columns(
            self,
            id_col_name='id',
            time_col_name='TimeStamp',
            class_col_name='class'):

        columns = list(self.derived_data.columns)
        columns.remove(id_col_name)
        columns.remove(time_col_name)
        columns.remove(class_col_name)
        return {'id': id_col_name,
                'time': time_col_name,
                'class': class_col_name,
                'attrs': columns}

    def drop_derived_data_columns(self, columns_names):
        self.derived_data = self.derived_data.drop(
            columns_names, errors='ignore', axis=1)
        return self.derived_data

    def rename_derived_data_column(self, old_name, new_name):
        self.derived_data.rename(
            columns={old_name: new_name}, inplace=True)
        return self.derived_data

    def clean_data(
            self,
            df=None,
            criterion='remove',
            value=0,
            headers=[],
            drop_columns=[]):
        '''
        This method will apply a criterion for data cleaning
        available criterions are:
            'remove': It will remove all rows with the specified value
            in 'value''s parameter in the dataframe's 'headers' parameter.
        '''
        data = self.derived_data if df is None else df

        if criterion == 'remove':
            data = self.__remove_rows(data, value, headers)
        else:
            raise Exception(
                'Not recognized criterion, available criterions are:\
                    \'remove\'')

        data = data.drop(drop_columns, axis=1, errors='ignore')
        return self.__save_data(data, df is None)

    def apply_codifications(
            self,
            codification_fns: list,
            headers=None,
            df=None,
            id_col_name='id',
            time_col_name='TimeStamp',
            class_col_name='class'):
        data = self.derived_data if df is None else df
        identifiers =\
            self.get_derived_data_identifiers(id_col_name=id_col_name)\
            if df is None else pd.unique(df[id_col_name])
        columns = self.get_derived_data_columns()['attrs']\
            if headers is None else headers

        new_df = self.__apply_codification_fn(
            data,
            codification_fns[0],
            identifiers,
            columns,
            id_col_name=id_col_name,
            time_col_name=time_col_name,
            class_col_name=class_col_name
        )

        for codification_fn in codification_fns[1:]:
            other_codification = self.__apply_codification_fn(
                data,
                codification_fn,
                identifiers,
                columns,
                id_col_name=id_col_name,
                time_col_name=time_col_name,
                class_col_name=class_col_name
            )

            new_df = pd.merge(
                left=new_df,
                right=other_codification,
                left_on=[id_col_name, class_col_name],
                right_on=[id_col_name, class_col_name]
            )

        return self.__save_data(new_df, df is None)

    def apply_codification_to_class(
            self,
            codificator,
            df=None,
            class_col_name='class'):
        data = self.derived_data if df is None else df

        codificator.fit(data[class_col_name])
        data[class_col_name] = codificator.transform(data[class_col_name])

        return self.__save_data(data, df is None)

    def remove_outliers(
            self,
            df=None,
            headers=None,
            low_quant=.05,
            high_quant=.95,
            outliers_limit=.3,
            id_col_name='id'):
        '''
        This method will remove all series with a outliers' percentage greater
        than outliers_limit's parameter.

        We consider an outlier a sample with at least a dimension out of the
        low_quant and high_quant's percentiles.
        '''
        data = self.derived_data if df is None else df

        remaining_data = self.__filter_outside_quantil_samples(
            data, headers, low_quant, high_quant)
        data = self.__remove_outliers_by_outliers_limit(
            data, remaining_data, outliers_limit, id_col_name)

        data.reset_index(inplace=True, drop=True)
        return self.__save_data(data, df is None)

    def reduce_sampling_rate(
            self,
            df=None,
            remove_one_each_n_samples=2,
            id_col_name='id'):
        data = self.derived_data if df is None else df

        new_df = pd.DataFrame()
        series_id = pd.unique(data[id_col_name])

        for serie_id in series_id:
            serie = data[data[id_col_name] == serie_id]\
                .reset_index(drop=True)
            new_df = new_df.append(
                serie.drop(
                    range(0, serie.shape[0], remove_one_each_n_samples),
                    axis=0
                ),
                ignore_index=True
            )

        return self.__save_data(new_df, df is None)

    def split_into_windows(
            self,
            df=None,
            n_windows=None,
            window_size=None,
            id_col_name='id'):
        data = self.derived_data if df is None else df

        if (n_windows is None and window_size is None):
            raise Exception("You have to specify a window size")

        new_df = pd.DataFrame()
        series_id = pd.unique(data[id_col_name])
        windows_per_serie = defaultdict(list)

        actual_n_windows = 0.0
        for serie_id in series_id:
            actual_serie = data[data[id_col_name] == serie_id]
            if (n_windows is None):
                actual_n_windows = math.ceil(
                    float(actual_serie.shape[0])/float(window_size))
            else:
                actual_n_windows = n_windows

            new_df = new_df.append(self.__split_serie_into_windows(
                actual_serie,
                windows_per_serie,
                serie_id,
                actual_n_windows,
                id_col_name
            ), ignore_index=True)

        return self.__save_data(
            new_df, df is None, windows_per_serie=windows_per_serie)

    def train_test_split(
            self,
            df=None,
            criterion='tfm_marta',
            windows_per_serie=None,
            train_size=None,
            test_size=None,
            random_state=None,
            standardize_columns=[],
            drop_columns=['id', 'class'],
            id_col_name='id',
            class_col_name='class'):
        '''
        This method will split train and test's sets.
        Available criterions are:
            'tfm_marta': For each serie 'train_size' of his windows
            will be put together in train's set (the rest will be put
            into test's set)
            'windowed': It will split train and test's sets by series
            identificators (windows of the same serie won't be in train
            and test's sets at the same time).
            'normal': It will simply split the data into train and test's sets.
        '''
        data = self.derived_data if df is None else df

        if criterion == 'tfm_marta':
            data_windows_per_serie = self.get_derived_data_windows_per_serie(
                default=windows_per_serie)
            return clfutils.tfm_marta_train_test_split(
                windowed_series=data,
                relation_with_series=data_windows_per_serie,
                train_size=train_size,
                test_size=test_size,
                seed=random_state,
                standardize_columns=standardize_columns,
                drop_columns=drop_columns,
                id_col_name=id_col_name,
                class_col_name=class_col_name)
        elif criterion == 'normal':
            return clfutils.train_test_split(
                df=data,
                train_size=train_size,
                test_size=test_size,
                seed=random_state,
                standardize_columns=standardize_columns,
                drop_columns=drop_columns,
                id_col_name=id_col_name,
                class_col_name=class_col_name)
        elif criterion == 'windowed':
            data_windows_per_serie = self.get_derived_data_windows_per_serie(
                default=windows_per_serie)
            return clfutils.windowed_train_test_split(
                windowed_series=data,
                relation_with_series=data_windows_per_serie,
                train_size=train_size,
                test_size=test_size,
                seed=random_state,
                standardize_columns=standardize_columns,
                drop_columns=drop_columns,
                id_col_name=id_col_name,
                class_col_name=class_col_name)
        else:
            raise Exception('Not recognized criterion, available criterions are:\
                    \'tfm_marta\', \'normal\', \'windowed\'')

    def plot_series_by_id(
            self,
            ids,
            attr,
            df=None,
            id_col_name="id",
            time_col_name="TimeStamp"):
        data = self.derived_data if df is None else df
        plot_series_from_df_by_id(data, ids, attr, id_col_name=id_col_name,
                                  time_col_name=time_col_name)

    def __remove_rows(self, df, value, headers):
        for header in headers:
            df = df.drop(df[df[header] == value].index)
        df.reset_index(drop=True, inplace=True)
        return df

    def __save_data(
            self,
            data_to_save,
            is_df_none,
            windows_per_serie=None):
        if is_df_none:
            self.derived_data = data_to_save
            if windows_per_serie is not None:
                self.derived_data_windows_per_serie = windows_per_serie
        else:
            return data_to_save, windows_per_serie

    def __apply_codification_fn(
            self,
            df,
            codification_fn,
            identifiers,
            headers,
            id_col_name='id',
            time_col_name='TimeStamp',
            class_col_name='class'):
        new_df = pd.DataFrame()
        for identifier in identifiers:
            serie = df[df[id_col_name] == identifier]
            new_df = new_df.append(codification_fn(
                serie,
                headers,
                id_col_name=id_col_name,
                time_col_name=time_col_name,
                class_col_name=class_col_name))

        new_df.reset_index(inplace=True, drop=True)
        return new_df

    def __filter_outside_quantil_samples(
            self,
            data,
            headers,
            low_quant,
            high_quant):
        standardizedData = codifications\
            .standardize_data(data, headers=headers)
        quant_df = standardizedData.quantile([low_quant, high_quant])

        for header in headers:
            standardizedData = standardizedData[
                (standardizedData[header] > quant_df.loc[low_quant, header])
                & (standardizedData[header] < quant_df.loc[high_quant, header])
            ]

        return data.loc[standardizedData.index]

    def __remove_outliers_by_outliers_limit(
            self,
            data,
            remaining_data,
            outliers_limit,
            id_col_name):
        identificators = pd.unique(data[id_col_name])
        for identif in identificators:
            actual_serie = data[data[id_col_name] == identif]
            remaining_serie =\
                remaining_data[remaining_data[id_col_name] == identif]

            remaining_serie_proportion =\
                float(remaining_serie.shape[0])/float(actual_serie.shape[0])

            if remaining_serie_proportion == 0 or\
                    1 - remaining_serie_proportion >= outliers_limit:
                data = data.drop(actual_serie.index)

        return data

    def __split_serie_into_windows(
            self,
            serie,
            windows_per_serie,
            serie_id,
            n_windows,
            id_col_name='id'):
        new_df = pd.DataFrame()
        window_id = 0
        window_size = serie.shape[0]//n_windows

        window_df = pd.DataFrame()
        for i in range(0, n_windows):
            window_name = str(serie_id) + 'w' + str(window_id)
            start_sample = i * window_size
            end_sample = start_sample + window_size

            window_df = serie.iloc[start_sample:end_sample, :]
            window_df = window_df.assign(**{id_col_name: window_name})
            new_df = new_df.append(window_df)

            windows_per_serie[serie_id].append(window_name)
            window_id += 1
        return new_df

import pandas as pd
import utils.classifier_utils as clfutils
import utils.codifications as codifications
from collections import defaultdict


class Data():
    '''
    This class' target is to build a flat Pandas DataFrame
    originated from a pre-determined folder's jerarchy.

    All functions will have a `df`'s parameter, if it isn't specified
    then changes will be done with `derived_data`'s attr and saved into
    `derived_data`'s attr.
    '''

    def __init__(self, df: pd.DataFrame):
        self.original_data = df
        self.derived_data = self.original_data.copy(deep=True)
        self.derived_data_windows_per_serie = None

    def get_derived_data_windows_per_serie(self, default=None):
        windows_per_serie = self.derived_data_windows_per_serie\
            if default is None else default

        if windows_per_serie is None:
            raise Exception('you have to split \'derived_data\' into windows first\
                (see \'split_into_windows(...)\')')

        return windows_per_serie

    def get_derived_data_identifiers(self, id_col_name='id'):
        return pd.unique(self.derived_data[id_col_name])

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
            high_quant=.95):
        data = self.derived_data if df is None else df

        standardizedData = codifications\
            .standardize_data(data, headers=headers)
        quant_df = standardizedData.quantile([low_quant, high_quant])

        for header in headers:
            standardizedData = standardizedData[
                (standardizedData[header] > quant_df.loc[low_quant, header])
                & (standardizedData[header] < quant_df.loc[high_quant, header])
            ]

        data = data.loc[standardizedData.index]
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
            n_windows=4,
            id_col_name='id'):
        data = self.derived_data if df is None else df

        new_df = pd.DataFrame()
        series_id = pd.unique(data[id_col_name])
        windows_per_serie = defaultdict(list)

        for serie_id in series_id:
            new_df = new_df.append(self.__split_serie_into_windows(
                data[data[id_col_name] == serie_id],
                windows_per_serie,
                serie_id,
                n_windows,
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

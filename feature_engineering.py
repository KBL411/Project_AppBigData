import pandas as pd
import numpy as np
from sklearn import preprocessing
from data_preparation import df_train


def encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function encode the non_numerical columns.
    We need it cause most model doesn't manage non-numerical columns
    :param dataframe: It take a dataframe as a parameter
    :return: It returns a numerical dataframe
    """
    encoder = preprocessing.LabelEncoder()
    dataframe = dataframe.apply(encoder.fit_transform)
    return dataframe


def normalise(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function apply a Min-Max Scaler on the whole dataframe. We cast the result as a dataframe.
    :param dataframe: It take a dataframe as a parameter
    :return: It return a normalised dataframe, less power hungry to train.
    """
    normalizer = preprocessing.MinMaxScaler()
    dataframe = pd.DataFrame(normalizer.fit_transform(dataframe), columns=dataframe.columns, index=dataframe.index)
    return dataframe


def featuring(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function do the complete featuring, it use the normalise and the encoding functions that are present in feature_engineering.py
    It split into numerical and non-numerical.
    Encode the non-numerical columns with the encoding function.
    Concatenate the 2 dataset now they all are numerical.
    And lastly we normalise the whole dataframe thank to the normalise function.
    We use it on the train and the test datasets.
    :param dataframe: It take a dataframe as a parameter
    :return: It return a featured dataframe ready to be used by training
    """
    df_non_numerical = dataframe.select_dtypes(object)
    df_numerical = dataframe.select_dtypes(include=np.number)

    df_non_numerical = encoding(df_non_numerical)
    dataframe = pd.concat([df_numerical, df_non_numerical], axis=1)

    dataframe = normalise(dataframe)
    return dataframe


df_train.drop(['FLAG_MOBIL', 'FLAG_DOCUMENT_2', 'SK_ID_CURR'], axis=1, inplace=True)
df_train = featuring(df_train)

print('end of data featuring\n')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from data_preparation import df


def encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoder = preprocessing.LabelEncoder()
    dataframe = dataframe.apply(encoder.fit_transform)
    return dataframe


def normalise(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalizer = preprocessing.MinMaxScaler()
    dataframe = pd.DataFrame(normalizer.fit_transform(dataframe), columns=dataframe.columns, index=dataframe.index)
    return dataframe


df_non_numerical = df.select_dtypes(object)
df_numerical = df.select_dtypes(include=np.number)

df_non_numerical = encoding(df_non_numerical)

df = pd.concat([df_numerical, df_non_numerical], axis=1)

df.drop(['FLAG_MOBIL', 'FLAG_DOCUMENT_2', 'SK_ID_CURR'], axis=1, inplace=True)

df = normalise(df)

import pandas as pd


def null_by_columns(dataframe: pd.DataFrame):
    """
    This fonction list the columns and their percentage of missing values.
    :param dataframe: It take a dataframe as a parameter
    :return: It doesn't have a return since it's a printing function
    """

    for col in dataframe.columns:
        percent = dataframe[col].isnull().mean()
        print(f'{col}: {round(percent * 100, 2)}%')


def missing_values(dataframe: pd.DataFrame, percent: int) -> pd.DataFrame:
    """
    This function suppress the columns that have a percentage of missing values higher than a treshold define by the users
    :param dataframe: It take a dataframe as a parameter
    :param percent: It is the threshold where
    :return: Return the same dataframe without the columns that
    """
    percent_missing = dataframe.isnull().sum() * 100 / len(dataframe)
    missing_value_df = pd.DataFrame({'column_name': dataframe.columns, 'percent_missing': percent_missing})
    missing_drop = list(missing_value_df[missing_value_df.percent_missing > percent].column_name)
    dataframe = dataframe.drop(missing_drop, axis=1)
    return dataframe


df_train = pd.read_csv("data/application_train.csv")

df_train = missing_values(df_train, 19)

df_train = df_train.dropna(how='any')

print('end of data preparation\n')

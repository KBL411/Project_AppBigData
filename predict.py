import pandas as pd
import joblib
from feature_engineering import featuring
from feature_engineering import df_train

df_test = pd.read_csv('data/application_test.csv')

model_GBC = joblib.load('model/GradientBoostingClass.model.joblib')
model_RFC = joblib.load('model/RandomForestClass.model.joblib')
model_XGB = joblib.load('model/Xgboost.model.joblib')

df_test = featuring(df_test)

df_train = df_train.drop(['TARGET'], axis=1)

df_test = df_test[df_train.columns]

df_test = df_test.dropna(how='any')

df_test =featuring(df_test)

#choose a model first
prediction = model_XGB.predict(df_test)

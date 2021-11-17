import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from feature_engineering import df_train


def model_accuracy(features_test: pd.DataFrame, target_test: pd.DataFrame, model):
    target_pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, target_pred)
    print(confusion_matrix(target_test, target_pred))
    print("Accuracy: ", (accuracy * 100.0))
    print('\n \n')


y = df_train[['TARGET']]
X = df_train.drop('TARGET', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('XGboost')
model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train.values.ravel())
model_accuracy(X_test, y_test, model_XGB)
joblib.dump(model_XGB, "model/Xgboost.model.joblib")

print('RandomForest')
model_RFC = RandomForestClassifier(n_estimators=100)
model_RFC.fit(X_train, y_train.values.ravel())
model_accuracy(X_test, y_test, model_RFC)
joblib.dump(model_RFC, "model/RandomForestClass.model.joblib")

print('GradientBoosting')
model_GBC = GradientBoostingClassifier(n_estimators=100)
model_GBC.fit(X_train, y_train.values.ravel())
model_accuracy(X_test, y_test, model_GBC)
joblib.dump(model_GBC, "model/GradientBoostingClass.model.joblib")


print('end of model training')
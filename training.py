import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from feature_engineering import df


def model_accuracy(features_test: pd.DataFrame, target_test: pd.DataFrame, model):
    target_pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, target_pred)
    print(confusion_matrix(target_test, target_pred))
    print("Accuracy: ", (accuracy * 100.0))


y = df[['TARGET']]
X = df.drop('TARGET', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)
model_accuracy(X_test, y_test, model_XGB)

model_RFC = RandomForestClassifier(n_estimators=100)
model_RFC.fit(X_train, y_train)
model_accuracy(X_test, y_test, model_RFC)

model_GBC = GradientBoostingClassifier(n_estimators=100)
model_GBC.fit(X_train, y_train)
model_accuracy(X_test, y_test, model_GBC)

model_XGB.save_model('XGBOOST.model')

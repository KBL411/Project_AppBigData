import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from feature_engineering import df_train
import mlflow
from urllib.parse import urlparse


def model_accuracy(features_test: pd.DataFrame, target_test: pd.DataFrame, model):
    target_pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, target_pred)
    mlflow.log_metric("accuracy", (accuracy * 100.0))
    cm = confusion_matrix(target_test, target_pred)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    mlflow.log_metric("true negative", true_neg)
    mlflow.log_metric("false positive", false_pos)
    mlflow.log_metric("false negative", false_neg)
    mlflow.log_metric("true positive", true_pos)
    print(cm)
    print("Accuracy: ", (accuracy * 100.0))


def train(random_state, n_estimators):
    y = df_train[['TARGET']]
    X = df_train.drop('TARGET', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    mlflow.log_param("data split random state", random_state)

    print('\nRandomForest')
    model_RFC = RandomForestClassifier(n_estimators=n_estimators)
    model_RFC.fit(X_train, y_train.values.ravel())
    mlflow.log_param("n estimators", n_estimators)
    model_accuracy(X_test, y_test, model_RFC)
    joblib.dump(model_RFC, "model/RandomForestClass.model.joblib", compress=3)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(model_RFC, "model", registered_model_name="RandomForest")
    else:
        mlflow.sklearn.log_model(model_RFC, "model")

    print('end of model training\n')


if __name__ == "__main__":
    random_state = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n_estimators = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    train(random_state, n_estimators)
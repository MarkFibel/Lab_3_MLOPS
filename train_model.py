import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

def scale_frame(frame, target_column):
    df = frame.copy()
    X, y = df.drop(columns=[target_column]), df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, y

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")  # Укажи путь к твоему датасету
    target_column = "Survived"  # Замени на имя целевой переменной
    X, y = scale_frame(df, target_column)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['hinge', 'log_loss', 'modified_huber'],
        "fit_intercept": [False, True],
    }
    
    mlflow.set_experiment("classification_experiment")
    with mlflow.start_run():
        model = SGDClassifier(random_state=42)
        clf = GridSearchCV(model, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)
        
        accuracy, precision, recall, f1 = eval_metrics(y_val, y_pred)
        mlflow.log_params({
            "alpha": best_model.alpha,
            "l1_ratio": best_model.l1_ratio,
            "penalty": best_model.penalty,
            "loss": best_model.loss,
            "fit_intercept": best_model.fit_intercept,
        })
        mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})
        
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        with open("best_model.pkl", "wb") as file:
            joblib.dump(best_model, file)
    
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)

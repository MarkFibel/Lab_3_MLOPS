
# from os import name
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import mlflow
# from sklearn.linear_model import SGDRegressor
# from sklearn.model_selection import GridSearchCV
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from mlflow.models import infer_signature
# import joblib


# def scale_frame(frame):
#     df = frame.copy()
#     X,y = df.drop(columns = ['Survived']), df['Survived']
#     scaler = StandardScaler()
#     power_trans = PowerTransformer()
#     X_scale = scaler.fit_transform(X.values)
#     Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
#     return X_scale, Y_scale, power_trans

# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


# if __name__ == "__main__":
#     df = pd.read_csv("data/processed_data.csv")
#     X,Y, power_trans = scale_frame(df)
#     X_train, X_val, y_train, y_val = train_test_split(X, Y,
#                                                     test_size=0.3,
#                                                     random_state=42)
    

#     params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
#             'l1_ratio': [0.001, 0.05, 0.01, 0.2],
#             "penalty": ["l1","l2","elasticnet"],
#             "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
#             "fit_intercept": [False, True],
#             }
    
#     mlflow.set_experiment("linear model titanic")
#     with mlflow.start_run():
#         lr = SGDRegressor(random_state=42)
#         clf = GridSearchCV(lr, params, cv = 3, n_jobs = 4)
#         clf.fit(X_train, y_train.reshape(-1))
#         best = clf.best_estimator_
#         y_pred = best.predict(X_val)
#         y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
#         (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
#         alpha = best.alpha
#         l1_ratio = best.l1_ratio
#         penalty = best.penalty
#         eta0 = best.eta0
#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_param("penalty", penalty)
#         mlflow.log_param("eta0", eta0)
#         mlflow.log_param("loss", best.loss)
#         mlflow.log_param("fit_intercept", best.fit_intercept)
#         mlflow.log_param("epsilon", best.epsilon)
        
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)
        
#         predictions = best.predict(X_train)
#         signature = infer_signature(X_train, predictions)
#         mlflow.sklearn.log_model(best, "model", signature=signature)
#         with open("lr_titanic.pkl", "wb") as file:
#             joblib.dump(lr, file)

#     dfruns = mlflow.search_runs()
#     path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model' #путь до эксперимента с лучшей моделью
#     print(path2model)
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

def scale_frame(frame, target_column):
    df = frame.copy()
    X, y = df.drop(columns=[target_column]), df[target_column]
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")  # Укажи путь к твоему датасету
    target_column = "Survived"  # Замени на имя целевой переменной
    X, Y, power_trans = scale_frame(df, target_column)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }
    
    mlflow.set_experiment("custom_model_experiment")
    with mlflow.start_run():
        model = SGDRegressor(random_state=42)
        clf = GridSearchCV(model, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.ravel())
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        
        rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        mlflow.log_params({
            "alpha": best_model.alpha,
            "l1_ratio": best_model.l1_ratio,
            "penalty": best_model.penalty,
            "eta0": best_model.eta0,
            "loss": best_model.loss,
            "fit_intercept": best_model.fit_intercept,
            "epsilon": best_model.epsilon,
        })
        mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
        
        predictions = best_model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        with open("best_model.pkl", "wb") as file:
            joblib.dump(best_model, file)
    
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)

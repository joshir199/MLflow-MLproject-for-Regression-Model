import datetime
import sys
import os
import numpy as np
import pandas as pd
from pprint import pprint
import warnings

from sklearn import datasets, model_selection, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow


def get_preprocessed_data(path):
    pprint("data path: ", path)
    # reading and indexing data
    raw_data = pd.read_csv(path, sep=',', header=0, parse_dates=['dteday'])
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
                                    axis=1)

    # defining features and target
    target = 'cnt'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'mnth', 'yr']
    categorical_features = ['season', 'workingday', 'holiday']

    # creating subset of data from entire dataset (2 months)
    reference_data = raw_data.loc["2011-01-01 00:00:00":"2011-02-28 23:00:00"]  # for training as reference model

    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        reference_data[numerical_features + categorical_features], reference_data[target], test_size=0.2)

    return train_X, train_Y, test_X, test_Y


def eval_metrics(actual, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual, predicted_values))
    mae = mean_absolute_error(actual, predicted_values)
    return rmse, mae


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')

    default_path = "dataset/hour.csv"
    default_seed = 33
    np.random.seed(default_seed)

    n_estimator = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    path = str(sys.argv[2]) if len(sys.argv) > 2 else default_path

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    train_x, train_y, test_x, test_y = get_preprocessed_data(data_path)

    with mlflow.start_run() as run:
        
        mlflow.sklearn.autolog()  # enable autologging
        
        regressor_model = ensemble.RandomForestRegressor(n_estimators=n_estimator, random_state=default_seed)

        regressor_model.fit(train_x, train_y)
        predicted_values = regressor_model.predict(test_x)

        rmse, mae = eval_metrics(test_y, predicted_values)

        mlflow.log_param("n_estimators", n_estimator)
        mlflow.log_metric("rmse", round(rmse, 3))
        mlflow.log_metric("mae", round(mae, 3))




from flask import Flask, json, request, jsonify
import numpy as np
import pandas as pd
import datetime
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)


def group_data(training_data):
    df = pd.DataFrame(training_data)
    group_by = df.groupby("Date")
    df = group_by["Stocks Sold"].sum()
    return df


def clean_X_data(df):
    X = {}
    X["Month"] = pd.Series(df.index).apply(lambda x: int(x[5:7]))
    X["Date"] = pd.Series(df.index).apply(lambda x: int(x[8:]))
    X = pd.DataFrame(X).values
    return X


def clean_X_testing_data(testing_data):
    df = pd.Series(testing_data)
    X = {}
    X["Month"] = df.apply(lambda x: int(x[5:7])).values
    X["Date"] = df.apply(lambda x: int(x[8:])).values
    return pd.DataFrame(X).values


def apply_regression_model(X, y):
    clf = RandomForestRegressor(n_estimators=10)
    clf.fit(X, y)
    return clf


def predict_y(clf, X_test):
    y_pred = clf.predict(X_test)
    return np.round(y_pred)


def generate_training_data(company, medicine):
    file_path = "Sample Data Generation/" + company + "_" + medicine + ".csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.to_dict()

    d = {"Date": [], "Stocks Sold": []}
    for _ in range(12000):
        start_date = datetime.date(2015, 1, 1)
        x = datetime.datetime.now()
        end_date = datetime.date(x.year, x.month, x.day)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        d["Date"].append(str(random_date).split(" ")[0])
        stocks_seed = random.randint(1, 200)
        stock = random.randint(1, stocks_seed)
        d["Stocks Sold"].append(stock)
    df = pd.DataFrame(d)
    df.to_csv(file_path, index=False)
    return d


@app.route("/predict-sales", methods=["POST"])
def post_predict_sales():
    data = request.get_json()
    company = data["company"]
    medicine = data["medicine"]
    training_data = data.get("training_data", None) or generate_training_data(
        company, medicine
    )
    testing_data = data["testing_data"]
    df = group_data(training_data)
    X = clean_X_data(df)
    y = df.values
    clf = apply_regression_model(X, y)
    X_test = clean_X_testing_data(testing_data["Date"])
    y_pred = predict_y(clf, X_test)
    return jsonify({"result": list(y_pred)})

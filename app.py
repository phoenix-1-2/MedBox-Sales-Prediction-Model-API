from flask import Flask, json , request,jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
def group_data(training_data):
    df = pd.DataFrame(training_data)
    group_by = df.groupby('Date')
    df = group_by['Stocks Sold'].sum()
    return df

def clean_X_data(df):
    X = {}
    X['Month'] = pd.Series(df.index).apply(lambda x : int(x[5 : 7]) )
    X['Date'] = pd.Series(df.index).apply(lambda x : int(x[8 : ]) )
    X = pd.DataFrame(X).values
    return X

def clean_X_testing_data(testing_data):
    df = pd.Series(testing_data)
    X = {}
    X['Month'] = df.apply(lambda x : int(x[5 : 7]) ).values
    X['Date'] = df.apply(lambda x : int(x[8 : ]) ).values
    return pd.DataFrame(X).values

def apply_regression_model(X,y):
    clf=RandomForestRegressor(n_estimators=10)
    clf.fit(X,y)
    return clf

def predict_y(clf,X_test):
    y_pred=clf.predict(X_test)
    return np.round(y_pred)

@app.route("/predict-sales",methods=['POST'])
def post_predict_sales():
    data = request.get_json()
    training_data  =data['training_data']
    testing_data = data['testing_data']
    df = group_data(training_data)
    X = clean_X_data(df)
    y = df.values
    clf = apply_regression_model(X,y)
    X_test = clean_X_testing_data(testing_data['Date'])
    y_pred = predict_y(clf,X_test)
    return jsonify({
        'result' : list(y_pred)
    })

     






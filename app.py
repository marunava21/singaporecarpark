from flask import Flask, render_template, request
from test import model
import requests
import json
import pandas as pd
import numpy as np
import math
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics
import pandas as pd
from train import preproc
from test import model, car_park_data, car_park_name, all_car_parks
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route("/")
def main():
    clf = pickle.load(open('model.pkl','rb'))
    map = all_car_parks(clf)
    return render_template('index.html', data=map._repr_html_())


@app.route('/predict', methods=['POST'])
def home():
    pincode = request.form['pincode']
    day = request.form['day']
    hour = request.form['hour']

    clf = pickle.load(open('model.pkl','rb'))
    mms1 = pickle.load(open('mms1.pkl','rb'))
    mms2 = pickle.load(open('mms2.pkl','rb'))
    [pred, names, fig] = model(clf, mms1, mms2,pincode, day, hour)
    # result = pred.result()
    return render_template('after.html', data=[names, pred._repr_html_(), fig._repr_html_(), pincode])
    


if __name__=='__main__':
    app.run(debug=True)
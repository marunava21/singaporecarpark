import requests
import json
import pandas as pd
import numpy as np
import math
import csv
import os
# import folium

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics
import pandas as pd
import warnings
import pickle
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

weather_dist = {}
lot_type_dist = {}
car_no_dist = {}
path= os.getcwd()

def preproc(filePath, train =True):
    df1 = pd.read_csv(filePath)
    df2 = df1.copy()
    # find the ratio between available and total car park
    df1['ratio'] = df1['lots_available']/df1['total_lots']
    df1 = df1[['carpark_number','ratio','day_type','weather','lot_type','day','hour']]
    df1=df1[df1["lot_type"]=="C"]
    if train == True:
        # Converting the data to category type for training
        df1['weather']=df1["weather"].astype('category')
        df1['lot_type']=df1["lot_type"].astype('category')
        df1['carpark_category']=df1["carpark_number"].astype('category')

        # saving the label for adjusting in test data
        weather_dist = dict(zip(df1['weather'],df1['weather'].cat.codes))
        lot_type_dist = dict(zip(df1['lot_type'],df1['lot_type'].cat.codes))
        car_no_dist = dict(zip(df1['carpark_category'],df1['carpark_category'].cat.codes))

        # Encoding the data for training
        df1["weather"] = df1["weather"].cat.codes
        df1["lot_type"] = df1["lot_type"].cat.codes
        df1["carpark_category"] = df1["carpark_category"].cat.codes

        w = csv.writer(open(path+"\\csvData\\"+"weather.csv", "w"))
        w1 = csv.writer(open(path+"\\csvData\\"+"lotType.csv", "w"))
        w2 = csv.writer(open(path+"\\csvData\\"+"carno.csv", "w"))
        for k, v in weather_dist.items():
            w.writerow([k,v])
        for k, v in lot_type_dist.items():
            w1.writerow([k,v])
        for k, v in car_no_dist.items():
            w2.writerow([k,v])
        # *csv.DictWriter()

    else:
        # Encoding the data for testing based on the category of train dataset
        w =pd.read_csv(path+"\\csvData\\"+'weather.csv', names=["weather", "value"])
        w= w.set_index("weather").T.to_dict('list')
        w1 =pd.read_csv(path+"\\csvData\\"+'lotType.csv', names=["weather", "value"])
        w1 = w1.set_index("weather").T.to_dict('list')
        w2 =pd.read_csv(path+"\\csvData\\"+'carno.csv', names=["weather", "value"])
        w2 = w2.set_index("weather").T.to_dict('list')
        df1["weather"] = df1.weather.map(lambda x:w[x][0] if x in list(w.keys()) else -1)
        df1["lot_type"] = df1.lot_type.map(lambda x:w1[x][0] if x in list(w1.keys()) else -1)
        df1["carpark_category"] = df1.carpark_number.map(lambda x:w2[x][0] if x in list(w2.keys()) else -1)

    # Chceking Null values in ratio and removing them
    bool_series = pd.isnull(df1["ratio"])
    df1[bool_series]
    df1 = df1[pd.notnull(df1['ratio'])]
    df1.isnull().any()

    # Final data needed 
    df1 = df1[['ratio','carpark_number','day_type','weather','hour', 'carpark_category', 'day']]
    return df1

def train(X, y):
    clf = DecisionTreeRegressor(max_depth=100, criterion='squared_error')
    clf.fit(X, y)
    pickle.dump(clf, open(path+"\\models\\"+"model.pkl", "wb"))
if __name__=="__main__":
    
    mms1 = MinMaxScaler()
    mms2 = MinMaxScaler()
    filepath = 'final_train_data.csv'
    df1 = preproc(path+"\\csvData\\"+filepath)
    X = df1.drop(["carpark_number"], axis = 1)
    df = pd.DataFrame(X, columns=['ratio','day_type','weather','hour', 'carpark_category','day'])
    X_train, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train = mms1.fit_transform(X_train)
    y_train = mms2.fit_transform(np.array(y).reshape(-1,1))
    pickle.dump(mms1, open(path+"\\models\\"+"mms1.pkl", "wb"))
    pickle.dump(mms2, open(path+"\\models\\"+"mms2.pkl", "wb"))
    # print(X_train)
    train(X_train, y_train)

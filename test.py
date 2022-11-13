import requests
import json
import pandas as pd
import numpy as np
import math
import pickle
import folium
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics
import pandas as pd
from train import preproc
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import plotly
import plotly.graph_objs as go
import pandas as pd

path= os.getcwd()
df2 = pd.read_csv(path+"\\csvData\\"+'lat_long_car_park1.csv')
dict_val = df2.set_index("carpark_number").T.to_dict()


def car_park_name(pincode):
    site = f'https://developers.onemap.sg/commonapi/search?searchVal='+str(pincode)+'&returnGeom=Y&getAddrDetails=N&pageNum=1'
    response_API = requests.get(site)
    data = response_API.text
    # loading the data in json
    data = json.loads(data)
    # Getting Lattitude and longitude 
    lat = float(data["results"][0]["LATITUDE"])
    long = float(data["results"][0]["LONGITUDE"])
    lat_r = np.radians(lat)
    long_r = np.radians(long)
    # finding the nearby carpark
    dist = sklearn.metrics.DistanceMetric.get_metric('haversine')
    dist_matrix = (dist.pairwise(df2[['lat_rad','long_rad']],[[lat_r,long_r]])*3959)
    df_dist_matrix = (pd.DataFrame(dist_matrix,index=df2['carpark_number']))
    car_park_name = df_dist_matrix[0].nsmallest(5).index.values
    return car_park_name,lat,long

def car_park_data(car_name,day,hour,lot_type):
    
    # get weather data
    site = f'https://api.data.gov.sg/v1/environment/24-hour-weather-forecast?date_time={"2022"}-{"11".zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}%3A{"30".zfill(2)}%3A{"0".zfill(2)}'
    response_API = requests.get(site)
    data = response_API.text
    # loading the data in json
    data = json.loads(data)
    day_val = pd.to_datetime(data["items"][0]["timestamp"]).day_name() 
    data = data["items"][0]["periods"]
    list_pred =[]
    if ((day_val == "Sunday") or(day_val == "Saturday")):
        day_type =1
    else:
        day_type = 0
    w =pd.read_csv(path+"\\csvData\\"+'weather.csv', names=["weather", "value"])
    w= w.set_index("weather").T.to_dict('list')
    w1 =pd.read_csv(path+"\\csvData\\"+'lotType.csv', names=["weather", "value"])
    w1 = w1.set_index("weather").T.to_dict('list')
    w2 =pd.read_csv(path+"\\csvData\\"+'carno.csv', names=["weather", "value"])
    w2 = w2.set_index("weather").T.to_dict('list')
    for c in car_name:
        if int(hour) <6:
            weather = data[0]['regions'][dict_val[c]['Region'].lower()]
        elif int(hour) < 12:
            weather = data[1]['regions'][dict_val[c]['Region'].lower()]
        elif int(hour) <18:
            weather = data[2]['regions'][dict_val[c]['Region'].lower()]
        else:
            weather = data[3]['regions'][dict_val[c]['Region'].lower()]
        weather_val = w[weather][0]
        car_val = w2[c][0]
        lot_val = w1[lot_type][0]
        list_pred.append([day_type,weather_val,int(hour), car_val, int(day)])
    return list_pred



def all_car_parks(clf):
    # name,lat,long = car_park_name(138639)
    timestamp = pd.Timestamp.now()
    day = timestamp.day
    site = f'https://api.data.gov.sg/v1/transport/carpark-availability?date_time={"2022"}-{str(timestamp.month).zfill(2)}-{str(timestamp.day).zfill(2)}T{str(timestamp.hour).zfill(2)}%3A{str(timestamp.minute).zfill(2)}%3A{str(timestamp.second).zfill(2)}'
    response_API = requests.get(site)
    data = response_API.text
    data = json.loads(data)
    try:
        timestamp = data["items"][0]["timestamp"]
        print("Got data at timestamp : ",timestamp)
    except KeyError as e:
        print("inside Key error")
        print(data)
        print("Trying again to get data")
    # Separate parking data from the dataset 
    data = data["items"][0]["carpark_data"]
    df = pd.DataFrame(data)
    # separating the values to make a complete dataset
    for heading in ("total_lots","lot_type","lots_available"):
        df[heading] = df["carpark_info"].apply(lambda x: x[0][heading])
    df = df.drop(["carpark_info"], axis=1)
    # df = df[df["lots_type"]=="C"]
    map = folium.Map(width=850,height=550,location=[1.3521,103.819],zoom_start=12,min_zoom=8,max_zoom=24,tiles ='Stamen Terrain',control_scale=True)
    for index, location_info in df.iterrows():
        if location_info.carpark_number in dict_val.keys():
            # print(dict_val[location_info.carpark_number]['lat'])
            folium.Marker([dict_val[location_info.carpark_number]['lat'], dict_val[location_info.carpark_number]["lon"]], popup=(str(f'{location_info["carpark_number"]} Available slots = ')+str(int(location_info['lots_available'])))).add_to(map)
    return map



def model(clf, mms1,mms2, pincode, day, hour):
    name,lat,long = car_park_name(pincode)

    # day here given is 4-october-2022
    list_pred = car_park_data(name,day,hour,'C')
    # Reshaping the array to send it for prediction
    list_pred = np.array(list_pred).reshape(-1,5)
    list_pred = mms1.transform(list_pred)
    y_pred=clf.predict(pd.DataFrame(list_pred))
    y_pred = np.ravel(mms2.inverse_transform(np.array(y_pred).reshape(-1,1)))

    # creating a dataframe with all values
    df_final = df2.loc[df2["carpark_number"].isin(name)].copy()
    y_ratio =[]
    # rearranging the ratio for each car park
    for index, val in df_final.iterrows():
        ind = np.where(name == val['carpark_number'])
        y_ratio.append(y_pred[ind[0][0]])
    # adding ratio and available slots for each car_park 
    df_final['ratio'] = y_ratio
    df_final["avail"] = df_final['ratio']*df_final["total_lots"]
    # plt.bar(df_final["carpark_number"], df_final["avail"])
    # plt.savefig("C:\\Users\\Arunava\\Documents\\StudyDoc\\EE4211\\Project\\templates\\map.jpg")
    data = [go.Bar(
        x = df_final['carpark_number'],
        y = df_final['avail'],
    )]
    layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color = "white",
            ),   
            xaxis=dict(
                title='Carpark Numbers',    
                
            ),
            yaxis=dict(
                title='Available Car Parks',  
            )
        )
    fig = go.Figure(data=data, layout=layout)
    map = folium.Map(width=850,height=550,location=[lat,long],zoom_start=16,min_zoom=8,max_zoom=24,tiles ='Stamen Terrain',control_scale=True)
    for index, location_info in df_final.iterrows():
        folium.Marker([location_info["lat"], location_info["lon"]], popup=(str(f'{location_info["carpark_number"]} Available slots = ')+str(int(location_info['avail'])))).add_to(map)
    folium.Marker([lat, long], popup=("location entered "),icon=folium.Icon(color='red')).add_to(map)
    return [map, name, fig]

if __name__=="__main__":
    clf = pickle.load(open('models//model.pkl','rb'))
    mms1 = pickle.load(open('models//mms1.pkl','rb'))
    model(clf,mms1, "138639", "10", "4")

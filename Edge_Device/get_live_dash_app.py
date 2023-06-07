import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
# import create subplots of plotly
import plotly.subplots as subplots

import pandas as pd
import os
import pysftp
import stat 

from Darts.get_solar_panel import SolarPanel

import datetime
from importlib import reload 
# import Forecast_Models
# reload(Forecast_Models)
# from Forecast_Models.RandomForest import RandomForest_Forecast_Refridge
# reload(RandomForest_Forecast_Refridge)
# from RandomForest import RandomForest_Forecast_Refridge

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel, NaiveSeasonal, NaiveDrift, ExponentialSmoothing
from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
from darts.metrics import mape
from darts.datasets import AirPassengersDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
from darts.utils.missing_values import fill_missing_values
import darts

# Pandas settings
pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

class Darts_RandomForest():
    def __init__(self, solar_data):
        print('RandomForestClass: init')
        data_solar = solar_data
        data_solar = data_solar.rename(columns={'date': 'time', 'value': 'SolarPower[W]'})
        data_solar.index = pd.to_datetime(data_solar['time'])
        data_solar_interp = data_solar.interpolate(method='time')
        df_solar_resamp = data_solar_interp.resample('900S').mean()
        print('RandomForestClass: resampled')
        # Drop NaN values by interpolating
        df_solar_resamp = df_solar_resamp.interpolate(method='time')
        df_solar_resamp = df_solar_resamp.resample('900S').mean()
        # Drop NaN values
        df_solar_resamp = df_solar_resamp.dropna()
        df_solar_resamp = df_solar_resamp.drop(columns=['Unnamed: 0'], axis=1)
        # df_solar_resamp = df_solar_resamp.clip(lower=0.1)
        print('RandomForestClass: NaN values dropped')
        ts_solar_df = TimeSeries.from_dataframe(df_solar_resamp, freq='900S')
        self.ts_solar_df = self.fill_missing_vals(ts_solar_df)
        self.train, self.val = self.train_test_split()
        # self.prediction = self.train_model()

    def fill_missing_vals(self, ts_solar_df):
        ts_solar_df = fill_missing_values(ts_solar_df, "auto")
        print('RandomForestClass: missing values filled')
        return ts_solar_df
    
    def train_test_split(self):
        # Split the data into training and validation sets
        # Get latest time of self.ts_solar_df
        latest_time = self.ts_solar_df.end_time()
        train, val = self.ts_solar_df.split_after(latest_time)
        return train, val
    
    def train_model(self):
        # lags = 200 werkt goed
        model = darts.models.RandomForest(lags=270)
        model.fit(self.train)
        prediction = model.predict(100)
        print('model forecast done')
        return prediction

class RandomForest_Forecast_Refridge():
    def __init__(self, data_ins, data_outs):
        print('RandomForestClass Fridge: init')
        # combine inside and outside data and interpolate based on time of inside
        # data_outs = pd.merge(data_outs, data_solar, on='time', how='outer')
        data = pd.merge(data_ins, data_outs, on='time', how='outer')
        # print(data)
        # Converting the index to DatetimeIndex
        data.index = pd.to_datetime(data['time'])
        data_interp = data.interpolate(method='time')
        # Get Out NaNs
        data_interp = data_interp.dropna()
        # data_interp = data_interp.drop(columns=['humidity_x', 'pressure_x', 'current', 'voltage', 'humidity_y', 'pressure_y', 'Live_Irms', 'temperature_y'])
        # create a TimeSeries object from the dataframe
        # resample the data to a 10-second frequency
        # df_resampled = data_interp.resample('900S').mean()
        # Drop NaN values by interpolating
        df_resampled = data_interp.interpolate(method='time')
        df_resampled = data_interp.resample('300S').mean()
        # Drop NaN values
        df_resampled = df_resampled.dropna()
        # print the TimeSeries object
        # print(series)
        # df_resampled['temperature_x']

        ts_df = TimeSeries.from_dataframe(df_resampled, freq='300S')
        self.ts_df = self.fill_missing_vals(ts_df)
        # self.train, self.val = self.train_test_split()
        self.train = self.ts_df

    def fill_missing_vals(self, ts_df):
        ts_df = fill_missing_values(ts_df, "auto")
        print('RandomForestClass Refridge: missing values filled')
        return ts_df

    # def train_test_split(self):
    #     # Split the data into training and validation sets
    #     # Get latest time of self.ts_solar_df
    #     latest_time = self.ts_df.end_time()
    #     train, val = self.ts_df.split_after(latest_time)
    #     return train, val
    
    def train_model(self):
        print('start training')
        model = darts.models.RandomForest(lags=100)
        model.fit(self.train)
        prediction = model.predict(100)
        print('model forecast done')
        return prediction

class DashApp():
    def __init__(self):
        self.solar_prediction = []
        # self.fridge_prediction = []
        self.get_data_sftp()
        self.make_app()

    def get_data_sftp(self):
        host = 'localhost'
        port = 8888
        username = 'tom'
        password= 'PiPyTom'
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None 
        with pysftp.Connection(host, username=username, password=password, port=port, cnopts=cnopts) as sftp:
            print("Connection succesfully stablished ... ")
            sftp.cwd('Code/Logged_Data/')

            # Get a list of all files and folders in directory 
            # Get a list of all folders in the directory
            folder_list = [f for f in sftp.listdir() if sftp.isdir(f)]

            # Find the folder with the latest modification time
            latest_folder = max(folder_list, key=lambda f: sftp.stat(f).st_mtime)

            # Print the latest folder name and modification time
            print("Latest folder:", latest_folder)
            print("Modification time:", sftp.stat(latest_folder).st_mtime)

            # make dir of latest folder in Logged_Data folder on local device
            try:
                os.mkdir('Logged_Data/' + latest_folder)
            except OSError:
                print("Creation of the directory %s failed, dir probably already made" % latest_folder)

            # Download all files in latest dir 
            # sftp.get_d(latest_folder, 'Logged_Data/' + latest_folder)
            # Download specific file in folder
            # latest_folder = '2023_5_15_14_10_25'
            sftp.get(latest_folder + '/data_inside.csv', 'Logged_Data/' + latest_folder + '/data_inside.csv')
            sftp.get(latest_folder + '/data_outside.csv', 'Logged_Data/' + latest_folder + '/data_outside.csv')

            print("Files downloaded successfully")

            # Define the file that you want to download from the remote directory
            # remoteFilePath = '/home/tom/Logged_Data/2021-09-01_13-26-57/data_inside_filtered.csv'
            # localFilePath = '/home/pi/Desktop/Logged_Data/2021-09-01_13-26-57/data_inside_filtered.csv'
            # # Download the file from the remote directory
            # sftp.get(remoteFilePath, localFilePath)
            # print("File downloaded successfully")

    # def get_ip_address(self):
    #     import socket
    #     # get ip adress of device
    #     ip_address = socket.gethostbyname(socket.gethostname())
    #     print(ip_address)
    #     return ip_address

    def make_app(self):
        # create dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = html.Div([
            dbc.Navbar(
                children=[
                    dbc.NavbarBrand(" DT Home Refrigerator", className="ml-2"),
                    dbc.Button("Update Data", color="primary", id='button', n_clicks=0, className="mr-2"),
                    dbc.Button("Retrain Prediction", color="primary", id='retrain', className="mr-2"),
                    dbc.Button("Update Predictions", color="primary", id='update_predictions', className="mr-2"),
                    # Add button to update Graph
                    dbc.Button("Update Graph", color="primary", id='update_graph', className="mr-2"),
                    # dbc.NavItem("Cooling: On", className="mr-2")
                ],
                color="dark",
                dark=True
            ),

            dcc.Loading(
                id="loading-icon",
                type="circle",
                children=[
                    html.Div(id="output")
                ]
            ), 
            # TODO: remove if navbar works
            # html.H1("DT Home Refrigerator Live Dashboard"),
            # Make button to update data
            # dbc.Button('Update Data', id='button', n_clicks=0, color="primary", className="mr-2"),
            # Make text to return if data is updated 
            html.Div(id='output-state'), 
            dcc.Graph(id="live-graph", animate=True, style={'height': '80vh', 'width': '96vw'}),
            dcc.Interval(
                id='graph-update',
                interval=20 * 1000,
                n_intervals=0,
                disabled=False
            ),
            html.H5("Newest Data at: None", id='NewDataTest'), 
            html.H5("Trained Forecast Data till: None", id='PredictedCooling'),
            html.H5("Predicted Data at: None", id='PredictedPower'),
        ])

        # make callback for button to update data
        @app.callback(Output('output-state', 'children'),
                    [Input('button', 'n_clicks')])
        def update_data(n_clicks):
            if n_clicks > 0:
                self.get_data_sftp()
                # get current time 
                now = datetime.datetime.now()
                return 'Data Updated at: ' + now.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return 'No Data Updated'

        # make callback for button to retrain model
        @app.callback(Output('PredictedCooling', 'children'),
                    [Input('retrain', 'n_clicks')])
        def retrain_model(n_clicks):
            try:
                if n_clicks > 0:
                    latest_folder_name = max([f.path for f in os.scandir('Logged_Data') if f.is_dir()], key=os.path.getctime) 
                    print('Retraining on data from: ', latest_folder_name)
                    df_solar = pd.read_csv(latest_folder_name + '/solar_data.csv')
                    print('Read solar data')
                    # df_inside = pd.read_csv(latest_folder_name + '/data_inside.csv')
                    # df_outside = pd.read_csv(latest_folder_name + '/data_outside.csv')
                    random_forest_model = Darts_RandomForest(solar_data=df_solar)
                    print('Created model')
                    forecast = random_forest_model.train_model()
                    # forecast_df = forecast.pd_dataframe()
                    forecast.to_csv(latest_folder_name + '/forecast_data.csv')
                    print('csv_exported')
                    # prediction = RandomForest_Forecast_Refridge(data_ins=pd.read_csv(latest_folder_name + '/data_inside.csv'), data_outs=pd.read_csv(latest_folder_name + '/data_outside.csv')).train_model()
                    # # random_forest_firdge_model = RandomForest_Forecast_Refridge(data_ins=pd.read_csv(latest_folder_name + '/data_inside.csv'), data_outs=pd.read_csv(latest_folder_name + '/data_outside.csv'))
                    # # forecast_fridge = random_forest_firdge_model.train_model()
                    # prediction.to_csv(latest_folder_name + '/forecast_data_fridge.csv')
                    # print('forecast: ', forecast)
                    # get latest time of df_solar
                    latest_time = df_solar['date'].iloc[-1]
                    print('Retrained Forecast Data till: ', latest_time)
                    return 'Retrained Forecast Data till: ' + latest_time
                else:
                    return 'No Retraining'
            except TypeError:
                return 'No Retraining (Dash just loaded)'
            
        # make callback for button to update predictions
        @app.callback(Output('PredictedPower', 'children'),
                    [Input('update_predictions', 'n_clicks')])
        def update_predictions(n_clicks):
            try:
                if n_clicks > 0:
                    print('updating forecast data')
                    latest_folder_name = max([f.path for f in os.scandir('Logged_Data') if f.is_dir()], key=os.path.getctime) 
                    try:
                        self.solar_prediction = pd.read_csv(latest_folder_name + '/forecast_data.csv')
                        latest_time = self.solar_prediction['time'].iloc[-1]
                        print('updated forecast data')
                        self.fridge_prediction = pd.read_csv(latest_folder_name + '/prediction_fridge.csv')
                        # take sqrt of Irms
                        self.fridge_prediction['Irms'] = self.fridge_prediction['Irms'].apply(lambda x: x**0.004)                
                        # take average of Irms data
                        Irms_avg = self.fridge_prediction['Irms'].mean()
                        self.fridge_prediction['Irms'] = self.fridge_prediction['Irms'].apply(lambda x: x - Irms_avg)
                        # if Irms is negative, set to 0
                        self.fridge_prediction['Irms'] = self.fridge_prediction['Irms'].apply(lambda x: 0 if x < 0 else x)
                        # to get watt, multiply by 230 and set Irms correctly
                        self.fridge_prediction['Irms'] = self.fridge_prediction['Irms'] * 150
                        self.fridge_prediction['Watt'] = self.fridge_prediction['Irms'] * 230
                        print('updated fridge forecast data')
                        return 'Predictions Updated till: ' + latest_time
                    except:
                        print('No forecast data found')
                        return 'No Prediction Update (no forecast data found)'
                    # get latest time 
                    
                else:
                    return 'No Prediction Update (model not retrained yet)'
            except TypeError:
                return 'No Prediction Update (Dash just builded)'

        # create a function to update the graph
        @app.callback([Output('live-graph', 'figure'), 
                    Output('NewDataTest', 'children'), 
                    Output("output", "children")],
                    [Input('update_graph', 'n_clicks')])
        def update_graph_scatter(n):
            # Get latest folder name in folder 'Logged_Data'
            latest_folder_name = max([f.path for f in os.scandir('Logged_Data') if f.is_dir()], key=os.path.getctime) 
            print(latest_folder_name)
            print("Watt plotted")
            # latest_folder_name = 'Logged_Data/2023_4_30_16_26_42'
            # open filtered csv files
            data_inside_filtered = pd.read_csv(latest_folder_name + "/data_inside.csv")
            data_outside_filtered = pd.read_csv(latest_folder_name + "/data_outside.csv")
            # convert time column to datetime
            data_inside_filtered['time'] = pd.to_datetime(data_inside_filtered['time'])
            data_outside_filtered['time'] = pd.to_datetime(data_outside_filtered['time'])
            # take sqrt of Irms
            data_outside_filtered['Irms'] = data_outside_filtered['Irms'].apply(lambda x: x**0.004)
    
            # take average of Irms data
            Irms_avg = data_outside_filtered['Irms'].mean()
            data_outside_filtered['Irms'] = data_outside_filtered['Irms'].apply(lambda x: x - Irms_avg)
            # if Irms is negative, set to 0
            data_outside_filtered['Irms'] = data_outside_filtered['Irms'].apply(lambda x: 0 if x < 0 else x)
            # to get watt, multiply by 230 and set Irms correctly
            data_outside_filtered['Irms'] = data_outside_filtered['Irms'] * 150
            data_outside_filtered['Watt'] = data_outside_filtered['Irms'] * 230

            # get start and end time of data
            start_time_time = str(data_inside_filtered['time'].iloc[0])
            end_time_time = str(data_inside_filtered['time'].iloc[-1])
            start_time = start_time_time.split(' ')[0]
            end_time = end_time_time.split(' ')[0]
            print(start_time, end_time)

            # get energy price data
            energy_price_last_30_days = pd.read_csv('EnergyPrices/11May_Energy_Prices_last30days.csv')
            energy_price_last_30_days['date'] = pd.to_datetime(energy_price_last_30_days['date'])
            energy_price_last_30_days = energy_price_last_30_days[(energy_price_last_30_days['date'] >= start_time_time) & (energy_price_last_30_days['date'] <= end_time_time)]
            # get solar energy data
            solar_panel = SolarPanel(start_time, end_time, power_or_energy='power')
            df_solar = solar_panel.get_df()
            # convert to datetime
            df_solar['date'] = pd.to_datetime(df_solar['date'])
            # filter to only between start and end time time
            df_solar = df_solar[(df_solar['date'] >= start_time_time) & (df_solar['date'] <= end_time_time)]
            df_solar.to_csv(latest_folder_name + '/solar_data.csv')
            print("Exported solar data")
            # data_outside_filtered.to_csv(latest_folder_name + '/data_outside.csv')
            # print("Exported outside data")
            # data_inside_filtered.to_csv(latest_folder_name + '/data_inside.csv')
            # print("Exported inside data")
            # create a figure with subplots
            fig = subplots.make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=["Solar Energy [W]", "Watt [W]", "Humidity", "Fridge Temp [C]"])
            # fig = go.Figure()
            # add traces
            fig.add_trace(go.Scatter(x=data_inside_filtered['time'], y=data_inside_filtered['temperature'], mode='lines', name='temperature inside'), row=2, col=2)
            fig.add_trace(go.Scatter(x=data_outside_filtered['time'], y=data_outside_filtered['Watt'], mode='lines', name='Watt'), row=1, col=2)
            fig.add_trace(go.Scatter(x=data_outside_filtered['time'], y=data_outside_filtered['temperature'], mode='lines', name='temperature outside'), row=2, col=1)
            fig.add_trace(go.Scatter(x=data_outside_filtered['time'], y=data_outside_filtered['humidity'], mode='lines', name='humidity outside'), row=2, col=1)
            # fig.add_trace(go.Scatter(x=data_inside_filtered['time'], y=data_inside_filtered['pressure'], mode='lines', name='pressure inside'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data_inside_filtered['time'], y=data_inside_filtered['humidity'], mode='lines', name='humidity inside'), row=2, col=1)
            # add presure outside
            # fig.add_trace(go.Scatter(x=data_outside_filtered['time'], y=data_outside_filtered['pressure'], mode='lines', name='pressure outside'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_solar['date'], y=df_solar['value'], mode='lines', name='solar energy'), row=1, col=1)
            fig.add_trace(go.Scatter(x=energy_price_last_30_days['date'], y=energy_price_last_30_days['prijsAIP'], mode='lines', name='energy price AIP'), row=1, col=1)
            
            if len(self.solar_prediction) > 0:
                fig.add_trace(go.Scatter(x=self.solar_prediction['time'], y=self.solar_prediction['SolarPower[W]'], mode='lines', name='solar prediction[W]'), row=1, col=1)
                print("Solar prediction added")
                fig.add_trace(go.Scatter(x=self.fridge_prediction['time'], y=self.fridge_prediction['temperature_x'], mode='lines', name='temp inside prediction'), row=2, col=2)
                
                fig.add_trace(go.Scatter(x=self.fridge_prediction['time'], y=self.fridge_prediction['Watt'], mode='lines', name='Watt prediction'), row=1, col=2)
            return fig, 'Newest data at (W): ' + str(data_outside_filtered['time'].iloc[-1]), "Loading Completed"

        app.run_server(port=8060, debug=True)

if __name__ == "__main__":
    dash_app = DashApp()
    
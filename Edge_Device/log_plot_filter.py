import pandas as pd
import plotly 
from multiprocessing import Process
import serial
import serial.tools.list_ports
import datetime
import os
import threading
import time
import sys
import glob
import csv
from test_serial_controller import SerialController

# imports for dash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from get_solar_live import SolarPanel_Live

class logger_controller:
    def __init__(self):
        self.serial_controller = SerialController('/dev/M5Inside', 115200)
        self.folder_name = self.make_folder_of_today()
        p1 = Process(target=self.logger_inside)
        p1.start()
        p2 = Process(target=self.logger_outside)
        p2.start()
        self.start_time_inside = datetime.datetime.now()
        self.start_time_outside = datetime.datetime.now()
        # p3 = Process(target=self.filter_files_live)
        # p3.start()
        # p4 = Process(target=self.plot_data_dash)
        # p4.start()

    def make_folder_of_today(self):
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        second = now.second
        folder_name = str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(second)
        folder_name = os.getcwd() + "/Logged_Data/" + folder_name
        # use try and except to avoid error when the folder already exists
        try:
            os.mkdir(folder_name)
            return folder_name  

        except OSError:
            print ("Creation of the directory %s failed" % folder_name)
            return False
        
    def logger_inside(self):
        # create a csv file to store the data
        file_name = self.folder_name + "/data_inside.csv"
        with open(file_name, 'w') as csvfile:
            fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage', 'command', 'solar_power']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        file_name_raw = self.folder_name + "/data_inside_raw.csv"
        with open(file_name_raw, 'w') as csvfile_raw:
            fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage']
            writer = csv.DictWriter(csvfile_raw, fieldnames=fieldnames)
            writer.writeheader()
        # create a serial object
        ser = serial.Serial('/dev/M5Inside', 115200)
        ser.flush()
        last_command = "Setup"
        # Lists for averaging
        temperature_list = []
        humidity_list = []
        pressure_list = []
        current_list = []
        voltage_list = []

        # start logging
        while True:
            if ser.in_waiting > 0:
                try: 
                    line = ser.readline().decode('utf-8').rstrip()
                    print("inside: ", line)
                    line_splitted = line.split(",")
                    with open(file_name_raw, 'a') as csvfile_raw:
                        fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage']
                        writer = csv.DictWriter(csvfile_raw, fieldnames=fieldnames)
                        try:
                            writer.writerow({'time': datetime.datetime.now(), 'temperature': line_splitted[0], 'humidity': line_splitted[1], 'pressure': line_splitted[2], 'current': line_splitted[3], 'voltage': line_splitted[4]})
                            temperature_list.append(float(line_splitted[0])), humidity_list.append(float(line_splitted[1])), pressure_list.append(float(line_splitted[2])), current_list.append(float(line_splitted[3])), voltage_list.append(float(line_splitted[4]))
                        except IndexError: 
                            print("Line error inside") 

                    if len(temperature_list) > 20:
                        # Take average of the last 20 values
                        temperature = sum(temperature_list) / len(temperature_list)
                        humidity = sum(humidity_list) / len(humidity_list)
                        pressure = sum(pressure_list) / len(pressure_list)
                        current = sum(current_list) / len(current_list)
                        voltage = sum(voltage_list) / len(voltage_list)
                        # Write the average values to the file
                        with open(file_name, 'a') as csvfile:
                            fieldnames = ["time", "temperature", "humidity", "pressure", "current", "voltage", "command", 'solar_power']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'time': datetime.datetime.now(), 'temperature': temperature, 'humidity': humidity, 'pressure': pressure, 'current': current, 'voltage': voltage, 'command': last_command, 'solar_power': SolarPanel_Live().get_data()})
                        # Empty the lists   
                        temperature_list = []
                        humidity_list = []
                        pressure_list = []
                        current_list = []
                        voltage_list = []
                        try:
                            print("Solar Panel Live: ", SolarPanel_Live().get_data())
                            if float(temperature) > 0 and last_command != "on" and SolarPanel_Live().get_data() > 20: 
                                self.serial_controller.write("on")
                                last_command = "on"
                            elif float(temperature) < 0 and last_command != "off": 
                                self.serial_controller.write("off")
                                last_command = "off"
                            elif float(temperature) > 10 and last_command != "on": 
                                self.serial_controller.write("on")
                                last_command = "on"
                        except ValueError:
                            print("Command Printed") 
                except:
                    print("Serial Exception Inside")
        #         except serial.serialutil.SerialException:
        #             print("Serial Exception Inside")
        # # close the serial port
        ser.close()

    def logger_outside(self):
        # create a csv file to store the data
        file_name = self.folder_name + "/data_outside.csv"
        with open(file_name, 'w') as csvfile:
            fieldnames = ["time","temperature","humidity","pressure","Live_Irms","Irms"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        file_name_raw = self.folder_name + "/data_outside_raw.csv"
        with open(file_name_raw, 'w') as csvfile_raw:
            fieldnames = ["time","temperature","humidity","pressure","Live_Irms","Irms"]
            writer = csv.DictWriter(csvfile_raw, fieldnames=fieldnames)
            writer.writeheader()

        # create a serial object
        ser = serial.Serial('/dev/M5Outside', 115200)
        ser.flush()
        temperature_list = []
        humidity_list = []
        pressure_list = []
        Live_Irms_list = []
        Irms_list = []
        # start logging
        while True:   
            if ser.in_waiting > 0:
                try: 
                    line = ser.readline().decode('utf-8').rstrip()
                    print("outside: ", line)
                    line_splitted = line.split(",")
                    with open(file_name_raw, 'a') as csvfile_raw:
                        fieldnames = ["time", "temperature", "humidity", "pressure", "Live_Irms", "Irms"]
                        writer = csv.DictWriter(csvfile_raw, fieldnames=fieldnames)
                        try: 
                            writer.writerow({'time': datetime.datetime.now(), 'temperature': line_splitted[0], 'humidity': line_splitted[1], 'pressure': line_splitted[2], 'Live_Irms': line_splitted[3], 'Irms': line_splitted[4]})       
                            temperature_list.append(float(line_splitted[0])), humidity_list.append(float(line_splitted[1])), pressure_list.append(float(line_splitted[2])), Live_Irms_list.append(float(line_splitted[3])), Irms_list.append(float(line_splitted[4]))
                        except IndexError:
                            print("Line error outside")
                    if len(temperature_list) > 20:
                        # Take average of the last 20 values
                        temperature = sum(temperature_list) / len(temperature_list)
                        humidity = sum(humidity_list) / len(humidity_list)
                        pressure = sum(pressure_list) / len(pressure_list)
                        Live_Irms = sum(Live_Irms_list) / len(Live_Irms_list)
                        Irms = sum(Irms_list) / len(Irms_list)
                        # Write the average values to the file
                        with open(file_name, 'a') as csvfile:
                            fieldnames = ["time", "temperature", "humidity", "pressure", "Live_Irms", "Irms"]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'time': datetime.datetime.now(), 'temperature': temperature, 'humidity': humidity, 'pressure': pressure, 'Live_Irms': Live_Irms, 'Irms': Irms})
                        # Empty the lists   
                        temperature_list = []
                        humidity_list = []
                        pressure_list = []
                        Live_Irms_list = []
                        Irms_list = []

                except serial.serialutil.SerialException:
                    print("Serial Exception Outside")
        # close the serial port
        ser.close()

    def keep_raw_make_new(self):
        inside_data = pd.read_csv(self.folder_name + "/data_inside.csv")
        outside_data = pd.read_csv(self.folder_name + "/data_outside.csv")
        inside_data_raw = pd.read_csv(self.folder_name + "/data_inside_raw.csv")
        outside_data_raw = pd.read_csv(self.folder_name + "/data_outside_raw.csv")
        inside_data_raw = pd.concat([inside_data_raw, inside_data])
        outside_data_raw = pd.concat([outside_data_raw, outside_data])
        inside_data_raw.to_csv(self.folder_name + "/data_inside_raw.csv")
        outside_data_raw.to_csv(self.folder_name + "/data_outside_raw.csv")
        # empty the files but keep headers
        outside = self.folder_name + "/data_outside.csv"
        with open(outside, 'w') as csvfile_outside:
            fieldnames = ["time", "temperature", "humidity", "pressure", "Live_Irms", "Irms"]
            writer = csv.DictWriter(csvfile_outside, fieldnames=fieldnames)
            writer.writeheader()
            
        # create a csv file to store the data
        inside = self.folder_name + "/data_inside.csv"
        with open(inside, 'w') as csvfile_inside:
            fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage']
            writer = csv.DictWriter(csvfile_inside, fieldnames=fieldnames)
            writer.writeheader()
        inside_data.to_csv(self.folder_name + "/data_inside.csv")
        outside_data.to_csv(self.folder_name + "/data_outside.csv")
        
    def filter_files_live(self): 
        outside_filtered = self.folder_name + "/data_outside_filtered.csv"
        with open(outside_filtered, 'w') as csvfile_outside:
            fieldnames = ["time", "temperature", "humidity", "pressure", "Live_Irms", "Irms"]
            writer = csv.DictWriter(csvfile_outside, fieldnames=fieldnames)
            writer.writeheader()
            
        # create a csv file to store the data
        inside_filtered = self.folder_name + "/data_inside_filtered.csv"
        with open(inside_filtered, 'w') as csvfile_inside:
            fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage']
            writer = csv.DictWriter(csvfile_inside, fieldnames=fieldnames)
            writer.writeheader()
            
        time.sleep(10)
        filtered_counter = 0 
        while True:
            # open csv files and filter them
            data_inside_unfiltered = pd.read_csv(self.folder_name + "/data_inside.csv")
            data_outside_unfiltered = pd.read_csv(self.folder_name + "/data_outside.csv")

            # convert time column to datetime
            data_inside_unfiltered['time'] = pd.to_datetime(data_inside_unfiltered['time'])
            data_outside_unfiltered['time'] = pd.to_datetime(data_outside_unfiltered['time'])

            self.end_time_inside = self.start_time_inside + datetime.timedelta(seconds=10)
            self.end_time_outside = self.start_time_outside + datetime.timedelta(seconds=10)
            # print(self.start_time_inside, self.end_time_inside) 
			# Get values between start time and end time
            data_inside = data_inside_unfiltered[(data_inside_unfiltered['time'] >= self.start_time_inside) & (data_inside_unfiltered['time'] <= self.end_time_inside)]
            data_outside = data_outside_unfiltered[(data_outside_unfiltered['time'] >= self.start_time_outside) & (data_outside_unfiltered['time'] <= self.end_time_outside)]

            # Get average values of all headers
            data_inside = data_inside.mean(numeric_only=True)
            data_outside = data_outside.mean(numeric_only=True)

            with open(inside_filtered, 'a') as csvfile_inside:
                fieldnames = ['time', 'temperature', 'humidity', 'pressure', 'current', 'voltage']
                writer = csv.DictWriter(csvfile_inside, fieldnames=fieldnames)
                writer.writerow({'time': self.start_time_inside, 'temperature': data_inside['temperature'], 'humidity': data_inside['humidity'], 'pressure': data_inside['pressure'], 'current': data_inside['current'], 'voltage': data_inside['voltage']})
            
            with open(outside_filtered, 'a') as csvfile_outside:
                fieldnames = ["time","temperature","humidity","pressure","Live_Irms","Irms"]
                writer = csv.DictWriter(csvfile_outside, fieldnames=fieldnames)
                writer.writerow({'time': self.start_time_outside, 'temperature': data_outside['temperature'], 'humidity': data_outside['humidity'], 'pressure': data_outside['pressure'], 'Live_Irms': data_outside['Live_Irms'], 'Irms': data_outside['Irms']})      

            print("average inside: ", data_inside)
            print("average outside: ", data_outside)
            self.start_time_inside = self.end_time_inside
            self.start_time_outside = self.end_time_outside
            filtered_counter += 1
            if filtered_counter > 10:
                self.keep_raw_make_new()
                filtered_counter = 0
            time.sleep(10)

    def plot_data_dash(self):
        # create dash app
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Live Data"),
            html.H5("New Data: None", id='NewDataTest'), 
            dcc.Graph(id="live-graph", animate=True),
            dcc.Interval(
                id='graph-update',
                interval=20 * 1000,
                n_intervals=0,
                disabled=False
            ),
        ])

        # create a function to update the graph
        @app.callback([Output('live-graph', 'figure'), 
					Output('NewDataTest', 'children')],
                    [Input('graph-update', 'n_intervals')])
        def update_graph_scatter(n):
            # open filtered csv files
            data_inside_filtered = pd.read_csv(self.folder_name + "/data_inside_filtered.csv")
            data_outside_filtered = pd.read_csv(self.folder_name + "/data_outside_filtered.csv")
            # convert time column to datetime
            data_inside_filtered['time'] = pd.to_datetime(data_inside_filtered['time'])
            data_outside_filtered['time'] = pd.to_datetime(data_outside_filtered['time'])
            # create a figure
            fig = go.Figure()
            # add traces
            fig.add_trace(go.Scatter(x=data_inside_filtered['time'], y=data_inside_filtered['temperature'], mode='lines', name='temperature inside'))
            fig.add_trace(go.Scatter(x=data_outside_filtered['time'], y=data_outside_filtered['Irms'], mode='lines', name='Irms'))

            fig.update_layout(title='Live Data', xaxis_title='time', yaxis_title='values')
            return fig, data_outside_filtered['time'].iloc[-1]
        
        app.run_server(debug=True, use_reloader=True)
        
if __name__ == '__main__':
    logger_controller()
